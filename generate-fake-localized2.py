import os
import random
import sys
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_DIR, CATEGORIES, SEED, N_FOLDS, SAMPLE_MANIFEST_PATH, GENERATED_MANIFEST_PATH, require_empty_outputs

PATCH_FRACTION_RANGE = (0.08, 0.2)   # patch side length as a fraction of image side
ROTATION_ANGLES = list(range(15, 166, 15))
TAMPER_TYPES = ["rotation", "copy_move", "splice"]

# Shared capture variation, applied BEFORE tampering to both classes.
# Scale uses normalized 0..1 intensities: larger values mean lighter noise.
CAPTURE_POISSON_SCALE_RANGE = (600.0, 1800.0)
CAPTURE_BLUR_KERNELS = (3, 5)
CAPTURE_BLUR_SIGMA_RANGE = (0.3, 1.0)
CAPTURE_BRIGHTNESS_ALPHA_RANGE = (0.96, 1.04)
CAPTURE_BRIGHTNESS_BETA_RANGE = (-5.0, 5.0)


def apply_capture_variation(image, source_id, seed):
    """Same deterministic, label-independent treatment for a source and donor.

    Every image gets mild brightness variation, Poisson noise, and ONE local
    blur patch. No whole-image blur is applied. Originals are never mutated.
    Separate RNGs prevent retries or tamper choices changing capture settings.
    """
    key = hashlib.sha256(f"capture:{seed}:{source_id}".encode()).digest()
    rng = random.Random(int.from_bytes(key[:16], 'big'))
    nrng = np.random.default_rng(int.from_bytes(key[16:], 'big'))
    alpha = rng.uniform(*CAPTURE_BRIGHTNESS_ALPHA_RANGE)
    beta = rng.uniform(*CAPTURE_BRIGHTNESS_BETA_RANGE)
    pixels = np.clip(image.astype(np.float32) * alpha + beta, 0, 255)
    scale = rng.uniform(*CAPTURE_POISSON_SCALE_RANGE)
    pixels = np.clip(nrng.poisson(pixels / 255.0 * scale) / scale * 255.0,
                     0, 255).astype(np.uint8)
    h, w = pixels.shape[:2]
    if min(h, w) < 16:
        raise ValueError("Images must be at least 16x16 for localized capture blur")
    fraction = rng.uniform(*PATCH_FRACTION_RANGE)
    ph, pw = max(3, int(h * fraction)), max(3, int(w * fraction))
    y, x = rng.randint(0, h - ph), rng.randint(0, w - pw)
    kernel = rng.choice(CAPTURE_BLUR_KERNELS)
    sigma = rng.uniform(*CAPTURE_BLUR_SIGMA_RANGE)
    patch = pixels[y:y + ph, x:x + pw]
    pixels[y:y + ph, x:x + pw] = cv2.GaussianBlur(patch, (kernel, kernel), sigma)
    return pixels, {
        'capture_noise': 'poisson', 'capture_poisson_scale': scale,
        'capture_brightness_alpha': alpha, 'capture_brightness_beta': beta,
        'capture_blur': 'localized', 'capture_blur_kernel': kernel,
        'capture_blur_sigma': sigma, 'capture_blur_x': x, 'capture_blur_y': y,
        'capture_blur_width': pw, 'capture_blur_height': ph,
        'capture_order': 'before',
    }


EDIT_JPEG_QUALITY_RANGE = (60, 90)
EDIT_ALPHA_RANGE = (0.96, 1.04)
EDIT_BETA_RANGE = (-5, 5)
EDIT_FEATHER_FRACTION = 0.15
GENERATOR_VERSION = 'team_style_three_v1'


def _edit_traces(patch):
    """JPEG recompression and slight brightness changes to the edit only."""
    quality = random.randint(*EDIT_JPEG_QUALITY_RANGE)
    alpha = random.uniform(*EDIT_ALPHA_RANGE)
    beta = random.randint(*EDIT_BETA_RANGE)
    ok, encoded = cv2.imencode('.jpg', patch, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError('Failed to encode edited patch as JPEG')
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != patch.shape:
        raise OSError('Failed to decode edited JPEG patch')
    # Preserve the other team's convertScaleAbs brightness operation.
    edited = cv2.convertScaleAbs(decoded, alpha=alpha, beta=beta)
    return edited, {'edit_jpeg_quality': quality, 'edit_alpha': alpha, 'edit_beta': beta}


def _paste_edit(out, patch, y, x, metadata=None):
    """Feather only the inserted patch. This never blurs the whole image."""
    patch, parameters = _edit_traces(patch)
    ph, pw = patch.shape[:2]
    # Guarantee a nonempty interior even on small images.
    margin = min(max(1, int(min(ph, pw) * EDIT_FEATHER_FRACTION)),
                 (min(ph, pw) - 1) // 2)
    if margin < 1:
        raise ValueError('Edited patch must be at least 3x3')
    mask = np.zeros((ph, pw), np.float32)
    mask[margin:ph-margin, margin:pw-margin] = 1.0
    mask = cv2.GaussianBlur(mask, (2*margin+1, 2*margin+1), 0)[..., None]
    region = out[y:y+ph, x:x+pw].astype(np.float32)
    if region.shape != patch.shape:
        raise ValueError('Edited patch falls outside image')
    out[y:y+ph, x:x+pw] = np.clip(
        mask*patch.astype(np.float32)+(1-mask)*region, 0, 255).astype(np.uint8)
    if metadata is not None:
        metadata.update(parameters)
        metadata.update(edit_feather_fraction=EDIT_FEATHER_FRACTION,
                        requested_edit_x=x, requested_edit_y=y,
                        requested_edit_width=pw, requested_edit_height=ph)
    return out


def _random_patch_size(h, w, fraction_range=PATCH_FRACTION_RANGE):
    frac = random.uniform(*fraction_range)
    ph = max(int(h * frac), 10)
    pw = max(int(w * frac), 10)
    return min(ph, h), min(pw, w)

def copy_move_tamper(image, metadata=None):
    """Copy a nonoverlapping patch, add edit traces, and feather it elsewhere."""
    h, w = image.shape[:2]
    ph, pw = _random_patch_size(h, w)

    sy, sx = random.randint(0, h - ph), random.randint(0, w - pw)
    patch = image[sy:sy + ph, sx:sx + pw].copy()

    for _ in range(100):
        dy, dx = random.randint(0, h - ph), random.randint(0, w - pw)
        if dy + ph <= sy or sy + ph <= dy or dx + pw <= sx or sx + pw <= dx:
            break
    else:
        raise ValueError("Cannot place a non-overlapping copy-move patch")

    tampered = image.copy()
    return _paste_edit(tampered, patch, dy, dx, metadata)


def splice_tamper(image, donor_image, metadata=None):
    """Paste a donor patch with JPEG/brightness traces and feathered edges."""
    h, w = image.shape[:2]
    ph, pw = _random_patch_size(h, w, fraction_range=(0.10, 0.25))

    dh, dw = donor_image.shape[:2]
    sph, spw = min(ph, dh), min(pw, dw)
    sy, sx = random.randint(0, dh - sph), random.randint(0, dw - spw)
    patch = donor_image[sy:sy + sph, sx:sx + spw]
    patch = cv2.resize(patch, (pw, ph))  # match the target patch size exactly

    dy, dx = random.randint(0, h - ph), random.randint(0, w - pw)

    tampered = image.copy()
    return _paste_edit(tampered, patch, dy, dx, metadata)

def rotation_tamper(image, metadata=None):
    """Rotate one random patch in place -- directly shifts the local gradient
    orientations that HOG measures, and creates a boundary mismatch with the
    un-rotated surroundings. (A patch that lands on a flat/blank background
    region can end up looking nearly unchanged -- a known, low-impact edge
    case; exact no-op edits are retried.)"""
    h, w = image.shape[:2]
    ph, pw = _random_patch_size(h, w)
    y, x = random.randint(0, h - ph), random.randint(0, w - pw)

    angle = random.choice(ROTATION_ANGLES)
    patch = image[y:y + ph, x:x + pw]
    rot_matrix = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
    rotated_patch = cv2.warpAffine(patch, rot_matrix, (pw, ph), borderMode=cv2.BORDER_REFLECT)

    tampered = image.copy()
    if metadata is not None:
        metadata['edit_rotation_angle'] = angle
    return _paste_edit(tampered, rotated_patch, y, x, metadata)


def make_fake_version(image, donor_image, tamper_type=None, return_metadata=False):
    edit_metadata = {}
    if tamper_type is None:
        tamper_type = random.choice(TAMPER_TYPES)
    if tamper_type == "copy_move":
        fake_img = copy_move_tamper(image, edit_metadata)
    elif tamper_type == "splice":
        if donor_image is None:
            raise ValueError("Splice requires a same-partition donor")
        fake_img = splice_tamper(image, donor_image, edit_metadata)
    elif tamper_type == "rotation":
        fake_img = rotation_tamper(image, edit_metadata)
    else:
        raise ValueError(f"Unsupported tampering method: {tamper_type}")
    return (fake_img, tamper_type, edit_metadata) if return_metadata else (fake_img, tamper_type)



def prepare_dataset(sample_dir, categories, seed, manifest_path, make_fake):
    random.seed(seed)
    np.random.seed(seed)
    root = Path(sample_dir).resolve()
    source_root = root / 'real'
    output_root = root
    require_empty_outputs(root / 'fake', root / 'augmented_real', manifest_path)
    records = []
    identities = {}
    provenance = {}
    samples = pd.read_csv(SAMPLE_MANIFEST_PATH)
    required = {"filepath", "document_type", "archive_member", "source_file_sha256"}
    if samples.empty or not required.issubset(samples):
        raise ValueError("Run verified sampling first")
    for row in samples.to_dict("records"):
        path = Path(row["filepath"])
        parts = str(row["archive_member"]).replace("\\", "/").lower().split("/")
        if "positive" not in parts or any(p.startswith("fraud") for p in parts):
            raise ValueError(f"Unverified genuine source: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["source_file_sha256"]:
            raise ValueError(f"Sample bytes changed since sampling: {path}")
        category = row["document_type"]
        if category not in categories:
            raise ValueError(f"Unexpected category: {category}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable source: {path}")
        digest = hashlib.sha256(str(image.shape).encode("ascii") + image.tobytes())
        source_id = digest.hexdigest()
        if source_id in identities and identities[source_id][0] != category:
            raise ValueError(f"Identical source pixels appear in multiple categories: {path}")
        identities.setdefault(source_id, (category, path))
        provenance[str(path)] = row
        records.append((category, path, source_id))

    if not records:
        raise ValueError(f'No readable source images in {source_root}')

    rng = random.Random(seed)
    folds = {}
    development_offset = 0
    test_offset = 0
    for category in sorted(categories):
        ids = sorted(k for k, (cat, _) in identities.items() if cat == category)
        rng.shuffle(ids)
        for i, source_id in enumerate(ids):
            # Every fifth distinct source is held out before fake generation.
            # Carry offsets between categories to accommodate small categories.
            if (test_offset + i) % 5 == 0:
                folds[source_id] = -1
            else:
                folds[source_id] = development_offset % N_FOLDS
                development_offset += 1
        test_offset = (test_offset + len(ids)) % 5

    # Restrict splices to the same document type AND saved evaluation fold.
    pools = {
        (category, fold): [sid for sid, (cat, _) in identities.items()
                           if cat == category and folds[sid] == fold]
        for category in categories for fold in [-1, *range(N_FOLDS)]
    }
    for key, pool in pools.items():
        if pool and len(pool) < 2:
            raise ValueError(f"Need at least two distinct sources in {key} for splicing")

    # Balance within category/fold; randomly choose remainder types and shuffle.
    # With 3 styles, counts differ by at most one within each such group.
    grouped = {}
    for index, (category, _, source_id) in enumerate(records):
        grouped.setdefault((category, folds[source_id]), []).append(index)
    schedule = {}
    plan_rng = random.Random(seed + 1)
    category_totals = {cat: {kind: 0 for kind in TAMPER_TYPES} for cat in categories}
    for key in sorted(grouped):
        indices = grouped[key]
        count, remainder = divmod(len(indices), len(TAMPER_TYPES))
        totals = category_totals[key[0]]
        remainder_choices = list(TAMPER_TYPES)
        plan_rng.shuffle(remainder_choices)
        remainder_choices.sort(key=lambda kind: totals[kind])
        plan = TAMPER_TYPES * count + remainder_choices[:remainder]
        for kind in plan:
            totals[kind] += 1
        plan_rng.shuffle(plan)
        schedule.update(zip(indices, plan))

    rows = []
    for index, (category, path, source_id) in enumerate(records):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Input became unreadable: {path}')
        image, capture_metadata = apply_capture_variation(image, source_id, seed)
        fold = folds[source_id]
        selected_type = schedule[index]
        donor_ids = [sid for sid in pools[(category, fold)] if sid != source_id]
        donor_id = ''
        donor = None
        if selected_type == 'splice':
            donor_id = random.choice(donor_ids)
            donor_path = identities[donor_id][1]
            donor_raw = cv2.imread(str(donor_path), cv2.IMREAD_COLOR)
            if donor_raw is None:
                raise ValueError(f'Unreadable donor: {donor_path}')
            donor, _ = apply_capture_variation(donor_raw, donor_id, seed)
        # Retry only exact no-ops; retain subtle edits and report their magnitude.
        for _ in range(10):
            before = image.copy()
            donor_before = donor.copy() if donor is not None else None
            fake, tamper_type, edit_metadata = make_fake(image, donor, selected_type, return_metadata=True)
            if tamper_type != selected_type:
                raise RuntimeError("Tamper callback changed the balanced schedule")
            if not np.array_equal(image, before) or (
                donor is not None and not np.array_equal(donor, donor_before)
            ):
                raise RuntimeError(f"{tamper_type} mutated a source or donor in place")
            changed = np.any(fake != image, axis=2)
            if changed.any():
                break
        else:
            raise ValueError(f'Ten exact no-op edits for {path}; inspect source')

        ys, xs = np.nonzero(changed)
        difference = np.abs(fake.astype(np.float32) - image.astype(np.float32))
        sample_record = provenance[str(path)]
        metadata = {
            'archive_container': sample_record.get('archive_container', ''),
            'archive_member': sample_record['archive_member'],
            'source_file_sha256': sample_record['source_file_sha256'],
            'source_id': source_id,
            'source_filepath': str(path),
            'fold_id': fold,
            'split': 'test' if fold == -1 else 'development',
            'document_type': category,
        }
        # Hash the relative path as well to distinguish duplicate source files.
        path_id = hashlib.sha256(
            path.relative_to(source_root).as_posix().encode('utf-8')
        ).hexdigest()
        for label, kind, pixels in [(0, 'real', image), (1, 'fake', fake)]:
            folder = 'fake' if label else 'augmented_real'
            destination = output_root / folder / category / f'{path_id}.png'
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), pixels):
                raise OSError(f'Failed to save {destination}')
            row = dict(metadata, filepath=str(destination), label=kind, artifact_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
            row.update(capture_metadata)
            row['generator_version'] = GENERATOR_VERSION
            # Keep diagnostics as metadata, never as feat_* model inputs.
            for name in ('edit_jpeg_quality', 'edit_alpha', 'edit_beta',
                         'edit_feather_fraction', 'edit_rotation_angle',
                         'requested_edit_x', 'requested_edit_y',
                         'requested_edit_width', 'requested_edit_height'):
                row[name] = edit_metadata.get(name, '') if label else ''
            row.update({
                'tamper_type': tamper_type if label else 'none',
                'donor_id': donor_id if label and tamper_type == 'splice' else '',
                'changed_fraction': float(changed.mean()) if label else 0.0,
                'mean_abs_change': float(difference.mean()) if label else 0.0,
                'changed_mean_abs_change': (
                    float(difference[changed].mean()) if label else 0.0
                ),
                # Bounding box of changed pixels, not the requested edit patch.
                'changed_x': int(xs.min()) if label else -1,
                'changed_y': int(ys.min()) if label else -1,
                'changed_width': int(xs.max() - xs.min() + 1) if label else 0,
                'changed_height': int(ys.max() - ys.min() + 1) if label else 0,
            })
            rows.append(row)

    manifest = pd.DataFrame(rows)
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    manifest.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    print(f'Prepared {len(manifest)} images; labels: real/fake')
    print(manifest.groupby(['fold_id', 'label']).size().to_string())
    print('Fake-only methods (balanced within document type/fold):')
    print(manifest.loc[manifest['label'] == 'fake'].groupby(
        ['document_type', 'fold_id', 'tamper_type']).size().to_string())
    print('Both classes: Poisson noise + mild localized blur + brightness variation.')
    print('Fakes only: rotation/copy-move/splice + patch JPEG quality 60-90,')
    print('small patch brightness changes, and feathered blending. No full-image blur.')
    print(f'Generator version: {GENERATOR_VERSION}')
    print('changed_* metrics compare fake to its augmented real counterpart.')
    print('Reserve split=test; use fold_id for development CV only.')
    return manifest

def generate_fakes():
    return prepare_dataset(
        SAMPLE_DIR, CATEGORIES, SEED, GENERATED_MANIFEST_PATH, make_fake_version
    )


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    generate_fakes()