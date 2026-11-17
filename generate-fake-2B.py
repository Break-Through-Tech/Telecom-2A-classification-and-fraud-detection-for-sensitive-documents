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
TAMPER_TYPES = ["rotation", "partial_blur", "copy_move", "splice", "text"]

# Same shared noise distribution as the supplied notebook. No shared blur
# or brightness shift. Freeze a source's draw independently of labels/edits.
GENERATOR_VERSION = 'team_consistent_rotation_v1'


def apply_capture_variation(image, source_id, seed):
    key = hashlib.sha256(f"capture:{seed}:{source_id}".encode()).digest()
    rng = random.Random(int.from_bytes(key[:16], 'big'))
    nrng = np.random.default_rng(int.from_bytes(key[16:], 'big'))
    if rng.random() < 0.5:
        sigma = rng.uniform(2.0, 6.0)
        pixels = np.clip(image.astype(np.float32) + nrng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)
        tag, scale = 'gauss', ''
    else:
        sigma = ''
        scale = rng.uniform(60.0, 120.0)
        values = np.clip(image.astype(np.float32) / 255.0 * scale, 0, None)
        pixels = np.clip(nrng.poisson(values) / scale * 255.0, 0, 255).astype(np.uint8)
        tag = 'poisson'
    return pixels, {'capture_noise': tag, 'capture_sigma': sigma,
                    'capture_poisson_scale': scale, 'capture_blur': 'none',
                    'capture_brightness_alpha': 1.0, 'capture_brightness_beta': 0.0,
                    'capture_order': 'before'}


def jpeg_compress(img, quality):
    # re-save as jpeg, lower quality = more compression artifacts
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def edit_traces(patch):
    # pasted areas don't match the rest of the card exactly, but only slightly.
    # milder than before (was quality 30-70, alpha 0.92-1.08, beta -10..10)
    patch = jpeg_compress(patch, random.randint(60, 90))
    return cv2.convertScaleAbs(patch, alpha=random.uniform(0.96, 1.04), beta=random.randint(-5, 5))


def feather_paste(out, patch, y, x, edge=0.15):
    # blend the patch in with soft edges instead of a hard rectangle border.
    # mask is 1 in the middle and fades to 0 near the edges
    ph, pw = patch.shape[:2]
    m = max(2, int(min(ph, pw) * edge))
    mask = np.zeros((ph, pw), np.float32)
    mask[m:ph-m, m:pw-m] = 1.0
    mask = cv2.GaussianBlur(mask, (2 * m + 1, 2 * m + 1), 0)[..., None]
    region = out[y:y+ph, x:x+pw].astype(np.float32)
    out[y:y+ph, x:x+pw] = (mask * patch.astype(np.float32) + (1 - mask) * region).astype(np.uint8)


def rand_region(h, w, lo, hi):
    # random box inside the image, size is lo-hi fraction of height/width
    rh, rw = int(h * random.uniform(lo, hi)), int(w * random.uniform(lo, hi))
    return rh, rw, random.randint(0, h - rh), random.randint(0, w - rw)


def blur_partial(img):
    # blur one area, blended in with soft edges (was kernel 5-21, hard edges)
    k = random.choice([5, 7, 9, 11])
    h, w = img.shape[:2]
    rh, rw, y, x = rand_region(h, w, 0.15, 0.30)
    out = img.copy()
    feather_paste(out, cv2.GaussianBlur(img[y:y+rh, x:x+rw], (k, k), 0), y, x)
    return out, f"partial{k}"


def copy_move(img):
    # copy part of the card and paste it elsewhere on the same card, soft edges
    h, w = img.shape[:2]
    rh, rw, y1, x1 = rand_region(h, w, 0.08, 0.20)
    y2, x2 = random.randint(0, h - rh), random.randint(0, w - rw)
    out = img.copy()
    feather_paste(out, edit_traces(img[y1:y1+rh, x1:x1+rw]), y2, x2)
    return out, "copymove"


def splice(img, donor):
    # paste a piece from a different card of the same type, soft edges
    donor = cv2.resize(donor, (img.shape[1], img.shape[0]))  # same size so coords match
    h, w = img.shape[:2]
    rh, rw, y, x = rand_region(h, w, 0.10, 0.25)
    out = img.copy()
    feather_paste(out, edit_traces(donor[y:y+rh, x:x+rw]), y, x)
    return out, "splice"


def text_tamper(img):
    # cover a small strip and write a fake date over it
    h, w = img.shape[:2]
    scale = w / 1200  # images are ~1000-1500px wide, so scale text to fit

    y = random.randint(int(h * 0.3), int(h * 0.85))
    x = random.randint(int(w * 0.3), int(w * 0.6))
    y0, y1 = max(0, y - int(30 * scale)), min(h, y + int(8 * scale))
    x0, x1 = x, min(w, x + int(220 * scale))

    # background color = median of the strip, closer to the real background than one pixel
    color = tuple(int(c) for c in np.median(img[y0:y1, x0:x1].reshape(-1, 3), axis=0))

    out = img.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), color, -1)
    cv2.putText(out, f"{random.randint(1, 28):02d}.{random.randint(1, 12):02d}.{random.randint(1950, 2005)}",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (30, 30, 30), max(1, int(2 * scale)),
                cv2.LINE_AA)  # anti-aliased text, smoother edges like printed text

    # edited strip saved slightly differently from the rest of the card
    out[y0:y1, x0:x1] = edit_traces(out[y0:y1, x0:x1])
    return out, "text"


def rotation_tamper(image):
    """Replace full blur with a local rotation, retaining team's edit processing."""
    h, w = image.shape[:2]
    ph, pw, y, x = rand_region(h, w, 0.08, 0.20)
    if min(ph, pw) < 3:
        raise ValueError('Image too small for a localized rotation')
    angle = random.choice(ROTATION_ANGLES)
    patch = image[y:y+ph, x:x+pw]
    matrix = cv2.getRotationMatrix2D((pw/2, ph/2), angle, 1.0)
    rotated = cv2.warpAffine(patch, matrix, (pw, ph), borderMode=cv2.BORDER_REFLECT)
    out = image.copy()
    feather_paste(out, edit_traces(rotated), y, x)
    return out, 'rotation'


def make_fake_version(image, donor_image, tamper_type=None, return_metadata=False):
    kind = tamper_type if tamper_type is not None else random.choice(TAMPER_TYPES)
    if kind == 'rotation':
        out, _ = rotation_tamper(image)
    elif kind == 'partial_blur':
        out, _ = blur_partial(image)
    elif kind == 'copy_move':
        out, _ = copy_move(image)
    elif kind == 'splice':
        if donor_image is None:
            raise ValueError('Splice requires a same-type, same-fold donor')
        out, _ = splice(image, donor_image)
    elif kind == 'text':
        out, _ = text_tamper(image)
    else:
        raise ValueError(f'Unsupported tampering method: {kind}')
    if out is None or out.shape != image.shape or out.dtype != np.uint8:
        raise ValueError(f'Invalid generated image for {kind}')
    return (out, kind, {}) if return_metadata else (out, kind)


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
    # With 5 styles, counts differ by at most one within each such group.
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
            # Match notebook output encoding: retain the original JPG/PNG suffix
            # for BOTH classes. JPEG output uses OpenCV's default, as theirs does.
            suffix = path.suffix.lower()
            if suffix not in ('.jpg', '.jpeg', '.png'):
                raise ValueError(f'Unsupported source format: {path}')
            destination = output_root / folder / category / f'{path_id}{suffix}'
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), pixels):
                raise OSError(f'Failed to save {destination}')
            row = dict(metadata, filepath=str(destination), label=kind, artifact_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
            row.update(capture_metadata)
            row['generator_version'] = GENERATOR_VERSION
            row['edit_processing'] = ('notebook_patch_processing' if label else 'none')
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
    print('Both classes: Gaussian sigma 2-6 OR Poisson scale 60-120; no shared blur/brightness shift.')
    print('Fakes only: rotation, partial blur, copy-move, splice, text. No full-image blur.')
    print('Copy-move/splice/text use the supplied notebook functions; rotation replaces full blur.')
    print('JPG/PNG output format follows original source for both classes.')
    print(f'Generator version: {GENERATOR_VERSION}')
    print('changed_* diagnostics compare pre-save fake and noisy real pixels, before final JPG/PNG encoding.')
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