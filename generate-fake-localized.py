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
BLUR_KERNEL_RANGE = (15, 35)         # Gaussian blur kernel size in pixels (odd values only)
IMPULSE_AMOUNT_RANGE = (0.05, 0.2)   # fraction of patch pixels flipped to black/white
POISSON_SCALE_RANGE = (0.3, 1.0)     # lower = noisier (Poisson noise is signal-dependent)
ROTATION_ANGLES = list(range(15, 166, 15))  # avoid near-0/180 angles, which barely change anything

TAMPER_TYPES = ["partial_blur", "impulse_noise", "poisson_noise", "copy_move", "splice", "rotation"]

def _random_odd_kernel_size():
    k = random.randint(*BLUR_KERNEL_RANGE)
    return k if k % 2 == 1 else k + 1  # cv2.GaussianBlur requires an odd kernel size

def _random_patch_size(h, w):
    frac = random.uniform(*PATCH_FRACTION_RANGE)
    ph = max(int(h * frac), 10)
    pw = max(int(w * frac), 10)
    return min(ph, h), min(pw, w)

def partial_blur_tamper(image):
    """Apply a strong Gaussian blur to one random patch, leaving the rest untouched."""
    h, w = image.shape[:2]
    ph, pw = _random_patch_size(h, w)
    y, x = random.randint(0, h - ph), random.randint(0, w - pw)
    k = _random_odd_kernel_size()
    tampered = image.copy()
    tampered[y:y + ph, x:x + pw] = cv2.GaussianBlur(image[y:y + ph, x:x + pw], (k, k), 0)
    return tampered

def impulse_noise_tamper(image):
    """Salt-and-pepper noise on one random patch -- random pixels pushed to pure
    black or white, disturbing both local texture (LBP) and creating spurious
    point-edges (HOG)."""
    h, w = image.shape[:2]
    ph, pw = _random_patch_size(h, w)
    y, x = random.randint(0, h - ph), random.randint(0, w - pw)
    amount = random.uniform(*IMPULSE_AMOUNT_RANGE)

    tampered = image.copy()
    patch = tampered[y:y + ph, x:x + pw]
    noise_mask = np.random.rand(ph, pw)
    patch[noise_mask < amount / 2] = 255
    patch[(noise_mask >= amount / 2) & (noise_mask < amount)] = 0
    return tampered


def poisson_noise_tamper(image):
    """Poisson (shot) noise on one random patch -- signal-dependent noise, a
    different statistical fingerprint than Gaussian/impulse noise."""
    h, w = image.shape[:2]
    ph, pw = _random_patch_size(h, w)
    y, x = random.randint(0, h - ph), random.randint(0, w - pw)
    scale = random.uniform(*POISSON_SCALE_RANGE)

    patch = image[y:y + ph, x:x + pw].astype(np.float64)
    noisy_patch = np.random.poisson(patch * scale) / scale

    tampered = image.copy()
    tampered[y:y + ph, x:x + pw] = np.clip(noisy_patch, 0, 255).astype(np.uint8)
    return tampered


def copy_move_tamper(image):
    """Copy a patch from elsewhere in the same image over another region --
    transplants real edge/gradient content, which HOG is sensitive to."""
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
    tampered[dy:dy + ph, dx:dx + pw] = patch
    return tampered


def splice_tamper(image, donor_image):
    """Copy a patch from a different document into this one -- same idea as
    copy-move, but the transplanted edge content comes from a foreign source."""
    h, w = image.shape[:2]
    ph, pw = _random_patch_size(h, w)

    dh, dw = donor_image.shape[:2]
    sph, spw = min(ph, dh), min(pw, dw)
    sy, sx = random.randint(0, dh - sph), random.randint(0, dw - spw)
    patch = donor_image[sy:sy + sph, sx:sx + spw]
    patch = cv2.resize(patch, (pw, ph))  # match the target patch size exactly

    dy, dx = random.randint(0, h - ph), random.randint(0, w - pw)

    tampered = image.copy()
    tampered[dy:dy + ph, dx:dx + pw] = patch
    return tampered


def rotation_tamper(image):
    """Rotate one random patch in place -- directly shifts the local gradient
    orientations that HOG measures, and creates a boundary mismatch with the
    un-rotated surroundings. (A patch that lands on a flat/blank background
    region can end up looking nearly unchanged -- a known, low-impact edge
    case since this is only one of six randomly chosen tamper types.)"""
    h, w = image.shape[:2]
    ph, pw = _random_patch_size(h, w)
    y, x = random.randint(0, h - ph), random.randint(0, w - pw)

    angle = random.choice(ROTATION_ANGLES)
    patch = image[y:y + ph, x:x + pw]
    rot_matrix = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
    rotated_patch = cv2.warpAffine(patch, rot_matrix, (pw, ph), borderMode=cv2.BORDER_REFLECT)

    tampered = image.copy()
    tampered[y:y + ph, x:x + pw] = rotated_patch
    return tampered


def make_fake_version(image, donor_image):
    tamper_type = random.choice(TAMPER_TYPES)

    if tamper_type == "partial_blur":
        fake_img = partial_blur_tamper(image)
    elif tamper_type == "impulse_noise":
        fake_img = impulse_noise_tamper(image)
    elif tamper_type == "poisson_noise":
        fake_img = poisson_noise_tamper(image)
    elif tamper_type == "copy_move":
        fake_img = copy_move_tamper(image)
    elif tamper_type == "splice":
        fake_img = splice_tamper(image, donor_image)
    else:  # "rotation"
        fake_img = rotation_tamper(image)

    return fake_img, tamper_type

def prepare_dataset(sample_dir, categories, seed, manifest_path, make_fake):
    random.seed(seed)
    np.random.seed(seed)
    root = Path(sample_dir).resolve()
    source_root = root / 'real'
    output_root = root
    require_empty_outputs(root / 'fake', manifest_path)
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

    pools = {
        fold: [sid for sid in identities if folds[sid] == fold]
        for fold in [-1, *range(N_FOLDS)]
    }
    if any(len(pool) < 2 for pool in pools.values()):
        raise ValueError(
            'Not enough distinct originals: test and every development fold '
            'need at least two sources for same-partition splicing. '
            'Add sources or reduce N_FOLDS consistently in both scripts.'
        )

    rows = []
    for category, path, source_id in records:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Input became unreadable: {path}')
        fold = folds[source_id]
        donor_ids = [sid for sid in pools[fold] if sid != source_id]
        # Retry only exact no-ops; retain subtle edits and report their magnitude.
        for _ in range(10):
            donor_id = random.choice(donor_ids)
            donor_path = identities[donor_id][1]
            donor = cv2.imread(str(donor_path), cv2.IMREAD_COLOR)
            if donor is None:
                raise ValueError(f'Unreadable donor: {donor_path}')
            before = image.copy()
            donor_before = donor.copy()
            fake, tamper_type = make_fake(image, donor)
            if not np.array_equal(image, before) or not np.array_equal(donor, donor_before):
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
            if label:
                destination = output_root / 'fake' / category / f'{path_id}.png'
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), pixels):
                    raise OSError(f'Failed to save {destination}')
            else:
                # Reuse the verified real input; never write into the real folder.
                destination = path
            row = dict(metadata, filepath=str(destination), label=kind, artifact_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
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