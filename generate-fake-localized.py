import glob
import os
import random
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_DIR, CATEGORIES, SEED

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

def _random_odd_kernel_size():
    k = random.randint(*BLUR_KERNEL_RANGE)
    return k if k % 2 == 1 else k + 1  # cv2.GaussianBlur requires an odd kernel size

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

    dy, dx = random.randint(0, h - ph), random.randint(0, w - pw)

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


def generate_fakes():
    real_dir = os.path.join(SAMPLE_DIR, "real")
    fake_dir = os.path.join(SAMPLE_DIR, "fake")

    all_real_files = glob.glob(os.path.join(real_dir, "*", "*.*"))

    for category in CATEGORIES.keys():
        real_files = glob.glob(os.path.join(real_dir, category, "*.*"))
        out_dir = os.path.join(fake_dir, category)
        os.makedirs(out_dir, exist_ok=True)

        tamper_counts = {t: 0 for t in TAMPER_TYPES}

        for f in real_files:
            img = cv2.imread(f)
            if img is None:
                continue

            donor_path = random.choice([p for p in all_real_files if p != f])
            donor_img = cv2.imread(donor_path)
            if donor_img is None:
                continue

            fake_img, tamper_type = make_fake_version(img, donor_img)
            tamper_counts[tamper_type] += 1

            fname = os.path.basename(f)
            cv2.imwrite(os.path.join(out_dir, fname), fake_img)

        print(f"{category}: generated {len(real_files)} fake images")
        print(f"  tamper type breakdown: {tamper_counts}")

    print("\nRandomized localized-tamper fake documents ready at:", os.path.abspath(fake_dir))


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    generate_fakes()