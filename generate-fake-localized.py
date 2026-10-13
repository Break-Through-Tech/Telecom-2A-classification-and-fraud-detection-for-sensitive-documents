import glob
import os
import random
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_DIR, CATEGORIES, SEED

PATCH_FRACTION_RANGE = (0.08, 0.2)  # patch side length as a fraction of image side


def _random_patch_size(h, w):
    frac = random.uniform(*PATCH_FRACTION_RANGE)
    ph = max(int(h * frac), 10)
    pw = max(int(w * frac), 10)
    return min(ph, h), min(pw, w)


def copy_move_tamper(image):
    """Copy a patch from elsewhere in the same image over another region."""
    h, w = image.shape[:2]
    ph, pw = _random_patch_size(h, w)

    sy, sx = random.randint(0, h - ph), random.randint(0, w - pw)
    patch = image[sy:sy + ph, sx:sx + pw].copy()

    dy, dx = random.randint(0, h - ph), random.randint(0, w - pw)

    tampered = image.copy()
    tampered[dy:dy + ph, dx:dx + pw] = patch
    return tampered


def splice_tamper(image, donor_image):
    """Copy a patch from a different document into this one."""
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


def make_fake_version(image, donor_image):
    if random.random() < 0.5:
        return copy_move_tamper(image)
    return splice_tamper(image, donor_image)


def generate_fakes():
    real_dir = os.path.join(SAMPLE_DIR, "real")
    fake_dir = os.path.join(SAMPLE_DIR, "fake")

    all_real_files = glob.glob(os.path.join(real_dir, "*", "*.*"))

    for category in CATEGORIES.keys():
        real_files = glob.glob(os.path.join(real_dir, category, "*.*"))
        out_dir = os.path.join(fake_dir, category)
        os.makedirs(out_dir, exist_ok=True)

        for f in real_files:
            img = cv2.imread(f)
            if img is None:
                continue

            donor_path = random.choice([p for p in all_real_files if p != f])
            donor_img = cv2.imread(donor_path)
            if donor_img is None:
                continue

            fake_img = make_fake_version(img, donor_img)
            fname = os.path.basename(f)
            cv2.imwrite(os.path.join(out_dir, fname), fake_img)

        print(f"{category}: generated {len(real_files)} localized-tamper fake images")

    print("\nLocalized-tamper fake documents ready at:", os.path.abspath(fake_dir))


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    generate_fakes()