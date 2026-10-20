import glob
import os
import random
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_DIR, CATEGORIES, SEED

PATCH_FRACTION_RANGE = (0.08, 0.2)  # patch side length as a fraction of image side
BLUR_KERNEL_RANGE = (15, 35)

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

def make_fake_version(image):
    return partial_blur_tamper(image)


def generate_fakes():
    real_dir = os.path.join(SAMPLE_DIR, "real")
    fake_dir = os.path.join(SAMPLE_DIR, "fake")

    for category in CATEGORIES.keys():
        real_files = glob.glob(os.path.join(real_dir, category, "*.*"))
        out_dir = os.path.join(fake_dir, category)
        os.makedirs(out_dir, exist_ok=True)

        for f in real_files:
            img = cv2.imread(f)
            if img is None:
                continue

            fake_img = make_fake_version(img)
            fname = os.path.basename(f)
            cv2.imwrite(os.path.join(out_dir, fname), fake_img)

        print(f"{category}: generated {len(real_files)} localized-blur fake images")

    print("\nLocalized-blur fake documents ready at:", os.path.abspath(fake_dir))



if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    generate_fakes()