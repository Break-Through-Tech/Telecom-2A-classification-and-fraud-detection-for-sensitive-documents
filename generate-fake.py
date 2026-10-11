import os
import glob
import sys
import random
 
import cv2
import albumentations as A
import numpy as np
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_DIR, CATEGORIES, SEED
 
TAMPER_TRANSFORM = A.Compose([
    # p=1.0 on the OneOf itself so every fake is guaranteed to differ from its
    # real source image (only the choice of blur and its strength are random).
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 9), p=1.0),
        A.MotionBlur(blur_limit=(3, 9), p=1.0),
    ], p=1.0),
    A.GaussNoise(std_range=(0.05, 0.2), p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ImageCompression(quality_range=(30, 80), p=0.5),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.05, 0.15), hole_width_range=(0.05, 0.3), p=0.3),
])


def make_fake_version(image):
    """Apply a randomized set of degradations to simulate document tampering."""
    return TAMPER_TRANSFORM(image=image)["image"]
 
 
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
 
        print(f"{category}: generated {len(real_files)} synthetic fake images")
 
    print("\nSynthetic fake documents ready at:", os.path.abspath(fake_dir))
 
 
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    generate_fakes()