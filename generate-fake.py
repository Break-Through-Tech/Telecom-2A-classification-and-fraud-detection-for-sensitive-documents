"""
Task 3: Generate the synthetic "fake" document class.
 
IDNet only contains genuine document templates, so we simulate fraud/tampering
by applying Gaussian blur + noise injection with OpenCV to create a matching
"fake" set. (Albumentations is a good alternative if you want more varied
augmentations later.)
 
Run from the project root, after task2_download_and_sample.py:
    python scripts/task3_generate_fake_class.py
"""
 
import os
import glob
import sys
 
import cv2
import numpy as np
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_DIR, CATEGORIES
 
 
def add_gaussian_noise(image, mean=0, sigma=15):
    """Add Gaussian noise to simulate document degradation/tampering."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)
 
 
def make_fake_version(image, blur_kernel=(7, 7), noise_sigma=15):
    """Apply Gaussian blur + noise to create a synthetic 'fake' document."""
    blurred = cv2.GaussianBlur(image, blur_kernel, 0)
    faked = add_gaussian_noise(blurred, sigma=noise_sigma)
    return faked
 
 
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
    generate_fakes()