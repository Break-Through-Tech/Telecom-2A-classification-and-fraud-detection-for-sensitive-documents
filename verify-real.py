
import glob
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_DIR, CATEGORIES

SALT_PEPPER_THRESHOLD = 0.01   # fraction of pure-black/pure-white pixels
BLOCK_GRID_SIZE = 8
Z_SCORE_THRESHOLD = 8.0        # one block's edge energy wildly unlike the rest


def _salt_pepper_fraction(gray_image):
    return float(np.mean(gray_image == 0) + np.mean(gray_image == 255))


def _max_block_zscore(gray_image, grid_size=BLOCK_GRID_SIZE):
    h, w = gray_image.shape
    bh, bw = h // grid_size, w // grid_size
    if bh == 0 or bw == 0:
        return 0.0
    lap_vars = []
    for i in range(grid_size):
        for j in range(grid_size):
            block = gray_image[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            lap_vars.append(cv2.Laplacian(block, cv2.CV_64F).var())
    lap_vars = np.array(lap_vars)
    std = lap_vars.std()
    if std < 1e-6:
        return 0.0
    return float(np.max(np.abs(lap_vars - lap_vars.mean())) / std)


def verify_real_samples():
    real_dir = os.path.join(SAMPLE_DIR, "real")
    suspicious = []
    total = 0

    for category in CATEGORIES.keys():
        files = glob.glob(os.path.join(real_dir, category, "*.*"))
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            total += 1

            sp_frac = _salt_pepper_fraction(img)
            z = _max_block_zscore(img)

            if sp_frac > SALT_PEPPER_THRESHOLD or z > Z_SCORE_THRESHOLD:
                suspicious.append((f, sp_frac, z))

    print(f"Scanned {total} images under {os.path.abspath(real_dir)}")
    if not suspicious:
        print("No contamination signs found (no salt-and-pepper noise, no localized anomalies).")
        return

    print(f"\n{len(suspicious)} suspicious image(s) -- review these manually:")
    for f, sp_frac, z in sorted(suspicious, key=lambda row: -max(row[1] * 100, row[2])):
        print(f"  salt-pepper={sp_frac:.3f}  block_zscore={z:6.2f}  {f}")


if __name__ == "__main__":
    verify_real_samples()