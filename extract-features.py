import os
import sys

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MANIFEST_PATH, FEATURES_PATH

# --- HOG settings (captures edge/shape structure) ---
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# --- LBP settings (captures micro-texture patterns) ---
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"


def extract_hog(gray_image):
    return hog(
        gray_image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        feature_vector=True,
    )


def extract_lbp(gray_image):
    lbp = local_binary_pattern(gray_image, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float32")
    hist /= hist.sum() + 1e-7  # normalize so the histogram is comparable across images
    return hist


def extract_features_for_image(filepath):
    # IMREAD_GRAYSCALE forces a single channel regardless of how the file is
    # stored, since some preprocessed formats (e.g. webp) keep 3 channels on disk.
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return np.concatenate([extract_hog(img), extract_lbp(img)])


def extract_all():
    manifest = pd.read_csv(MANIFEST_PATH)

    feature_rows = []
    skipped = 0
    for _, row in manifest.iterrows():
        features = extract_features_for_image(row["filepath"])
        if features is None:
            skipped += 1
            continue
        feature_row = {f"feat_{i}": v for i, v in enumerate(features)}
        feature_row["label"] = row["label"]
        feature_row["document_type"] = row["document_type"]
        feature_rows.append(feature_row)

    df = pd.DataFrame(feature_rows)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"Extracted features for {len(df)} images ({skipped} skipped)")
    print(f"Features written to: {os.path.abspath(FEATURES_PATH)}")
    return df


if __name__ == "__main__":
    extract_all()
