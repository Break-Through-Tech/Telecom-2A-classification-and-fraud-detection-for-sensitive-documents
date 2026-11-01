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

BLOCK_GRID_SIZE = 8  # image is split into BLOCK_GRID_SIZE x BLOCK_GRID_SIZE blocks

def extract_hog(gray_image):
    return hog(
        gray_image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        feature_vector=True,
    )

def _image_blocks(gray_image, grid_size=BLOCK_GRID_SIZE):
    h, w = gray_image.shape
    bh, bw = h // grid_size, w // grid_size
    return [
        gray_image[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
        for i in range(grid_size)
        for j in range(grid_size)
    ]

def extract_block_anomaly_features(gray_image):
    blocks = _image_blocks(gray_image)
    lap_vars = np.array([cv2.Laplacian(b, cv2.CV_64F).var() for b in blocks])
    lap_mean, lap_std = lap_vars.mean(), lap_vars.std()
    lap_max_zscore = np.max(np.abs(lap_vars - lap_mean)) / (lap_std + 1e-6)
    intensity_means = np.array([b.mean() for b in blocks])
    intensity_std = intensity_means.std()

    flat_blocks = [b.astype(np.float64).flatten() for b in blocks]
    stds = np.array([f.std() for f in flat_blocks])
    min_std = 5.0
    max_ncc = -1.0
    for i in range(len(flat_blocks)):
        if stds[i] < min_std:
            continue
        a = flat_blocks[i] - flat_blocks[i].mean()
        a_norm = np.linalg.norm(a)
        for j in range(i + 1, len(flat_blocks)):
            if stds[j] < min_std:
                continue
            b_centered = flat_blocks[j] - flat_blocks[j].mean()
            denom = a_norm * np.linalg.norm(b_centered)
            if denom < 1e-6:
                continue
            ncc = np.dot(a, b_centered) / denom
            max_ncc = max(max_ncc, ncc)
    lbp_hists = []
    for b in blocks:
        lbp = local_binary_pattern(b, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
        hist, _ = np.histogram(lbp, bins=LBP_N_POINTS + 2, range=(0, LBP_N_POINTS + 2))
        hist = hist.astype("float32")
        hist /= hist.sum() + 1e-7
        lbp_hists.append(hist)
    texture_heterogeneity = np.array(lbp_hists).std(axis=0).mean()

    return {
        "feat_block_lap_mean": lap_mean,
        "feat_block_lap_std": lap_std,
        "feat_block_lap_max_zscore": lap_max_zscore,
        "feat_block_intensity_std": intensity_std,
        "feat_block_max_patch_ncc": max_ncc,
        "feat_block_lbp_heterogeneity": texture_heterogeneity,
    }


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
    pooled_features = np.concatenate([extract_hog(img), extract_lbp(img)])
    block_features = extract_block_anomaly_features(img)
    return pooled_features, block_features


def extract_all():
    manifest = pd.read_csv(MANIFEST_PATH)

    feature_rows = []
    skipped = 0
    for _, row in manifest.iterrows():
        result = extract_features_for_image(row["filepath"])
        if result is None:
            skipped += 1
            continue
        pooled_features, block_features = result
        feature_row = {f"feat_{i}": v for i, v in enumerate(pooled_features)}
        feature_row.update(block_features)
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
