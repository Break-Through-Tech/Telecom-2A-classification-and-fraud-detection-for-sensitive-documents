import os
import sys
from pathlib import Path
import hashlib
import argparse

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MANIFEST_PATH, FEATURES_PATH, IMG_SIZE, TEXTURE_MAX_SIDE, require_empty_outputs
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

LBP_RADIUS = 1
LBP_N_POINTS = 8
LBP_METHOD = "uniform"

LBP_RADII = (1, 2, 3)
LBP_GRID_SIZE = 4
BLOCK_GRID_SIZE = 8
ORIENTATION_BINS = 18
ORIENTATION_PATCH_FRACTIONS = (0.08, 0.16, 0.24)
ORIENTATION_MIN_COHERENCE = 0.2

def fit_hog_canvas(gray_image):
    height, width = gray_image.shape
    target_width, target_height = IMG_SIZE
    ratio = min(target_width / width, target_height / height)
    new_width = min(target_width, max(1, round(width * ratio)))
    new_height = min(target_height, max(1, round(height * ratio)))
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(gray_image, (new_width, new_height), interpolation=interpolation)
    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2
    return cv2.copyMakeBorder(
        resized, top, target_height - new_height - top,
        left, target_width - new_width - left, cv2.BORDER_REPLICATE,
    )


def texture_image(gray_image):
    height, width = gray_image.shape
    if TEXTURE_MAX_SIDE is None or max(height, width) <= TEXTURE_MAX_SIDE:
        return gray_image
    ratio = TEXTURE_MAX_SIDE / max(height, width)
    return cv2.resize(gray_image, (max(1, round(width * ratio)), max(1, round(height * ratio))), interpolation=cv2.INTER_AREA)

def extract_hog(gray_image):
    return hog(gray_image, orientations=HOG_ORIENTATIONS,
              pixels_per_cell=HOG_PIXELS_PER_CELL,
              cells_per_block=HOG_CELLS_PER_BLOCK, feature_vector=True)

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


def extract_lbp(gray_image, radii=None):
    if min(gray_image.shape) < 2 * max(LBP_RADII) + 1:
        raise ValueError("Image too small for texture extraction")
    ys = np.linspace(0, gray_image.shape[0], LBP_GRID_SIZE + 1, dtype=int)
    xs = np.linspace(0, gray_image.shape[1], LBP_GRID_SIZE + 1, dtype=int)
    features = {}
    for radius in (LBP_RADII if radii is None else radii):
        points = 8 * radius
        lbp = local_binary_pattern(gray_image, points, radius, method="uniform")
        for i, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
            for j, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
                values = lbp[y0:y1, x0:x1].astype(np.intp).ravel()
                histogram = np.bincount(values, minlength=points + 2).astype(np.float32)
                histogram /= histogram.sum()
                for k, value in enumerate(histogram):
                    features[f"feat_lbp_r{radius}_y{i}_x{j}_b{k}"] = float(value)
    return features


def _orientation_histogram(angles, weights, bins):
    """Magnitude-weighted, circular bins: 0 and 180 degrees are equivalent."""
    position = np.mod(angles, np.pi) * (bins / np.pi)
    lower = np.floor(position).astype(np.intp)
    fraction = position - lower
    histogram = np.bincount(
        np.mod(lower, bins).ravel(),
        weights=(weights * (1.0 - fraction)).ravel(), minlength=bins,
    )
    histogram += np.bincount(
        np.mod(lower + 1, bins).ravel(),
        weights=(weights * fraction).ravel(), minlength=bins,
    )
    total = histogram.sum()
    return histogram / total if total > 0 else np.zeros(bins, dtype=np.float64)


def _orientation_js_distance(first, second):
    """Jensen-Shannon divergence, bounded between 0 and 1 for normalized histograms."""
    middle = 0.5 * (first + second)
    epsilon = 1e-12
    divergence = 0.5 * np.sum(first * np.log2((first + epsilon) / (middle + epsilon)))
    divergence += 0.5 * np.sum(second * np.log2((second + epsilon) / (middle + epsilon)))
    return float(np.clip(divergence, 0.0, 1.0))


def _orientation_summaries(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {name: 0.0 for name in ("mean", "std", "p50", "p90", "p95", "max")}
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)), "max": float(values.max()),
    }


def extract_local_edge_orientation(gray_image):
    """Compare overlapping edge-direction histograms at three patch scales.

    Uses the unpadded texture view, not HOG's small canvas. Blank/weak-edge
    patches are skipped; angular comparisons also require directional coherence.
    Thresholds below are starting settings to evaluate on development folds.
    """
    if gray_image.ndim != 2 or min(gray_image.shape) < 8:
        raise ValueError("Orientation features require a grayscale image at least 8x8")
    image = gray_image.astype(np.float32) / 255.0
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    threshold = max(0.01, 0.1 * float(np.percentile(magnitude, 90)))
    strong = magnitude > threshold
    weights = np.where(strong, magnitude, 0.0)
    global_histogram = _orientation_histogram(angle, weights, ORIENTATION_BINS)
    centers = np.arange(ORIENTATION_BINS) * (np.pi / ORIENTATION_BINS)
    cosines, sines = np.cos(2 * centers), np.sin(2 * centers)
    height, width = image.shape
    features = {}

    for fraction in ORIENTATION_PATCH_FRACTIONS:
        ph = min(height, max(8, round(height * fraction)))
        pw = min(width, max(8, round(width * fraction)))
        ys = sorted(set(range(0, height - ph + 1, max(1, ph // 2))) | {height - ph})
        xs = sorted(set(range(0, width - pw + 1, max(1, pw // 2))) | {width - pw})
        patches = {}
        values = {name: [] for name in (
            "coherence", "entropy", "edge_density", "global_js",
            "neighbor_js", "neighbor_angle",
        )}
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                region = np.s_[y:y + ph, x:x + pw]
                count = int(strong[region].sum())
                if count < max(8, int(np.ceil(0.02 * ph * pw))):
                    continue
                histogram = _orientation_histogram(angle[region], weights[region], ORIENTATION_BINS)
                real = float(histogram @ cosines)
                imaginary = float(histogram @ sines)
                coherence = float(np.clip(np.hypot(real, imaginary), 0.0, 1.0))
                direction = 0.5 * np.arctan2(imaginary, real)
                entropy = -float(np.sum(histogram * np.log2(histogram + 1e-12))) / np.log2(ORIENTATION_BINS)
                patches[i, j] = (histogram, coherence, direction)
                values["coherence"].append(coherence)
                values["entropy"].append(float(np.clip(entropy, 0.0, 1.0)))
                values["edge_density"].append(count / (ph * pw))
                values["global_js"].append(_orientation_js_distance(histogram, global_histogram))

        coherent_pairs = 0
        for (i, j), (histogram, coherence, direction) in patches.items():
            # Right/down neighbors only, so each pair is counted once.
            for neighbor in ((i, j + 1), (i + 1, j)):
                if neighbor not in patches:
                    continue
                other_histogram, other_coherence, other_direction = patches[neighbor]
                values["neighbor_js"].append(_orientation_js_distance(histogram, other_histogram))
                if min(coherence, other_coherence) >= ORIENTATION_MIN_COHERENCE:
                    delta = direction - other_direction
                    # Circular distance handles the 0/180 degree wrap correctly.
                    distance = 0.5 * abs(np.arctan2(np.sin(2 * delta), np.cos(2 * delta)))
                    values["neighbor_angle"].append(float(distance / (np.pi / 2)))
                    coherent_pairs += 1

        prefix = f"feat_orientation_s{fraction:g}"
        possible_pairs = len(ys) * max(0, len(xs) - 1) + max(0, len(ys) - 1) * len(xs)
        features[prefix + "_valid_fraction"] = len(patches) / (len(ys) * len(xs))
        features[prefix + "_neighbor_pair_fraction"] = len(values["neighbor_js"]) / max(1, possible_pairs)
        features[prefix + "_coherent_pair_fraction"] = coherent_pairs / max(1, possible_pairs)
        histograms = np.asarray([patch[0] for patch in patches.values()])
        features[prefix + "_histogram_dispersion"] = float(histograms.std(axis=0).mean()) if histograms.size else 0.0
        for name, samples in values.items():
            for statistic, value in _orientation_summaries(samples).items():
                features[f"{prefix}_{name}_{statistic}"] = value
    return features


def extract_local_residuals(gray_image):
    image = gray_image.astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
    median = cv2.medianBlur(gray_image, 3).astype(np.float32) / 255.0
    residual = np.abs(image - median)
    highpass = image - cv2.GaussianBlur(image, (5, 5), 0)
    height, width = gray_image.shape
    features = {}
    names = ("brightness", "contrast", "lap_var", "highpass_std", "median_residual", "large_residual_fraction")
    for fraction in (0.08, 0.16, 0.24):
        ph, pw = min(height, max(8, round(height * fraction))), min(width, max(8, round(width * fraction)))
        ys = sorted(set(range(0, height - ph + 1, max(1, ph // 2))) | {height - ph})
        xs = sorted(set(range(0, width - pw + 1, max(1, pw // 2))) | {width - pw})
        rows = []
        for y in ys:
            for x in xs:
                patch = np.s_[y:y + ph, x:x + pw]
                rows.append([
                    image[patch].mean(), image[patch].std(), laplacian[patch].var(),
                    highpass[patch].std(), residual[patch].mean(),
                    (residual[patch] > 0.20).mean(),
                ])
        values = np.asarray(rows)
        for i, name in enumerate(names):
            for percentile, value in zip((5, 25, 50, 75, 95), np.percentile(values[:, i], (5, 25, 50, 75, 95))):
                features[f"feat_residual_s{fraction:g}_{name}_p{percentile}"] = float(value)
            features[f"feat_residual_s{fraction:g}_{name}_std"] = float(values[:, i].std())
    return features


def extract_spatial_orientation_features(gray_image):
    """Retain coarse directions and fine mismatch locations, not just global summaries."""
    image = gray_image.astype(np.float32) / 255.0
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    strength = np.hypot(gx, gy)
    angles = np.mod(np.arctan2(gy, gx), np.pi)
    mask = strength > max(0.01, 0.1 * float(np.percentile(strength, 90)))
    weights = np.where(mask, strength, 0.0)
    features = {}
    centers = np.arange(9) * (np.pi / 9)
    for grid in (4, 8):
        ys = np.linspace(0, image.shape[0], grid + 1, dtype=int)
        xs = np.linspace(0, image.shape[1], grid + 1, dtype=int)
        cells = {}
        for i in range(grid):
            for j in range(grid):
                region = np.s_[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                area = mask[region].size
                valid = bool(area and mask[region].sum() >= max(4, int(np.ceil(.02 * area))))
                histogram = _orientation_histogram(angles[region], weights[region], 9) if valid else np.zeros(9)
                real, imaginary = histogram @ np.cos(2 * centers), histogram @ np.sin(2 * centers)
                coherence = float(np.clip(np.hypot(real, imaginary), 0, 1))
                direction = .5 * np.arctan2(imaginary, real)
                density = float(mask[region].mean()) if area else 0.0
                cells[i, j] = (valid, histogram, coherence, direction)
                prefix = f"feat_direction_g{grid}_y{i}_x{j}"
                features[prefix + "_density"] = density
                features[prefix + "_coherence"] = coherence
                if grid == 4:
                    for k, value in enumerate(histogram):
                        features[f"{prefix}_bin{k}"] = float(value)
        if grid == 8:
            for (i, j), (valid, histogram, coherence, direction) in cells.items():
                divergences, rotations = [], []
                if valid:
                    for neighbor in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
                        if neighbor not in cells or not cells[neighbor][0]:
                            continue
                        _, other, other_coherence, other_direction = cells[neighbor]
                        divergences.append(_orientation_js_distance(histogram, other))
                        if min(coherence, other_coherence) >= ORIENTATION_MIN_COHERENCE:
                            delta = direction - other_direction
                            rotations.append(float(abs(np.arctan2(np.sin(2*delta), np.cos(2*delta))) / np.pi))
                prefix = f"feat_direction_g{grid}_y{i}_x{j}"
                features[prefix + "_valid"] = float(valid)
                features[prefix + "_neighbor_js"] = max(divergences, default=0.0)
                features[prefix + "_neighbor_angle"] = max(rotations, default=0.0)
    return features


def _copy_move_statistics(points, descriptors, image_shape):
    """Self-match SIFT descriptors, reject local/ambiguous matches, group translations."""
    names = (
        "keypoint_density", "pair_fraction", "close_pair_fraction", "best_distance_min",
        "best_distance_p10", "best_distance_p50", "ratio_p10", "ratio_p50",
        "translation_peak", "translation_peak_fraction", "translation_top3_fraction",
        "translation_entropy", "peak_source_span", "peak_target_span",
    )
    output = {"feat_copy_" + name: 0.0 for name in names}
    height, width = image_shape
    n = len(points)
    output["feat_copy_keypoint_density"] = n / max(1, height * width) * 10000
    if descriptors is None or n < 4:
        return output
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    neighbors = matcher.knnMatch(descriptors, descriptors, k=min(24, n))
    minimum_separation = max(12, .05 * min(height, width))
    coordinates = np.asarray([p.pt for p in points], dtype=float)
    best = {}
    for i, matches in enumerate(neighbors):
        candidates = [m for m in matches if m.trainIdx != i and
                      np.linalg.norm(coordinates[i] - coordinates[m.trainIdx]) >= minimum_separation]
        if len(candidates) < 2:
            continue
        first, second = candidates[:2]
        # SIFT descriptor norms are about 512 in OpenCV's standard float output.
        ratio = float(first.distance / max(second.distance, 1e-12))
        if second.distance > 1e-6 and ratio < .75 and first.distance / 512 < .35:
            best[i] = (first.trainIdx, float(first.distance / 512), ratio)
    pairs = []
    for i, (j, distance, ratio) in best.items():
        if i < j and j in best and best[j][0] == i:
            pairs.append((i, j, distance, ratio))
    if not pairs:
        return output
    output["feat_copy_pair_fraction"] = 2 * len(pairs) / n
    distances = np.array([p[2] for p in pairs])
    ratios = np.array([p[3] for p in pairs])
    output["feat_copy_close_pair_fraction"] = float((distances < .1).sum() * 2 / n)
    for suffix, value in zip(("min", "p10", "p50"), (distances.min(), *np.percentile(distances, (10, 50)))):
        output["feat_copy_best_distance_" + suffix] = float(value)
    for suffix, value in zip(("p10", "p50"), np.percentile(ratios, (10, 50))):
        output["feat_copy_ratio_" + suffix] = float(value)
    # Copied content should produce multiple correspondences with a common
    # displacement, unlike scattered lookalike letters or natural repeats.
    offsets = []
    oriented_pairs = []
    for i, j, _, _ in pairs:
        offset = coordinates[j] - coordinates[i]
        if offset[0] < 0 or (offset[0] == 0 and offset[1] < 0):
            i, j, offset = j, i, -offset
        offsets.append(offset)
        oriented_pairs.append((i, j))
    buckets = np.rint(np.asarray(offsets) / 4.0).astype(int)
    unique, inverse, counts = np.unique(buckets, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(counts)[::-1]
    peak = int(order[0])
    output["feat_copy_translation_peak"] = float(counts[peak])
    output["feat_copy_translation_peak_fraction"] = float(counts[peak] / len(pairs))
    output["feat_copy_translation_top3_fraction"] = float(counts[order[:3]].sum() / len(pairs))
    probabilities = counts / counts.sum()
    output["feat_copy_translation_entropy"] = float(-np.sum(probabilities * np.log2(probabilities)))
    chosen = [pair for index, pair in enumerate(oriented_pairs) if inverse[index] == peak]
    for column, side in ((0, "source"), (1, "target")):
        locations = coordinates[[pair[column] for pair in chosen]]
        span = np.ptp(locations, axis=0) if len(locations) > 1 else np.zeros(2)
        output[f"feat_copy_peak_{side}_span"] = float(span.prod() / max(1, height * width))
    return output


def extract_copy_move_features(gray_image):
    """Keypoint matching handles copied regions at arbitrary pixel positions."""
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("Copy-move features require an OpenCV build with SIFT_create. Update your existing opencv-python package.")
    height, width = gray_image.shape
    ratio = min(1.0, 768 / max(height, width))
    view = cv2.resize(gray_image, (max(1, round(width*ratio)), max(1, round(height*ratio))), interpolation=cv2.INTER_AREA) if ratio < 1 else gray_image
    detector = cv2.SIFT_create(nfeatures=1200, contrastThreshold=.02)
    points, descriptors = detector.detectAndCompute(view, None)
    return _copy_move_statistics(points or (), descriptors, view.shape)


def extract_features_for_image(filepath, profile="compact"):
    image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unreadable processed image: {filepath}")
    texture = texture_image(image)
    if profile == "full":
        canvas = fit_hog_canvas(image)
        features = {f"feat_hog_{i}": float(v) for i, v in enumerate(extract_hog(canvas))}
        features.update(extract_lbp(texture))
        features.update(extract_block_anomaly_features(canvas))
    elif profile in ("compact", "compact_lbp2"):
        # Residuals already summarize blur/noise at three scales; omit HOG,
        # repeated LBP radii and the six coarse block descriptors by default.
        features = extract_lbp(texture, radii=(1, 2) if profile == "compact_lbp2" else (1,))
    else:
        raise ValueError("profile must be compact, compact_lbp2, or full")
    features.update(extract_local_residuals(texture))
    features.update(extract_local_edge_orientation(texture))
    features.update(extract_spatial_orientation_features(texture))
    features.update(extract_copy_move_features(texture))
    return features


def extract_all(manifest_path=None, features_path=None, profile="compact"):
    input_manifest = manifest_path or MANIFEST_PATH
    output_path = features_path or FEATURES_PATH
    require_empty_outputs(output_path)
    manifest = pd.read_csv(input_manifest)
    required = {"filepath", "label", "document_type", "source_id", "donor_id", "split", "fold_id", "tamper_type", "artifact_sha256"}
    if manifest.empty or not required.issubset(manifest.columns):
        raise ValueError("Regenerate using the patched sampling/generator/preprocessing stages first")
    rows = []
    for record in manifest.to_dict("records"):
        path = Path(record["filepath"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["artifact_sha256"]:
            raise ValueError(f"Processed artifact changed: {path}")
        result = extract_features_for_image(path, profile=profile)
        result.update(record)  # Provenance and evaluation columns are metadata.
        result["feature_profile"] = profile
        result["feature_schema"] = "targeted_v1_lbp2" if profile == "compact_lbp2" else "targeted_v1"
        rows.append(result)
    output = pd.DataFrame(rows)
    feature_columns = [c for c in output if c.startswith("feat_")]
    if not np.isfinite(output[feature_columns].to_numpy()).all():
        raise ValueError("Feature matrix contains missing or infinite values")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    output.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    print(f"Extracted {len(output)} rows, {len(feature_columns)} features ({profile}): {destination}")
    for family in ("lbp", "residual", "orientation", "direction", "copy", "hog", "block"):
        count = sum(c.startswith(f"feat_{family}_") for c in feature_columns)
        if count:
            print(f"  {family}: {count}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-file", help="Use an existing clean processed manifest")
    parser.add_argument("--output-file", help="Save a new feature CSV for a baseline comparison")
    parser.add_argument("--profile", choices=("compact", "compact_lbp2", "full"), default="compact",
                        help="compact: 898 features; compact_lbp2: 1186 with radius-2 LBP; full includes legacy groups")
    args = parser.parse_args()
    extract_all(args.manifest_file, args.output_file, args.profile)
