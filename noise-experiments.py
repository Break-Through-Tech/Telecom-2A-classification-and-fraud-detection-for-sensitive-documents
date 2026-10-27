"""
Noise experiments: do impulse / Poisson noise make harder fakes than Gaussian blur?
Self-contained: reads the real images the pipeline already compressed, makes fakes
in memory, runs HOG + LBP + Random Forest with grouped 5-fold CV, and writes
noise_experiment_results.csv. It does not modify any other pipeline file.
Run from the project root:   python noise-experiments.py
"""
import os
import glob
import sys

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import COMPRESSED_DIR, CATEGORIES, IMG_SIZE, SEED

N_PER_CATEGORY = 100        # real images used per document type (lower = faster)
LOCAL_FRAC = (0.08, 0.30)   # local noise covers 8-30% of the image
WEBP_QUALITY = 80           # match compress.py if it uses something else

np.random.seed(SEED)


def impulse_noise(img, amount=0.02):
    out = img.copy()
    mask = np.random.rand(*img.shape[:2])
    out[mask < amount / 2] = 0
    out[mask > 1 - amount / 2] = 255
    return out


def poisson_noise(img, scale=0.5):
    noisy = np.random.poisson(img.astype(np.float32) * scale) / scale
    return np.clip(noisy, 0, 255).astype(np.uint8)


def gaussian_blur(img):
    return cv2.GaussianBlur(img, (7, 7), 0)


def identity(img):
    return img


def apply_local(img, fn):
    h, w = img.shape[:2]
    area = np.random.uniform(*LOCAL_FRAC) * h * w
    ph = max(1, min(h - 1, int(np.sqrt(area))))
    pw = max(1, min(w - 1, int(area / ph)))
    y = np.random.randint(0, h - ph + 1)
    x = np.random.randint(0, w - pw + 1)
    out = img.copy()
    out[y:y + ph, x:x + pw] = fn(img[y:y + ph, x:x + pw])
    return out


CONDITIONS = {
    "control_no_noise":    identity,
    "gaussian_blur_whole": gaussian_blur,
    "impulse_whole":       impulse_noise,
    "impulse_local":       lambda im: apply_local(im, impulse_noise),
    "poisson_whole":       poisson_noise,
    "poisson_local":       lambda im: apply_local(im, poisson_noise),
}


def webp_roundtrip(img):
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, WEBP_QUALITY])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def to_gray_resized(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)


def features(gray):
    h = hog(gray, orientations=9, pixels_per_cell=(16, 16),
            cells_per_block=(2, 2), block_norm="L2-Hys")
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7
    return np.concatenate([h, hist])


def load_real_images():
    paths = []
    for cat in CATEGORIES:
        files = sorted(glob.glob(os.path.join(COMPRESSED_DIR, "real", cat, "*.*")))
        paths += files[:N_PER_CATEGORY]
    imgs = [cv2.imread(p) for p in paths]
    return [im for im in imgs if im is not None]


def run_condition(name, fake_fn, real_imgs, real_feats):
    fake_feats = []
    for im in real_imgs:
        fake = webp_roundtrip(fake_fn(im))
        fake_feats.append(features(to_gray_resized(fake)))
    fake_feats = np.array(fake_feats)

    X = np.vstack([real_feats, fake_feats])
    y = np.array([0] * len(real_feats) + [1] * len(fake_feats))
    groups = np.concatenate([np.arange(len(real_feats))] * 2)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs, aucs, importances = [], [], []
    for tr, te in cv.split(X, y, groups):
        rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        proba = rf.predict_proba(X[te])[:, 1]
        accs.append(accuracy_score(y[te], proba > 0.5))
        aucs.append(roc_auc_score(y[te], proba))
        importances.append(rf.feature_importances_)

    imp = np.mean(importances, axis=0)
    sorted_imp = np.sort(imp)[::-1]
    n_for_50 = int(np.searchsorted(np.cumsum(sorted_imp), 0.5) + 1)
    return {
        "condition": name,
        "accuracy": round(float(np.mean(accs)), 3),
        "roc_auc": round(float(np.mean(aucs)), 3),
        "roc_auc_std": round(float(np.std(aucs)), 3),
        "top_feature_importance": round(float(sorted_imp[0]), 4),
        "features_for_50pct_importance": n_for_50,
        "total_features": int(len(imp)),
    }


def main():
    real_imgs = load_real_images()
    if not real_imgs:
        sys.exit("No readable images in %s/real/. Run the earlier pipeline steps first." % COMPRESSED_DIR)
    print("Loaded %d real images" % len(real_imgs))

    real_feats = np.array([features(to_gray_resized(webp_roundtrip(im))) for im in real_imgs])

    rows = []
    for name, fn in CONDITIONS.items():
        print("Running %s ..." % name)
        rows.append(run_condition(name, fn, real_imgs, real_feats))
        print("  ", rows[-1])

    df = pd.DataFrame(rows)
    df.to_csv("noise_experiment_results.csv", index=False)
    print("\n", df.to_string(index=False))
    print("\nSaved noise_experiment_results.csv")


if __name__ == "__main__":
    main()
