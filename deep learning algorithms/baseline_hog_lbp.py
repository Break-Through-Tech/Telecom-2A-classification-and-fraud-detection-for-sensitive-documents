"""
Classical baseline: HOG + LBP features with SVM / random forest / logistic regression.
Same data, same grouped 5-fold split and metrics as train_cnn.py so the numbers are comparable.
"""

import argparse
import json
import os

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import hog, local_binary_pattern

from train_cnn import OUT, SEED, load_paths


def features(path, size, reencode):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if reencode:
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, reencode])[1], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    h = hog(img, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2))
    lbp = local_binary_pattern(img, 8, 1, "uniform")
    # LBP histogram per 4x4 grid cell keeps some spatial info about where texture changes
    gh, gw = img.shape[0] // 4, img.shape[1] // 4
    hists = [np.bincount(lbp[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw].astype(int).ravel(), minlength=10) / (gh * gw)
             for i in range(4) for j in range(4)]
    return np.concatenate([h, *hists])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reencode", type=int, default=0)
    ap.add_argument("--name", default="baseline_hog_lbp")
    args = ap.parse_args()
    paths, labels, groups = load_paths()
    X = np.array([features(p, (568, 402), args.reencode) for p in paths])
    print("feature matrix", X.shape, flush=True)
    models = {
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=10, probability=True, class_weight="balanced", random_state=SEED)),
        "random_forest": RandomForestClassifier(500, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced")),
    }
    skf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    splits = list(skf.split(X, labels, groups))
    results = {}
    for name, model in models.items():
        probs = np.zeros(len(paths))
        for tr, te in splits:
            model.fit(X[tr], labels[tr])
            probs[te] = model.predict_proba(X[te])[:, 1]
        pred = (probs >= 0.5).astype(int)
        per_type = {}
        for t in sorted({os.path.basename(os.path.dirname(p)) for p, l in zip(paths, labels) if l == 1}):
            idx = [i for i, p in enumerate(paths) if labels[i] == 1 and os.path.basename(os.path.dirname(p)) == t]
            per_type[t] = float((pred[idx] == 1).mean())
        results[name] = {
            "accuracy": accuracy_score(labels, pred), "precision": precision_score(labels, pred, zero_division=0),
            "recall": recall_score(labels, pred), "f1": f1_score(labels, pred), "roc_auc": roc_auc_score(labels, probs),
            "confusion_matrix [[TN,FP],[FN,TP]]": confusion_matrix(labels, pred).tolist(),
            "recall_by_forgery_type": per_type, "args": {"model": name, "reencode": args.reencode, "folds": 5}}
        np.savez(f"{OUT}/{args.name}_{name}_probs.npz", probs=probs, labels=labels, paths=np.array(paths))
        with open(f"{OUT}/{args.name}_{name}.json", "w") as f:
            json.dump(results[name], f, indent=2)
        print(name, {k: round(v, 3) for k, v in results[name].items() if isinstance(v, float)}, per_type, flush=True)


if __name__ == "__main__":
    main()
