"""
Scores documents with every classical model defined in the team's train-classifier.py
(generate-fake-test branch), so they can be compared with the CNN.

train_all(run_id): fits each model on the development rows of a pipeline run's feature CSV, scores
it on that run's reserved test split, and saves the fitted models to agent/models/classical/.
score_image(path): extracts the same 295 'team' features from one image and asks every model.

These models learn the pipeline's synthetic tamper types (rotation, blur, copy-move, splice, text),
NOT IDNet's real forgeries that the CNN was trained on, so they can disagree with the CNN.

PyTorch and XGBoost/LightGBM each ship their own OpenMP runtime and crash if loaded into one process on
macOS, so train_all() and score_images() run the work in a separate Python process (this module's
__main__) and never import xgboost/lightgbm in the caller.
"""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "agent" / "models" / "classical"
META_PATH = MODEL_DIR / "metadata.json"
WORKSPACES = ROOT / "agent" / "workspaces"

_mods = {}


def _load(script, name):
    """Import one of the team's hyphen-named scripts straight from the repo root."""
    if name not in _mods:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        spec = importlib.util.spec_from_file_location(name, ROOT / script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _mods[name] = mod
    return _mods[name]


def _slug(name):
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def _latest_features():
    runs = sorted(p for p in WORKSPACES.glob("*/idnet_features.csv")) if WORKSPACES.exists() else []
    return runs[-1] if runs else None


def is_trained():
    return META_PATH.exists()


def metadata():
    return json.loads(META_PATH.read_text()) if META_PATH.exists() else None


def _train_all_impl(features_path=None):
    """Fit all models on development rows, evaluate on the reserved test rows, save everything."""
    features_path = Path(features_path) if features_path else _latest_features()
    if features_path is None or not features_path.exists():
        raise FileNotFoundError("no feature file yet: run the data pipeline first (generate -> features)")
    tc = _load("train-classifier.py", "team_train")
    X, y, meta = tc.load_features("all", str(features_path))
    dev = meta["split"].eq("development")
    test = meta["split"].eq("test")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, model in tc.build_models().items():
        t0 = time.time()
        model.fit(X.loc[dev], y.loc[dev])
        pred = model.predict(X.loc[test])
        try:
            scores = tc.get_decision_scores(model, X.loc[test])
            auc = float(roc_auc_score(y.loc[test], scores))
        except Exception:
            auc = None
        results[name] = {
            "file": f"{_slug(name)}.joblib",
            "test_accuracy": round(float(accuracy_score(y.loc[test], pred)), 3),
            "test_precision": round(float(precision_score(y.loc[test], pred, zero_division=0)), 3),
            "test_recall": round(float(recall_score(y.loc[test], pred, zero_division=0)), 3),
            "test_f1": round(float(f1_score(y.loc[test], pred, zero_division=0)), 3),
            "test_roc_auc": None if auc is None else round(auc, 3),
            "has_probability": hasattr(model, "predict_proba"),
            "fit_seconds": round(time.time() - t0, 1),
        }
        joblib.dump(model, MODEL_DIR / results[name]["file"])
    info = {"trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "features_file": str(features_path),
            "feature_columns": list(X.columns), "n_train": int(dev.sum()), "n_test": int(test.sum()),
            "fake_types": sorted(meta.loc[meta["label"].astype(str).str.lower().eq("fake"), "tamper_type"].unique()),
            "models": results}
    META_PATH.write_text(json.dumps(info, indent=1))
    return info


def _score_impl(path):
    """Return {model name: {label, fake_probability or None}} for one image, or {'error': ...}."""
    info = metadata()
    if info is None:
        return {"error": "classical models are not trained yet (Models tab -> Train)"}
    try:
        color = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if color is None:
            raise ValueError("unreadable image")
        feats = _load("extract-features-2B.py", "team_features").extract_team_features(color)
        row = pd.DataFrame([feats])[info["feature_columns"]]
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    out = {}
    for name, m in info["models"].items():
        model = joblib.load(MODEL_DIR / m["file"])
        label = "fake" if int(model.predict(row)[0]) == 1 else "genuine"
        prob = float(model.predict_proba(row)[0, 1]) if m["has_probability"] else None
        out[name] = {"label": label, "fake_probability": None if prob is None else round(prob, 3)}
    return out


# ---- public API: run the work in a clean subprocess -------------------------------------------

_MARK = "@@JSON@@"


def _worker(*args, timeout=1800):
    proc = subprocess.run([sys.executable, "-m", "agent.classical", *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_MARK):
            return json.loads(line[len(_MARK):])
    raise RuntimeError(f"classical worker failed (exit {proc.returncode}): {proc.stderr.strip()[-600:]}")


def train_all(features_path=None):
    return _worker("train", *( [features_path] if features_path else [] ))


def score_images(paths):
    """{path: {model: {label, fake_probability}} or {'error': ...}} for many images in one subprocess call."""
    return _worker("score", *paths) if paths else {}


def score_image(path):
    info = metadata()
    if info is None:
        return {"error": "classical models are not trained yet (Models tab -> Train)"}
    return score_images([str(path)])[str(path)]


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "train":
        result = _train_all_impl(rest[0] if rest else None)
    else:
        result = {p: _score_impl(p) for p in rest}
    print(_MARK + json.dumps(result))
