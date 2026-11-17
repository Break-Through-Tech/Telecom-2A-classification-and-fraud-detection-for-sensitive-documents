"""
Tools the fraud agent can call. Everything here is deterministic: the CNN classifier makes the
fraud decision, the LLM (Ollama) only decides which tool to run and explains the result.
"""

import importlib.util
import json
import os
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "deep learning algorithms"
DEFAULT_WEIGHTS = Path(os.environ.get("FRAUD_MODEL", MODEL_DIR / "final_r50.pt"))
UPLOAD_DIR = ROOT / "agent" / "uploads"
LOG_PATH = ROOT / "agent" / "runs" / "scan_log.jsonl"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")

# the classifier was trained on 1136x803 Greek passports only, so flag anything else
EXPECTED_SIZE = (1136, 803)
# decision policy: auto-decide only when the model is confident, send the middle to a human
APPROVE_BELOW = 0.2
REJECT_ABOVE = 0.8

# the LLM may only read files under these folders
ALLOWED_ROOTS = [ROOT.resolve(), UPLOAD_DIR.resolve()]

_clf = {}


def _load_classifier(weights):
    weights = str(weights)
    if weights not in _clf:
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"model weights not found at {weights}. Train them with: cd 'deep learning algorithms' && "
                "python train_cnn.py --arch resnet50 --final --name final_r50")
        spec = importlib.util.spec_from_file_location("predict", MODEL_DIR / "predict.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _clf[weights] = mod.ForgeryClassifier(weights)
    return _clf[weights]


def safe_path(path):
    p = Path(path).expanduser().resolve()
    if not any(p == r or r in p.parents for r in ALLOWED_ROOTS):
        raise PermissionError(f"{path} is outside the allowed folders")
    return p


def decide(fake_probability):
    if fake_probability < APPROVE_BELOW:
        return "approve"
    if fake_probability > REJECT_ABOVE:
        return "reject"
    return "manual_review"


def classify_document(path, weights=DEFAULT_WEIGHTS):
    """Validate an image, score it with the CNN, and return a verdict dict."""
    t0 = time.time()
    result = {"file": str(path), "warnings": []}
    try:
        p = safe_path(path)
        if p.suffix.lower() not in IMAGE_EXT:
            raise ValueError(f"unsupported file type {p.suffix}")
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            result["size"] = list(im.size)
        if tuple(result["size"]) != EXPECTED_SIZE:
            result["warnings"].append(
                f"image is {result['size'][0]}x{result['size'][1]}, the model was trained on "
                f"{EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]} Greek passports; treat the score with caution")
        pred = _load_classifier(weights).predict(p)
        result.update(fake_probability=pred["fake_probability"], model_label=pred["label"],
                      decision=decide(pred["fake_probability"]))
    except Exception as e:  # report instead of crashing a batch
        result.update(error=f"{type(e).__name__}: {e}", decision="error")
    result["seconds"] = round(time.time() - t0, 2)
    return result


def scan_folder(folder, weights=DEFAULT_WEIGHTS, limit=200):
    """Classify every image in a folder (non-recursive) and log each result."""
    f = safe_path(folder)
    files = sorted(x for x in f.iterdir() if x.suffix.lower() in IMAGE_EXT)[:limit]
    results = [classify_document(x, weights) for x in files]
    log_results(results)
    return results


def log_results(results):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(LOG_PATH, "a") as fh:
        for r in results:
            fh.write(json.dumps({"time": stamp, **r}) + "\n")


def scan_summary():
    """Counts of every decision ever logged, for ad-hoc analytics questions."""
    if not LOG_PATH.exists():
        return {"total": 0}
    rows = [json.loads(x) for x in LOG_PATH.read_text().splitlines() if x.strip()]
    counts = {}
    for r in rows:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    flagged = sorted((r for r in rows if r.get("fake_probability") is not None),
                     key=lambda r: -r["fake_probability"])[:5]
    return {"total": len(rows), "by_decision": counts,
            "most_suspicious": [{"file": r["file"], "fake_probability": r["fake_probability"]} for r in flagged]}


def model_report():
    """Cross-validation results of the trained models (from the saved result files)."""
    out = {}
    for name in ("r50", "r50_jpg90", "results", "effb0"):
        fp = MODEL_DIR / f"{name}.json"
        if fp.exists():
            r = json.loads(fp.read_text())
            out[name] = {k: round(r[k], 3) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
            out[name]["recall_by_forgery_type"] = r["recall_by_forgery_type"]
    return out


TOOL_FUNCS = {"classify_document": classify_document, "scan_folder": scan_folder,
              "scan_summary": scan_summary, "model_report": model_report}

# schemas in the format Ollama's /api/chat expects
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "classify_document",
        "description": "Check one identity document image and return fake_probability and a decision (approve, manual_review, reject).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "path to the image"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "scan_folder",
        "description": "Check every image in a folder and return one result per image.",
        "parameters": {"type": "object", "properties": {"folder": {"type": "string"}}, "required": ["folder"]}}},
    {"type": "function", "function": {
        "name": "scan_summary",
        "description": "Counts of approve/manual_review/reject/error decisions from all past scans, plus the most suspicious files.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "model_report",
        "description": "Accuracy, precision, recall, F1 and ROC-AUC of the trained classifiers from cross-validation.",
        "parameters": {"type": "object", "properties": {}}}},
]
