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


def classify_all_models(path, weights=DEFAULT_WEIGHTS):
    """Score one document with the CNN and with every classical model from the generate-fake-test branch."""
    from agent import classical
    cnn = classify_document(path, weights)
    out = {"file": cnn["file"], "cnn": {k: cnn.get(k) for k in ("decision", "fake_probability", "error")}}
    if cnn.get("error"):
        return out
    classic = classical.score_image(safe_path(path))
    out["classical"] = classic
    if "error" not in classic:
        votes = sum(v["label"] == "fake" for v in classic.values())
        out["classical_fake_votes"] = f"{votes}/{len(classic)}"
    return out


def train_classical_models():
    """Train and save every classical model from train-classifier.py (runs the data pipeline first if needed)."""
    from agent import classical
    ran_pipeline = None
    if classical._latest_features() is None:
        ran_pipeline = run_data_pipeline()
        if not ran_pipeline["succeeded"]:
            return {"error": "data pipeline failed", "pipeline": ran_pipeline}
    info = classical.train_all()
    return {"trained_models": list(info["models"]), "n_train": info["n_train"], "n_test": info["n_test"],
            "fake_types_learned": info["fake_types"], "ran_pipeline": bool(ran_pipeline)}


def compare_models():
    """CNN cross-validation metrics next to the classical models' held-out test metrics."""
    from agent import classical
    info = classical.metadata()
    classical_scores = None if info is None else {
        k: {m: v[m] for m in ("test_accuracy", "test_f1", "test_roc_auc")} for k, v in info["models"].items()}
    best = None
    if classical_scores:
        name = max(classical_scores, key=lambda k: classical_scores[k]["test_f1"])
        best = {"name": name, **classical_scores[name]}
    return {"best_classical_model_by_f1": best,
            "note": "CNN and classical models were evaluated on different data, so do not rank one against the other",
            "cnn_cross_validation_real_forgeries": model_report(),
            "classical_held_out_test_synthetic_fakes": classical_scores}


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


def start_pipeline_run(source_folder, document_type="passport"):
    """Create an isolated workspace for the data/model pipeline from a folder of genuine images."""
    from agent import pipeline
    return pipeline.start_run(safe_path(Path(ROOT) / source_folder), document_type)


def run_pipeline_stage(run_id, stage, feature_set=None):
    """Run one pipeline stage (generate, preprocess, features, train) in a workspace."""
    from agent import pipeline
    return pipeline.run_stage(run_id, stage, feature_set=feature_set)


def run_full_pipeline(run_id, feature_set=None):
    """Run all four stages in order, stopping at the first failure."""
    from agent import pipeline
    return pipeline.run_all(run_id, feature_set=feature_set)


def run_data_pipeline(source_folder="data/sample/GRC/positive", document_type="passport", feature_set=None):
    """One call: create a workspace from genuine images and run every stage. Returns the run_id and each stage's result."""
    run = start_pipeline_run(source_folder, document_type)
    from agent import pipeline
    steps = pipeline.run_all(run["run_id"], feature_set=feature_set)
    ok = all(x["status"] == "ok" for x in steps) and len(steps) == len(pipeline.STAGES)
    return {"run_id": run["run_id"], "n_genuine_images": run["n_genuine_images"],
            "succeeded": ok, "stages": steps}


def pipeline_status(run_id):
    from agent import pipeline
    return pipeline.pipeline_status(run_id)


def list_pipeline_runs():
    from agent import pipeline
    return pipeline.list_runs()


TOOL_FUNCS = {"classify_document": classify_document, "scan_folder": scan_folder,
              "scan_summary": scan_summary, "model_report": model_report,
              "start_pipeline_run": start_pipeline_run, "run_pipeline_stage": run_pipeline_stage,
              "run_full_pipeline": run_full_pipeline, "run_data_pipeline": run_data_pipeline,
              "classify_all_models": classify_all_models, "train_classical_models": train_classical_models,
              "compare_models": compare_models, "pipeline_status": pipeline_status,
              "list_pipeline_runs": list_pipeline_runs}

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
        "description": "Metrics of the CNN ONLY (not the classical models). For any question about which model is best or about the classical models, use compare_models instead.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "run_data_pipeline",
        "description": "Run the whole data and model pipeline in one call (generate fakes, preprocess, extract features, train) on a folder of genuine IDNet images, in an isolated workspace. Returns run_id, whether it succeeded, and each stage's status.",
        "parameters": {"type": "object", "properties": {
            "source_folder": {"type": "string", "description": "folder of genuine images, default data/sample/GRC/positive"},
            "document_type": {"type": "string", "description": "passport, drivers_license or national_id"}}}}},
    {"type": "function", "function": {
        "name": "classify_all_models",
        "description": "Check one document with the CNN and all 12 classical models and report how many say fake.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "train_classical_models",
        "description": "Train and save all classical models from the generate-fake-test branch (runs the data pipeline first if needed).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "compare_models",
        "description": "Accuracy, F1 and ROC-AUC of the CNN AND of each of the 12 classical models. Use this for questions like which model is best or how accurate the models are.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "pipeline_status",
        "description": "Stage statuses and output files for a pipeline run.",
        "parameters": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]}}},
    {"type": "function", "function": {
        "name": "list_pipeline_runs",
        "description": "List existing pipeline run ids.",
        "parameters": {"type": "object", "properties": {}}}},
]
