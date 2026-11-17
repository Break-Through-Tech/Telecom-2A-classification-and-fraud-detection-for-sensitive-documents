"""
Runs the team's data/model pipeline (generate-fake-2B -> preprocess -> extract-features-2B ->
train-classifier) inside an isolated workspace, so the shared idnet_* folders and CSVs are never
touched. Each run gets its own copy of the stage scripts and config.py (whose paths resolve to the
workspace), a copy of the genuine images, and a sample manifest built from them.

Genuine images must come from a folder named 'positive' (IDNet's genuine-document folder); the
generator refuses sources it cannot verify as genuine, and we do not fake that provenance.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "agent" / "workspaces"
IMAGE_EXT = (".png", ".jpg", ".jpeg")
MIN_IMAGES = 10

# stage name -> script (in dependency order)
STAGES = {
    "generate": "generate-fake-2B.py",
    "preprocess": "preprocess.py",
    "features": "extract-features-2B.py",
    "train": "train-classifier.py",
}
COPY_FILES = ["config.py", *STAGES.values()]
TRAIN_CHOICES = {"feature_set": ("all", "forensic", "hog")}


def _state_path(run_dir):
    return run_dir / "state.json"


def _load_state(run_dir):
    p = _state_path(run_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def _save_state(run_dir, state):
    _state_path(run_dir).write_text(json.dumps(state, indent=1))


def _run_dir(run_id):
    d = (RUNS_DIR / str(run_id)).resolve()
    if RUNS_DIR.resolve() not in d.parents or not d.is_dir():
        raise FileNotFoundError(f"no pipeline run '{run_id}'")
    return d


def start_run(source_folder, document_type="passport"):
    """Create a workspace from a folder of verified-genuine images. Returns {run_id, n_images, ...}."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cfg", ROOT / "config.py")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    if document_type not in cfg.CATEGORIES:
        raise ValueError(f"document_type must be one of {sorted(cfg.CATEGORIES)}")
    src = Path(source_folder).expanduser().resolve()
    if "positive" not in [p.lower() for p in src.parts]:
        raise ValueError("source folder must be IDNet's 'positive' (genuine) folder, e.g. data/sample/GRC/positive")
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if len(files) < MIN_IMAGES:
        raise ValueError(f"need at least {MIN_IMAGES} genuine images, found {len(files)}")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    (run_dir / "logs").mkdir(parents=True)
    for name in COPY_FILES:
        shutil.copy(ROOT / name, run_dir / name)
    dest = run_dir / "idnet_sample" / "real" / document_type
    dest.mkdir(parents=True)
    code = cfg.CATEGORIES[document_type][0]
    rows = []
    for f in files:
        shutil.copy(f, dest / f.name)
        rows.append({"filepath": str(dest / f.name), "document_type": document_type,
                     "archive_member": f"{code}/positive/{f.name}",
                     "source_file_sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(run_dir / "sample_manifest.csv", index=False)
    _save_state(run_dir, {"source": str(src), "document_type": document_type, "n_genuine": len(files), "stages": {}})
    return {"run_id": run_id, "n_genuine_images": len(files), "stages": list(STAGES)}


def run_stage(run_id, stage, timeout=3600, feature_set=None):
    """Run one stage in the workspace. Stages must run in order. Returns status plus the log tail."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {list(STAGES)}")
    run_dir = _run_dir(run_id)
    state = _load_state(run_dir)
    order = list(STAGES)
    prior = order[: order.index(stage)]
    missing = [s for s in prior if state["stages"].get(s, {}).get("status") != "ok"]
    if missing:
        return {"run_id": run_id, "stage": stage, "status": "blocked", "detail": f"run these first: {missing}"}
    cmd = [sys.executable, STAGES[stage]]
    if stage == "train" and feature_set:
        if feature_set not in TRAIN_CHOICES["feature_set"]:
            raise ValueError(f"feature_set must be one of {TRAIN_CHOICES['feature_set']}")
        cmd += ["--feature-set", feature_set]
    log = run_dir / "logs" / f"{stage}.log"
    t0 = time.time()
    try:
        with open(log, "w") as fh:
            proc = subprocess.run(cmd, cwd=run_dir, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
        status = "ok" if proc.returncode == 0 else "failed"
        code = proc.returncode
    except subprocess.TimeoutExpired:
        status, code = "timeout", None
    state["stages"][stage] = {"status": status, "returncode": code, "seconds": round(time.time() - t0, 1)}
    _save_state(run_dir, state)
    tail = log.read_text(errors="replace").strip().splitlines()[-25:]
    return {"run_id": run_id, "stage": stage, "status": status, "seconds": state["stages"][stage]["seconds"],
            "log_tail": "\n".join(tail)}


def run_all(run_id, feature_set=None, timeout=3600):
    out = []
    for stage in STAGES:
        r = run_stage(run_id, stage, timeout=timeout, feature_set=feature_set)
        out.append({k: r[k] for k in ("stage", "status") if k in r} | {"seconds": r.get("seconds")})
        if r["status"] != "ok":
            out[-1]["log_tail"] = r.get("log_tail", r.get("detail"))
            break
    return out


def pipeline_status(run_id):
    run_dir = _run_dir(run_id)
    state = _load_state(run_dir)
    state["artifacts"] = sorted(str(p.relative_to(run_dir)) for p in run_dir.glob("*.csv"))
    return state


def list_runs():
    if not RUNS_DIR.exists():
        return []
    return sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir())
