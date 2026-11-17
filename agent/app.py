"""
Streamlit UI for the fraud-detection agent.   Run from the repo root:  streamlit run agent/app.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import agent, classical, tools  # noqa: E402

ICON = {"approve": "✅ approve", "manual_review": "⚠️ manual review", "reject": "❌ reject", "error": "🚫 error"}

st.set_page_config(page_title="ID fraud detection agent", layout="wide")
st.title("Identity document fraud detection")

with st.sidebar:
    st.header("Ollama (local LLM)")
    model = st.text_input("model", agent.OLLAMA_MODEL)
    running, installed, msg = agent.ollama_status(model=model)
    (st.success if running and installed else st.warning)(msg)
    st.caption("The CNN makes the fraud decision. Ollama only explains results and answers questions, "
               "and runs locally so document images never leave this machine.")
    st.header("Classifier")
    st.write(f"weights: `{Path(tools.DEFAULT_WEIGHTS).name}`" +
             ("" if Path(tools.DEFAULT_WEIGHTS).exists() else "  — **missing**"))
    st.write(f"approve below {tools.APPROVE_BELOW}, reject above {tools.REJECT_ABOVE}, otherwise manual review")
    st.caption("Trained on 75 genuine + 75 forged Greek passports only. Other document types or countries are out of scope.")


def show_results(results):
    df = pd.DataFrame([{
        "file": Path(r["file"]).name,
        "decision": ICON[r["decision"]],
        "fake probability": r.get("fake_probability"),
        "notes": "; ".join(r["warnings"] + ([r["error"]] if r.get("error") else [])),
    } for r in results])
    st.dataframe(df, use_container_width=True, hide_index=True)


tab_check, tab_scan, tab_pipe, tab_models, tab_ask, tab_log = st.tabs(
    ["Check documents", "Scan a folder", "Run data pipeline", "Models", "Ask the agent", "History"])

with tab_check:
    files = st.file_uploader("Upload one or more document images", type=["png", "jpg", "jpeg", "webp"],
                             accept_multiple_files=True)
    use_all = st.checkbox("Also score with the generate-fake-test classical models (12)", value=classical.is_trained(),
                          disabled=not classical.is_trained(),
                          help="Train them first in the Models tab." if not classical.is_trained() else None)
    if files and st.button("Run pipeline", key="run_upload"):
        tools.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in files:
            p = tools.UPLOAD_DIR / f"{int(time.time())}_{Path(f.name).name}"
            p.write_bytes(f.getbuffer())
            paths.append(str(p))
        with st.spinner("classifying..."):
            results, summary, llm_used = agent.pipeline(paths, model=model)
        st.subheader("Summary" + (" (written by Ollama)" if llm_used else " (offline summary, Ollama not used)"))
        st.write(summary)
        show_results(results)
        cols = st.columns(min(len(paths), 4))
        for col, p, r in zip(cols, paths, results):
            col.image(p, caption=f"{ICON[r['decision']]}  {r.get('fake_probability', '')}")
        if use_all:
            st.subheader("All models")
            rows = {}
            scored = classical.score_images(paths)
            for p, r in zip(paths, results):
                name = Path(p).name.split("_", 1)[-1][:34]
                cell = {"CNN (ResNet50)": f"{r['decision']} ({r.get('fake_probability')})" if r["decision"] != "error" else "error"}
                for model, v in scored[p].items():
                    if model == "error":
                        cell["classical"] = v
                    else:
                        cell[model] = v["label"] + ("" if v["fake_probability"] is None else f" ({v['fake_probability']})")
                rows[name] = cell
            table = pd.DataFrame(rows).T
            st.dataframe(table, use_container_width=True)
            st.caption("The CNN learned IDNet's real forgeries; the classical models learned the pipeline's synthetic "
                       "tampering, so they can disagree.")

with tab_scan:
    folder = st.text_input("Folder inside the repo", "data/GRC/fraud3_face_replacement")
    n = st.number_input("max images", 1, 200, 10)
    if st.button("Scan folder", key="run_scan"):
        try:
            d = tools.safe_path(Path(tools.ROOT) / folder)
            paths = sorted(str(x) for x in d.iterdir() if x.suffix.lower() in tools.IMAGE_EXT)[: int(n)]
            with st.spinner(f"classifying {len(paths)} images..."):
                results, summary, llm_used = agent.pipeline(paths, model=model)
            st.write(summary)
            show_results(results)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

with tab_pipe:
    st.caption("Runs the team's pipeline (generate fakes → preprocess → extract features → train) in an isolated "
               "workspace under agent/workspaces/, so shared data folders are never overwritten.")
    src = st.text_input("Folder of genuine IDNet images (must be a 'positive' folder)", "data/sample/GRC/positive")
    if st.button("Run full pipeline", key="run_pipe"):
        try:
            run = tools.start_pipeline_run(src)
            st.write(f"workspace `{run['run_id']}` with {run['n_genuine_images']} genuine images")
            for stage in ("generate", "preprocess", "features", "train"):
                with st.spinner(f"running {stage}..."):
                    r = tools.run_pipeline_stage(run["run_id"], stage)
                (st.success if r["status"] == "ok" else st.error)(f"{stage}: {r['status']} in {r.get('seconds')}s")
                with st.expander(f"{stage} log"):
                    st.code(r.get("log_tail", r.get("detail", "")))
                if r["status"] != "ok":
                    break
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
    st.write("existing runs:", tools.list_pipeline_runs() or "none")

with tab_models:
    st.caption("Trains every model defined in the team's train-classifier.py (generate-fake-test branch) so uploads can be "
               "scored by all of them next to the CNN. If no pipeline run exists yet, one is run first (about 40 s).")
    if st.button("Train / retrain classical models", key="train_models"):
        with st.spinner("running pipeline if needed and training 12 models..."):
            try:
                st.success(f"trained {len(tools.train_classical_models()['trained_models'])} models")
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
    info = classical.metadata()
    if info:
        st.write(f"trained {info['trained_at']} on {info['n_train']} images, held-out test {info['n_test']} images; "
                 f"fake types learned: {', '.join(info['fake_types'])}")
        table = pd.DataFrame(info["models"]).T[["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]]
        st.dataframe(table, use_container_width=True)
        st.caption("Held-out test split of the pipeline run (30 images), synthetic fakes only. Small sample: treat as rough.")
    else:
        st.info("Classical models are not trained yet.")
    st.subheader("CNN (cross-validation on real IDNet forgeries)")
    cnn = tools.model_report()
    if cnn:
        st.dataframe(pd.DataFrame({k: {m: v[m] for m in ("accuracy", "precision", "recall", "f1", "roc_auc")}
                                   for k, v in cnn.items()}).T, use_container_width=True)
    st.caption("Different data and protocol from the table above, so the two are not directly comparable.")

with tab_ask:
    st.caption("Ask things like: 'how many documents were rejected so far?' or 'how accurate is the model?'")
    q = st.text_input("question")
    if q and st.button("Ask", key="ask"):
        if not (running and installed):
            st.error(msg)
        else:
            with st.spinner("agent is working..."):
                try:
                    answer, trace = agent.chat(q, model=model)
                    st.write(answer)
                    st.caption("tools called: " + (", ".join(t["tool"] for t in trace) or "none"))
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")

with tab_log:
    st.json(tools.scan_summary())
