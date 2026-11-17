"""
Streamlit UI for the fraud-detection agent.   Run from the repo root:  streamlit run agent/app.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import agent, tools  # noqa: E402

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


tab_check, tab_scan, tab_ask, tab_log = st.tabs(["Check documents", "Scan a folder", "Ask the agent", "History"])

with tab_check:
    files = st.file_uploader("Upload one or more document images", type=["png", "jpg", "jpeg", "webp"],
                             accept_multiple_files=True)
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
