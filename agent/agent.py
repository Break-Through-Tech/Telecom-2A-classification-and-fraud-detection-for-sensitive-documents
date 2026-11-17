"""
Fraud-detection agent.

pipeline(paths): always runs the CNN on every document (deterministic), logs the results, then
asks the local Ollama model to write a short explanation for the reviewer.
chat(question):  lets the Ollama model call the tools in tools.py to answer ad-hoc questions.

If Ollama is not running, the pipeline still works and returns a plain-text summary instead.

CLI:  python -m agent.agent scan <folder>
      python -m agent.agent ask "how many documents were rejected?"
"""

import json
import os
import sys

import requests

from agent import tools

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")  # any Ollama model that supports tool calling
MAX_STEPS = 5

SYSTEM = (
    "You assist a Verizon fraud analyst checking identity documents. A CNN classifier makes the fraud "
    "decision; you never override it and never invent scores. Use the tools to get facts, then answer "
    "briefly. The classifier was only trained on Greek passports, so say so when results look out of scope. "
    "To run the data/model pipeline, call run_data_pipeline once; it does everything."
)


def ollama_status(url=OLLAMA_URL, model=OLLAMA_MODEL):
    """Return (running, model_installed, message)."""
    try:
        names = [m["name"] for m in requests.get(f"{url}/api/tags", timeout=3).json().get("models", [])]
    except Exception:
        return False, False, f"Ollama is not running at {url} (start it with `ollama serve`)"
    if not any(n == model or n.split(":")[0] == model for n in names):
        return True, False, f"Ollama is running but model '{model}' is not installed (run `ollama pull {model}`)"
    return True, True, f"Ollama running, model {model} ready"


def _chat(messages, model, url, use_tools=True):
    body = {"model": model, "messages": messages, "stream": False}
    if use_tools:
        body["tools"] = tools.TOOL_SCHEMAS
    r = requests.post(f"{url}/api/chat", json=body, timeout=180)
    r.raise_for_status()
    return r.json()["message"]


def summarize_offline(results):
    c = {}
    for r in results:
        c[r["decision"]] = c.get(r["decision"], 0) + 1
    lines = [f"{len(results)} document(s): " + ", ".join(f"{v} {k}" for k, v in sorted(c.items()))]
    for r in results:
        if r["decision"] == "error":
            lines.append(f"- {os.path.basename(r['file'])}: error ({r['error']})")
        else:
            lines.append(f"- {os.path.basename(r['file'])}: {r['decision']} (fake probability {r['fake_probability']:.2f})")
    return "\n".join(lines)


def pipeline(paths, model=OLLAMA_MODEL, url=OLLAMA_URL, weights=tools.DEFAULT_WEIGHTS):
    """Classify documents, log them, and explain. Returns (results, summary, llm_used)."""
    results = [tools.classify_document(p, weights) for p in paths]
    tools.log_results(results)
    running, installed, _ = ollama_status(url, model)
    if not (running and installed):
        return results, summarize_offline(results), False
    prompt = ("Write a 2-3 sentence summary for the analyst of these classifier results. State which "
              "documents need action and why. Results:\n" + json.dumps(results, indent=1))
    try:
        msg = _chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                    model, url, use_tools=False)
        return results, msg["content"].strip(), True
    except Exception as e:
        return results, summarize_offline(results) + f"\n(LLM unavailable: {e})", False


def chat(question, model=OLLAMA_MODEL, url=OLLAMA_URL):
    """Tool-calling loop. Returns (answer, trace) where trace lists the tool calls made."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    trace = []
    for _ in range(MAX_STEPS):
        msg = _chat(messages, model, url)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg["content"].strip(), trace
        for call in calls:
            name, args = call["function"]["name"], call["function"].get("arguments") or {}
            try:
                out = tools.TOOL_FUNCS[name](**args)
            except Exception as e:
                out = {"error": f"{type(e).__name__}: {e}"}
            trace.append({"tool": name, "args": args})
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(out)})
    return "Stopped after too many tool calls.", trace


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("scan", "ask"):
        sys.exit(__doc__)
    if sys.argv[1] == "scan":
        folder = tools.safe_path(sys.argv[2])
        files = sorted(str(x) for x in folder.iterdir() if x.suffix.lower() in tools.IMAGE_EXT)
        _, summary, _ = pipeline(files)
        print(summary)
    else:
        answer, trace = chat(" ".join(sys.argv[2:]))
        print(answer)
        print("tools used:", [t["tool"] for t in trace])
