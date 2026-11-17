"""
Fraud-detection agent, built with LangGraph and a local Ollama model.

Two graphs:
  pipeline graph  classify (CNN) -> check_llm -> summarize_llm | summarize_offline
                  The CNN always makes the decision. If Ollama is not running, the offline summary node
                  runs instead, so the pipeline still works.
  chat graph      agent <-> tools loop (ReAct style). The LLM chooses tools from agent/tools.py to answer
                  questions; the tools themselves are plain deterministic Python.

CLI:  python -m agent.agent scan <folder>
      python -m agent.agent ask "how many documents were rejected?"
      python -m agent.agent graph        (prints both graphs as Mermaid diagrams)
"""

import json
import os
import sys
from typing import Annotated, TypedDict

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent import tools

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")  # any Ollama model that supports tool calling
MAX_STEPS = 5  # max tool-calling rounds in the chat graph

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


def _llm(model, url, with_tools):
    llm = ChatOllama(model=model, base_url=url, temperature=0)
    return llm.bind_tools(tools.TOOL_SCHEMAS) if with_tools else llm


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


# ---------------------------------------------------------------------------------------------
# Pipeline graph
# ---------------------------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    paths: list
    model: str
    url: str
    weights: str
    results: list
    llm_ready: bool
    summary: str
    llm_used: bool


def _classify(state):
    results = [tools.classify_document(p, state["weights"]) for p in state["paths"]]
    tools.log_results(results)
    return {"results": results}


def _check_llm(state):
    running, installed, _ = ollama_status(state["url"], state["model"])
    return {"llm_ready": bool(running and installed)}


def _route_llm(state):
    return "summarize_llm" if state["llm_ready"] else "summarize_offline"


def _summarize_llm(state):
    prompt = ("Write a 2-3 sentence summary for the analyst of these classifier results. State which "
              "documents need action and why. Results:\n" + json.dumps(state["results"], indent=1))
    try:
        msg = _llm(state["model"], state["url"], False).invoke(
            [SystemMessage(SYSTEM), HumanMessage(prompt)])
        return {"summary": msg.content.strip(), "llm_used": True}
    except Exception as e:
        return {"summary": summarize_offline(state["results"]) + f"\n(LLM unavailable: {e})", "llm_used": False}


def _summarize_offline(state):
    return {"summary": summarize_offline(state["results"]), "llm_used": False}


def build_pipeline_graph():
    g = StateGraph(PipelineState)
    g.add_node("classify", _classify)
    g.add_node("check_llm", _check_llm)
    g.add_node("summarize_llm", _summarize_llm)
    g.add_node("summarize_offline", _summarize_offline)
    g.add_edge(START, "classify")
    g.add_edge("classify", "check_llm")
    g.add_conditional_edges("check_llm", _route_llm, ["summarize_llm", "summarize_offline"])
    g.add_edge("summarize_llm", END)
    g.add_edge("summarize_offline", END)
    return g.compile()


_pipeline_graph = None


def pipeline(paths, model=OLLAMA_MODEL, url=OLLAMA_URL, weights=tools.DEFAULT_WEIGHTS):
    """Classify documents, log them, and explain. Returns (results, summary, llm_used)."""
    global _pipeline_graph
    _pipeline_graph = _pipeline_graph or build_pipeline_graph()
    out = _pipeline_graph.invoke({"paths": [str(p) for p in paths], "model": model, "url": url,
                                  "weights": str(weights)})
    return out["results"], out["summary"], out["llm_used"]


# ---------------------------------------------------------------------------------------------
# Chat graph: LLM <-> tools loop
# ---------------------------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def _tool_node(state):
    out = []
    for call in state["messages"][-1].tool_calls:
        name, args = call["name"], call.get("args") or {}
        try:
            result = tools.TOOL_FUNCS[name](**args)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        out.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"], name=name))
    return {"messages": out}


def _route_agent(state):
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else END


def build_chat_graph(model=OLLAMA_MODEL, url=OLLAMA_URL):
    llm = _llm(model, url, True)
    g = StateGraph(ChatState)
    g.add_node("agent", lambda state: {"messages": [llm.invoke(state["messages"])]})
    g.add_node("tools", _tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", _route_agent, ["tools", END])
    g.add_edge("tools", "agent")
    return g.compile()


def chat(question, model=OLLAMA_MODEL, url=OLLAMA_URL):
    """Tool-calling loop. Returns (answer, trace) where trace lists the tool calls made."""
    graph = build_chat_graph(model, url)
    inputs = {"messages": [SystemMessage(SYSTEM), HumanMessage(question)]}
    config = {"recursion_limit": 2 * MAX_STEPS + 1}
    try:
        state = graph.invoke(inputs, config)
    except GraphRecursionError:
        return "Stopped after too many tool calls.", []
    trace = [{"tool": c["name"], "args": c.get("args") or {}}
             for m in state["messages"] if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
    return state["messages"][-1].content.strip(), trace


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("scan", "ask", "graph"):
        sys.exit(__doc__)
    if sys.argv[1] == "graph":
        print("%% pipeline graph\n" + build_pipeline_graph().get_graph().draw_mermaid())
        print("%% chat graph\n" + build_chat_graph().get_graph().draw_mermaid())
    elif sys.argv[1] == "scan":
        folder = tools.safe_path(sys.argv[2])
        files = sorted(str(x) for x in folder.iterdir() if x.suffix.lower() in tools.IMAGE_EXT)
        print(pipeline(files)[1])
    else:
        answer, trace = chat(" ".join(sys.argv[2:]))
        print(answer)
        print("tools used:", [t["tool"] for t in trace])
