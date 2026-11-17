"""
Smoke tests for the agent pipeline. Run from the repo root:  python agent/test_agent.py
Uses a fake Ollama server, so it checks the plumbing (tools, decisions, tool-calling loop, UI),
not the quality of any real LLM. Documents that were in training score trivially well.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import agent, tools  # noqa: E402

ROOT = tools.ROOT
GENUINE = sorted((ROOT / "data/sample/GRC/positive").glob("*.png"))[:3]
FORGED = sorted((ROOT / "data/GRC/fraud3_face_replacement").glob("*.png"))[:3]


class FakeOllama(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"models": [{"name": "fake:1b"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.calls += 1
        if req.get("tools") and not any(m["role"] == "tool" for m in req["messages"]):
            self._send({"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "scan_summary", "arguments": {}}}]}})
        elif req.get("tools"):
            tool_out = [m for m in req["messages"] if m["role"] == "tool"][-1]["content"]
            self._send({"message": {"role": "assistant", "content": f"scan summary was: {tool_out}"}})
        else:
            self._send({"message": {"role": "assistant", "content": "LLM summary: review flagged documents."}})


def main():
    assert GENUINE and FORGED, "sample images missing (data/sample/GRC/positive, data/GRC)"

    # 1. classifier + decision policy
    g = tools.classify_document(GENUINE[0])
    f = tools.classify_document(FORGED[0])
    print("genuine:", g["decision"], g["fake_probability"], "| forged:", f["decision"], f["fake_probability"])
    assert g["decision"] == "approve" and f["decision"] == "reject"

    # 2. bad input is reported, not crashed; files outside the repo are refused
    assert tools.classify_document(ROOT / "README.md")["decision"] == "error"
    assert "outside the allowed" in tools.classify_document("/etc/hosts")["error"]

    # 3. offline pipeline (no Ollama)
    results, summary, used = agent.pipeline([str(p) for p in GENUINE + FORGED], url="http://localhost:1")
    assert not used and len(results) == 6
    print(summary)

    # 4. with a (fake) Ollama: explanation + tool-calling loop
    server = HTTPServer(("localhost", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://localhost:{server.server_port}"
    assert agent.ollama_status(url, "fake:1b")[:2] == (True, True)
    assert agent.ollama_status(url, "other")[:2] == (True, False)
    _, summary, used = agent.pipeline([str(FORGED[0])], model="fake:1b", url=url)
    assert used and summary.startswith("LLM summary")
    answer, trace = agent.chat("how many documents were rejected?", model="fake:1b", url=url)
    assert trace[0]["tool"] == "scan_summary" and "total" in answer
    print("tool loop ok:", trace)

    # 5. data/model pipeline in an isolated workspace (needs the local genuine sample)
    positive = ROOT / "data/sample/GRC/positive"
    if positive.exists():
        import time
        t0 = time.time()
        run = tools.start_pipeline_run("data/sample/GRC/positive")
        steps = tools.run_full_pipeline(run["run_id"])
        print("pipeline:", [(x["stage"], x["status"]) for x in steps], f"{time.time() - t0:.0f}s")
        assert [x["status"] for x in steps] == ["ok"] * 4
        assert tools.pipeline_status(run["run_id"])["stages"]["train"]["status"] == "ok"
        assert tools.run_pipeline_stage(run["run_id"], "train")["status"] == "ok"
        # a stage cannot run before its predecessors, and non-genuine folders are refused
        run2 = tools.start_pipeline_run("data/sample/GRC/positive")
        assert tools.run_pipeline_stage(run2["run_id"], "train")["status"] == "blocked"
        try:
            tools.start_pipeline_run("data/GRC/fraud3_face_replacement")
            raise AssertionError("forged folder should be refused")
        except ValueError:
            pass
    else:
        print("SKIPPED pipeline test: data/sample/GRC/positive not found")

    # 6. UI renders without errors
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "agent/app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    print("UI ok, tabs:", len(at.tabs))
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
