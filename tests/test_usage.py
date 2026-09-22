from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from woke.events import Event
from woke.host import Host
from woke.model import ModelReply, OpenAICompatModel, ScriptModel
from woke.policy import AutoAllow
from woke.tui import context_tokens


class _Handler(BaseHTTPRequestHandler):
    mode = "stream"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.mode == "reject" and body.get("stream"):
            blob = b'{"error":{"message":"stream unsupported"}}'
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)
            return
        if body.get("stream"):
            chunks = [
                {"choices": [{"delta": {"content": "hi"}}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                    },
                },
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        payload = {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        }
        blob = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


class _Server:
    def __init__(self, mode: str) -> None:
        _Handler.mode = mode
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class UsageParseTests(unittest.TestCase):
    def test_stream_reports_usage(self) -> None:
        server = _Server("stream")
        self.addCleanup(server.close)
        model = OpenAICompatModel("key", "model", server.url)
        reply = model.complete([{"role": "user", "content": "hi"}], [])
        self.assertEqual(reply.text, "hi")
        self.assertEqual(reply.usage["total_tokens"], 7)

    def test_non_stream_fallback_reports_usage(self) -> None:
        server = _Server("reject")
        self.addCleanup(server.close)
        model = OpenAICompatModel("key", "model", server.url)
        reply = model.complete([{"role": "user", "content": "hi"}], [])
        self.assertEqual(reply.text, "hi")
        self.assertEqual(reply.usage["total_tokens"], 12)


class UsageRecordingTests(unittest.TestCase):
    def test_turn_records_usage_on_model_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(
                        text="hi",
                        usage={
                            "prompt_tokens": 4,
                            "completion_tokens": 1,
                            "total_tokens": 5,
                        },
                    )
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                host.run_turn(sid, "hello")
                message = next(e for e in host.events(sid) if e.kind == "model.message")
        self.assertEqual(message.payload["usage"]["total_tokens"], 5)

    def test_context_tokens_prefers_reported_usage(self) -> None:
        events = [
            Event(
                seq=1,
                ts="t",
                session_id="s",
                turn_id="t1",
                run_id="r1",
                kind="user.message",
                payload={"text": "x" * 4000},
            ),
            Event(
                seq=2,
                ts="t",
                session_id="s",
                turn_id="t1",
                run_id="r1",
                kind="model.message",
                payload={"text": "ok", "usage": {"total_tokens": 1234}},
            ),
        ]
        self.assertEqual(context_tokens(events), 1234)

    def test_context_tokens_falls_back_to_estimate(self) -> None:
        events = [
            Event(
                seq=1,
                ts="t",
                session_id="s",
                turn_id="t1",
                run_id="r1",
                kind="user.message",
                payload={"text": "x" * 400},
            )
        ]
        self.assertGreater(context_tokens(events), 0)


if __name__ == "__main__":
    unittest.main()
