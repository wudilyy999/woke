from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from woke.host import Host, HostClient
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoAllow


class GateModel:
    """Blocks inside complete() until the test lets it go."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.finished = threading.Event()

    def complete(self, messages, tools, on_delta=None) -> ModelReply:
        self.entered.set()
        self.gate.wait(timeout=30)
        self.finished.set()
        return ModelReply(text="streamed answer")

    def summarize(self, text: str) -> str:
        return text


class StreamTests(unittest.TestCase):
    def test_background_turn_streams_events_as_they_land(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = GateModel()
            with Host(root, model=model, policy=AutoAllow()) as host:
                host.serve_background()
                client = HostClient(root)
                sid = client.create_session(str(ws), title="stream")
                client.start_turn(sid, "say something")
                self.assertTrue(model.entered.wait(timeout=5))

                kinds: list[str] = []
                for event in client.stream_events(sid):
                    kinds.append(event.kind)
                    if event.kind == "turn.started":
                        # Delivered while the model is still gated: the stream is live.
                        self.assertTrue(model.entered.is_set())
                        self.assertFalse(model.finished.is_set())
                        model.gate.set()
                    if event.kind == "turn.terminated":
                        break

            self.assertEqual(kinds[0], "session.created")
            self.assertLess(kinds.index("turn.started"), kinds.index("model.message"))
            self.assertEqual(kinds[-1], "turn.terminated")
            self.assertTrue(model.finished.is_set())

    def test_stream_replays_history_and_stops_at_turn_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            with Host(root, model=ScriptModel([ModelReply(text="done")]), policy=AutoAllow()) as host:
                host.serve_background()
                client = HostClient(root)
                sid = client.create_session(str(ws), title="stream")
                client.run_turn(sid, "hello")
                kinds = [event.kind for event in client.stream_events(sid)]
            self.assertEqual(kinds[0], "session.created")
            self.assertEqual(kinds[-1], "turn.terminated")
            self.assertIn("user.message", kinds)

    def test_background_failure_lands_in_the_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            (ws / ".woke").mkdir()
            (ws / ".woke" / "hooks.json").write_text(json.dumps({"hooks": "nope"}), encoding="utf-8")
            model = ScriptModel(
                [
                    ModelReply(
                        text="",
                        tool_calls=[
                            ToolCall(id="c1", name="read_file", arguments={"path": "notes.txt"})
                        ],
                    )
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                host.serve_background()
                client = HostClient(root)
                sid = client.create_session(str(ws), title="stream")
                client.start_turn(sid, "read it")
                terminal = None
                for event in client.stream_events(sid):
                    if event.kind == "turn.terminated":
                        terminal = event.payload
                        break
            self.assertEqual(terminal["status"], "failed")
            self.assertIn("hooks.json", terminal["error"])


if __name__ == "__main__":
    unittest.main()
