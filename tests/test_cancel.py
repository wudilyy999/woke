from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from woke.errors import ToolCancelled
from woke.host import Host
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoAllow, WaitUser
from woke.sandbox import SandboxManager
from woke.tools import execute


class StreamingModel:
    """Emits one delta before returning, so a cancel can land mid-stream."""

    def complete(self, messages, tools, on_delta=None):
        if on_delta is not None:
            on_delta("partial")
        return ModelReply(text="partial")

    def summarize(self, text: str) -> str:
        return "summary"


def _sandbox() -> SandboxManager:
    sandbox = SandboxManager()
    if not sandbox.available:
        raise unittest.SkipTest("no platform sandbox available")
    return sandbox


class CancelTests(unittest.TestCase):
    def test_cancel_during_stream_terminates_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            with Host(root, model=StreamingModel(), policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                outcome = host.run_turn(sid, "hello", on_delta=lambda _text: host.cancel_turn(sid))
                events = host.events(sid)
            self.assertEqual(outcome.status, "cancelled")
            runs = [e.payload for e in events if e.kind == "run.terminated"]
            turns = [e.payload for e in events if e.kind == "turn.terminated"]
            self.assertEqual(runs[-1]["status"], "aborted")
            self.assertEqual(turns[-1]["status"], "cancelled")
            self.assertFalse(any(e.kind == "model.message" for e in events))

    def test_cancel_paused_turn_and_accepts_new_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(
                        tool_calls=[
                            ToolCall(id="c1", name="run_shell", arguments={"command": "echo hi"})
                        ]
                    ),
                    ModelReply(text="done"),
                ]
            )
            with Host(root, model=model, policy=WaitUser()) as host:
                sid = host.create_session(str(ws))
                self.assertEqual(host.run_turn(sid, "go").status, "paused")
                self.assertTrue(host.cancel_turn(sid))
                self.assertIsNone(host.get_session(sid)["open_turn"])
                self.assertFalse(host.cancel_turn(sid))
                self.assertEqual(host.run_turn(sid, "again", yes=True).status, "completed")

    def test_cancel_stops_running_shell_turn(self) -> None:
        _sandbox()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(
                        tool_calls=[
                            ToolCall(id="c1", name="run_shell", arguments={"command": "sleep 30"})
                        ]
                    ),
                    ModelReply(text="done"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                timer = threading.Timer(0.3, lambda: host.cancel_turn(sid))
                timer.start()
                started = time.monotonic()
                try:
                    outcome = host.run_turn(sid, "go")
                finally:
                    timer.cancel()
                elapsed = time.monotonic() - started
                events = host.events(sid)
            self.assertEqual(outcome.status, "cancelled")
            self.assertLess(elapsed, 10)
            results = [e.payload for e in events if e.kind == "tool.result"]
            self.assertEqual(results[-1]["error"], "cancelled")


class ShellStreamTests(unittest.TestCase):
    def test_shell_output_streams_per_line(self) -> None:
        sandbox = _sandbox()
        with tempfile.TemporaryDirectory() as tmp:
            seen: list[str] = []
            ok, output = execute(
                "run_shell",
                {"command": "printf 'alpha\\nbeta\\n'"},
                Path(tmp),
                sandbox,
                on_output=seen.append,
            )
        self.assertTrue(ok)
        self.assertEqual(seen, ["alpha\n", "beta\n"])
        self.assertIn("exit 0", output)
        self.assertIn("alpha", output)

    def test_shell_stops_when_cancelled(self) -> None:
        sandbox = _sandbox()
        with tempfile.TemporaryDirectory() as tmp:
            seen: list[str] = []
            state = {"cancel": False}

            def on_output(text: str) -> None:
                seen.append(text)
                state["cancel"] = True

            with self.assertRaises(ToolCancelled):
                execute(
                    "run_shell",
                    {"command": "echo first; sleep 30"},
                    Path(tmp),
                    sandbox,
                    on_output=on_output,
                    should_cancel=lambda: state["cancel"],
                )
        self.assertEqual(seen, ["first\n"])


if __name__ == "__main__":
    unittest.main()
