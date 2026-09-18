from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.host import Host
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoAllow


class TurnTests(unittest.TestCase):
    def test_hello_event_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel([ModelReply(text="hello")])
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws), title="demo")
                outcome = host.run_turn(sid, "hi")
            self.assertEqual(outcome.status, "completed")
            kinds = [e.kind for e in host_events(root, sid)]
            self.assertEqual(
                kinds,
                [
                    "session.created",
                    "turn.started",
                    "run.started",
                    "user.message",
                    "model.message",
                    "run.terminated",
                    "turn.terminated",
                ],
            )

    def test_write_file_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(
                        text="",
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="write_file",
                                arguments={"path": "out.txt", "content": "hello-woke"},
                            )
                        ],
                    ),
                    ModelReply(text="wrote it"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                outcome = host.run_turn(sid, "write out.txt")
                self.assertEqual(outcome.status, "completed")
            self.assertEqual((ws / "out.txt").read_text(), "hello-woke")
            kinds = [e.kind for e in host_events(root, sid)]
            self.assertIn("tool.call", kinds)
            self.assertIn("permission.requested", kinds)
            self.assertIn("permission.decided", kinds)
            self.assertIn("tool.result", kinds)

    def test_new_session_does_not_reuse_old_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel([ModelReply(text="one"), ModelReply(text="two")])
            with Host(root, model=model, policy=AutoAllow()) as host:
                first = host.create_session(str(ws), title="a")
                host.run_turn(first, "hello there")
                second = host.create_session(str(ws), title="b")
                self.assertNotEqual(first, second)
                self.assertEqual(len(host.events(second)), 1)
                roots = host.list_root_sessions()
                self.assertEqual(roots[0]["id"], second)
                self.assertEqual(roots[1]["id"], first)
                self.assertIn("hello there", roots[1]["preview"])
                reused = host.session_for_workspace(str(ws))
                self.assertEqual(reused, first)


def host_events(root: Path, session_id: str):
    from woke.store import Store

    store = Store(root)
    try:
        return store.read_session(session_id)
    finally:
        store.close()
