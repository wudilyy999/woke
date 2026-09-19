from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.host import Host, HostClient
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoAllow
from woke.registry import SPAWN_NAME


class SubagentTests(unittest.TestCase):
    def test_remote_child_session_uses_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            with Host(root, model=ScriptModel([ModelReply(text="hello")]), policy=AutoAllow()) as host:
                host.serve_background()
                client = HostClient(root)
                parent = client.create_session(str(ws), title="parent")
                child = client.create_session(str(ws), title="child", parent_session_id=parent)
                self.assertEqual(client.get_session(child)["parent_session_id"], parent)

    def test_spawn_creates_child_session(self) -> None:
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
                                name=SPAWN_NAME,
                                arguments={"task": "say hi as a child", "label": "worker"},
                            )
                        ],
                    ),
                    ModelReply(text="child finished the slice"),
                    ModelReply(text="parent integrated the result"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws), title="parent")
                outcome = host.run_turn(sid, "split the work")
                self.assertEqual(outcome.status, "completed")
                result = [e for e in host.events(sid) if e.kind == "tool.result"][0]
                self.assertTrue(result.payload["ok"])
                self.assertIn("child_session=", result.payload["output"])
                self.assertIn("child finished the slice", result.payload["output"])
                children = [
                    s
                    for s in host.list_sessions()
                    if s.get("parent_session_id") == sid
                ]
                self.assertEqual(len(children), 1)
                self.assertEqual(children[0]["title"], "worker")
                child_events = host.events(children[0]["id"])
                self.assertIn("user.message", [e.kind for e in child_events])
                self.assertIn("turn.terminated", [e.kind for e in child_events])

    def test_child_cannot_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(
                        text="",
                        tool_calls=[
                            ToolCall(id="c1", name=SPAWN_NAME, arguments={"task": "nested"})
                        ],
                    ),
                    ModelReply(
                        text="",
                        tool_calls=[
                            ToolCall(id="c2", name=SPAWN_NAME, arguments={"task": "too deep"})
                        ],
                    ),
                    ModelReply(text="child stopped"),
                    ModelReply(text="parent done"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                host.run_turn(sid, "go")
                children = [s for s in host.list_sessions() if s.get("parent_session_id") == sid]
                self.assertEqual(len(children), 1)
                child_results = [e for e in host.events(children[0]["id"]) if e.kind == "tool.result"]
                self.assertTrue(child_results)
                self.assertFalse(child_results[0].payload["ok"])
                self.assertIn("max spawn depth", child_results[0].payload.get("error") or child_results[0].payload.get("output") or "")
