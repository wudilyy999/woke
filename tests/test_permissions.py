from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.host import Host
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoDeny, GradedPolicy, WaitUser


class PermissionTests(unittest.TestCase):
    def test_deny_does_not_execute_shell(self) -> None:
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
                                name="run_shell",
                                arguments={"command": "echo pwned > owned.txt"},
                            )
                        ],
                    ),
                    ModelReply(text="stopped"),
                ]
            )
            with Host(root, model=model, policy=AutoDeny()) as host:
                sid = host.create_session(str(ws))
                outcome = host.run_turn(sid, "run a command")
                self.assertEqual(outcome.status, "completed")
                events = host.events(sid)
            self.assertFalse((ws / "owned.txt").exists())
            decided = [e for e in events if e.kind == "permission.decided"]
            self.assertEqual(decided[0].payload["decision"], "deny")
            result = [e for e in events if e.kind == "tool.result"][0]
            self.assertFalse(result.payload["ok"])
            self.assertEqual(result.payload["error"], "permission denied")

    def test_wait_then_approve(self) -> None:
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
                                arguments={"path": "ok.txt", "content": "yes"},
                            )
                        ],
                    ),
                    ModelReply(text="done"),
                ]
            )
            with Host(root, model=model, policy=WaitUser()) as host:
                sid = host.create_session(str(ws))
                paused = host.run_turn(sid, "write")
                self.assertEqual(paused.status, "paused")
                self.assertEqual(paused.pending_call_id, "c1")
                self.assertFalse((ws / "ok.txt").exists())
                continued = host.decide_permission(sid, "c1", "allow")
                self.assertEqual(continued.status, "completed")
            self.assertEqual((ws / "ok.txt").read_text(), "yes")

    def test_session_grant_skips_later_wait(self) -> None:
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
                                arguments={"path": "a.txt", "content": "one"},
                            )
                        ],
                    ),
                    ModelReply(text="first"),
                    ModelReply(
                        text="",
                        tool_calls=[
                            ToolCall(
                                id="c2",
                                name="write_file",
                                arguments={"path": "b.txt", "content": "two"},
                            )
                        ],
                    ),
                    ModelReply(text="second"),
                ]
            )
            with Host(root, model=model, policy=WaitUser()) as host:
                sid = host.create_session(str(ws))
                paused = host.run_turn(sid, "write a")
                self.assertEqual(paused.status, "paused")
                host.decide_permission(sid, "c1", "allow", scope="session")
                second = host.run_turn(sid, "write b")
                self.assertEqual(second.status, "completed")
            self.assertEqual((ws / "a.txt").read_text(), "one")
            self.assertEqual((ws / "b.txt").read_text(), "two")
            grants = [e for e in host_events(root, sid) if e.kind == "permission.decided"]
            self.assertTrue(any(e.payload.get("scope") == "session" for e in grants))

    def test_edits_allows_write_but_asks_shell(self) -> None:
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
                                arguments={"path": "ok.txt", "content": "x"},
                            )
                        ],
                    ),
                    ModelReply(text="wrote"),
                    ModelReply(
                        text="",
                        tool_calls=[
                            ToolCall(id="c2", name="run_shell", arguments={"command": "echo hi"})
                        ],
                    ),
                    ModelReply(text="done"),
                ]
            )
            with Host(root, model=model, policy=GradedPolicy("edits")) as host:
                sid = host.create_session(str(ws))
                first = host.run_turn(sid, "write")
                self.assertEqual(first.status, "completed")
                self.assertEqual((ws / "ok.txt").read_text(), "x")
                paused = host.run_turn(sid, "shell")
                self.assertEqual(paused.status, "paused")
                self.assertEqual(paused.pending_call_id, "c2")
            self.assertFalse((ws / "owned.txt").exists())

    def test_readonly_denies_write(self) -> None:
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
                                arguments={"path": "nope.txt", "content": "x"},
                            )
                        ],
                    ),
                    ModelReply(text="stopped"),
                ]
            )
            with Host(root, model=model, policy=GradedPolicy("readonly")) as host:
                sid = host.create_session(str(ws))
                outcome = host.run_turn(sid, "write")
                self.assertEqual(outcome.status, "completed")
            self.assertFalse((ws / "nope.txt").exists())


def host_events(root: Path, session_id: str):
    from woke.store import Store

    store = Store(root)
    try:
        return store.read_session(session_id)
    finally:
        store.close()
