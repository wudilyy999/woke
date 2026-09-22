from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.host import Host, HostClient
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoAllow
from woke.registry import TODO_NAME


class PlanModeTests(unittest.TestCase):
    def test_plan_turn_denies_dangerous_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="write_file",
                                arguments={"path": "out.txt", "content": "x"},
                            )
                        ]
                    ),
                    ModelReply(text="plan only"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                outcome = host.run_turn(sid, "plan the change", mode="plan")
                events = host.events(sid)
            started = [e for e in events if e.kind == "turn.started"][0]
            result = [e for e in events if e.kind == "tool.result"][0]
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(started.payload["mode"], "plan")
            self.assertFalse(result.payload["ok"])
            self.assertEqual(result.payload["error"], "permission denied")
            self.assertFalse((ws / "out.txt").exists())

    def test_plan_turn_marks_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel([ModelReply(text="ok")])
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                host.run_turn(sid, "plan it", mode="plan")
            system = model.calls[0][0]
            self.assertEqual(system["role"], "system")
            self.assertIn("plan mode", system["content"])

    def test_execute_turn_keeps_write_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="write_file",
                                arguments={"path": "out.txt", "content": "x"},
                            )
                        ]
                    ),
                    ModelReply(text="done"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                host.run_turn(sid, "write it")
            self.assertTrue((ws / "out.txt").is_file())

    def test_plan_mode_travels_over_host_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="write_file",
                                arguments={"path": "out.txt", "content": "x"},
                            )
                        ]
                    ),
                    ModelReply(text="planned"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                host.serve_background()
                client = HostClient(root)
                sid = client.create_session(str(ws))
                outcome = client.run_turn(sid, "plan it", mode="plan")
                started = [
                    e for e in client.events(sid) if e.kind == "turn.started"
                ][0]
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(started.payload["mode"], "plan")
            self.assertFalse((ws / "out.txt").exists())


class TodoTests(unittest.TestCase):
    def test_todo_write_records_list_and_feeds_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(
                        tool_calls=[
                            ToolCall(
                                id="t1",
                                name=TODO_NAME,
                                arguments={
                                    "items": [
                                        {"text": "inspect", "status": "completed"},
                                        {"text": "edit", "status": "in_progress"},
                                    ]
                                },
                            )
                        ]
                    ),
                    ModelReply(text="working"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                outcome = host.run_turn(sid, "go")
                events = host.events(sid)
            todo = next(e for e in events if e.kind == "todo.updated")
            result = [e for e in events if e.kind == "tool.result"][0]
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(
                [item["status"] for item in todo.payload["items"]],
                ["completed", "in_progress"],
            )
            self.assertEqual(result.payload["output"], "1/2 todos complete")
            second_prompt = model.calls[1][0]["content"]
            self.assertIn("# Task list", second_prompt)
            self.assertIn("[x] inspect", second_prompt)
            self.assertIn("[>] edit", second_prompt)

    def test_todo_write_rejects_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(
                        tool_calls=[
                            ToolCall(id="t1", name=TODO_NAME, arguments={"items": []})
                        ]
                    ),
                    ModelReply(text="done"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                host.run_turn(sid, "go")
                events = host.events(sid)
            result = [e for e in events if e.kind == "tool.result"][0]
            self.assertFalse(result.payload["ok"])
            self.assertIn("non-empty", result.payload["error"])
            self.assertFalse(any(e.kind == "todo.updated" for e in events))


if __name__ == "__main__":
    unittest.main()
