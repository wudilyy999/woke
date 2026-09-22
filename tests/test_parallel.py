from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Callable

from woke.host import Host
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoAllow
from woke.registry import SPAWN_NAME, ToolRegistry


class BarrierRegistry(ToolRegistry):
    """Blocks inside execute() until every call of the batch has arrived."""

    def __init__(self, parties: int) -> None:
        super().__init__()
        self.barrier = threading.Barrier(parties)
        self.seen: list[str] = []

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        workspace: Path,
        on_output: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[bool, str]:
        self.seen.append(name)
        self.barrier.wait(timeout=5)
        return True, f"{name} {arguments.get('path')}"


class RecordingRegistry(ToolRegistry):
    """Runs the real tools and keeps the dispatch order."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        workspace: Path,
        on_output: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[bool, str]:
        self.seen.append(name)
        return super().execute(
            name, arguments, workspace, on_output=on_output, should_cancel=should_cancel
        )


class TreeModel:
    """Parent asks for two subagents; each child answers from its own task text."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_delta: Callable[[str], None] | None = None,
    ) -> ModelReply:
        user = next(str(m.get("content")) for m in messages if m.get("role") == "user")
        if user.startswith("task-"):
            return ModelReply(text=f"child digest for {user}")
        if any(m.get("role") == "tool" for m in messages):
            return ModelReply(text="parent integrated both children")
        return ModelReply(
            text="",
            tool_calls=[
                ToolCall(id="s1", name=SPAWN_NAME, arguments={"task": "task-one", "label": "one"}),
                ToolCall(id="s2", name=SPAWN_NAME, arguments={"task": "task-two", "label": "two"}),
            ],
        )

    def summarize(self, text: str) -> str:
        return text


class ParallelToolTests(unittest.TestCase):
    def test_read_only_batch_runs_in_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(
                        text="",
                        tool_calls=[
                            ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"}),
                            ToolCall(id="c2", name="read_file", arguments={"path": "b.txt"}),
                        ],
                    ),
                    ModelReply(text="read both"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                registry = BarrierRegistry(2)
                host.registry = registry
                sid = host.create_session(str(ws), title="batch")
                outcome = host.run_turn(sid, "read both files")
            self.assertEqual(outcome.status, "completed")
            results = [e.payload for e in outcome.events if e.kind == "tool.result"]
            self.assertEqual([item["id"] for item in results], ["c1", "c2"])
            self.assertTrue(all(item["ok"] for item in results))
            self.assertEqual(registry.seen, ["read_file", "read_file"])

    def test_write_batch_keeps_order_and_diffs(self) -> None:
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
                                arguments={"path": "one.txt", "content": "one"},
                            ),
                            ToolCall(
                                id="c2",
                                name="write_file",
                                arguments={"path": "two.txt", "content": "two"},
                            ),
                        ],
                    ),
                    ModelReply(text="wrote both"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                registry = RecordingRegistry()
                host.registry = registry
                sid = host.create_session(str(ws), title="batch")
                outcome = host.run_turn(sid, "write both files")
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(registry.seen, ["write_file", "write_file"])
            self.assertEqual((ws / "one.txt").read_text(), "one")
            self.assertEqual((ws / "two.txt").read_text(), "two")
            diffs = [e.payload["diff"] for e in outcome.events if e.kind == "tool.result"]
            self.assertTrue(diffs[0].startswith("--- a/one.txt"))
            self.assertTrue(diffs[1].startswith("--- a/two.txt"))

    def test_subagents_run_in_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            with Host(root, model=TreeModel(), policy=AutoAllow()) as host:
                sid = host.create_session(str(ws), title="parent")
                outcome = host.run_turn(sid, "split the work")
                children = [
                    item for item in host.list_sessions() if item.get("parent_session_id") == sid
                ]
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(len(children), 2)
            self.assertEqual({item["title"] for item in children}, {"one", "two"})
            outputs = [e.payload["output"] for e in outcome.events if e.kind == "tool.result"]
            self.assertEqual(len(outputs), 2)
            self.assertTrue(any("child digest for task-one" in text for text in outputs))
            self.assertTrue(any("child digest for task-two" in text for text in outputs))


if __name__ == "__main__":
    unittest.main()
