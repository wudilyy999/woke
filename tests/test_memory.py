from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.host import Host
from woke.memory import load_briefing
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoAllow
from woke.runner import _with_system
from woke.tools import execute


class MemoryTests(unittest.TestCase):
    def test_briefing_loads_agents_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "AGENTS.md").write_text("use tabs", encoding="utf-8")
            (ws / ".woke").mkdir()
            (ws / ".woke" / "memory.md").write_text("ship date is Friday", encoding="utf-8")
            brief = load_briefing(ws)
            self.assertIn("use tabs", brief)
            self.assertIn("ship date is Friday", brief)

    def test_memory_tools_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            ok, _ = execute("memory_write", {"text": "prefer pytest"}, ws)
            self.assertTrue(ok)
            model = ScriptModel([ModelReply(text="ok")])
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                host.run_turn(sid, "hi")
                messages = _with_system(host.events(sid))
            self.assertIn("prefer pytest", messages[0]["content"])
            ok, text = execute("memory_read", {}, ws)
            self.assertTrue(ok)
            self.assertIn("prefer pytest", text)

    def test_memory_write_via_turn(self) -> None:
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
                                name="memory_write",
                                arguments={"text": "root cause was the lock"},
                            )
                        ],
                    ),
                    ModelReply(text="noted"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                outcome = host.run_turn(sid, "remember that")
                self.assertEqual(outcome.status, "completed")
            self.assertIn("root cause was the lock", (ws / ".woke" / "memory.md").read_text())
