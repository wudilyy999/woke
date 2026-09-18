from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.host import Host
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoAllow
from woke.projection import estimate_tokens, project_messages
from woke.store import Store


class CompactionTests(unittest.TestCase):
    def test_budget_compacts_prior_turn_and_keeps_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(text="n" * 4000),
                    ModelReply(text="second turn"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow(), token_budget=80) as host:
                sid = host.create_session(str(ws))
                first = host.run_turn(sid, "ramble")
                self.assertEqual(first.status, "completed")
                second = host.run_turn(sid, "continue")
                self.assertEqual(second.status, "completed")
                events = host.events(sid)

            kinds = [e.kind for e in events]
            self.assertIn("compaction.applied", kinds)
            compact = next(e for e in events if e.kind == "compaction.applied")
            originals = [e for e in events if e.seq <= compact.payload["to_seq"]]
            self.assertGreater(len(originals), 0)
            self.assertTrue(any(e.kind == "model.message" and "n" * 100 in (e.payload.get("text") or "") for e in events))

            messages = project_messages(events)
            self.assertLess(estimate_tokens(messages), 80 + 50)
            self.assertTrue(any("summary" in (m.get("content") or "").lower() for m in messages))

    def test_force_compact_writes_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel([ModelReply(text="hello"), ModelReply(text="again")])
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                host.run_turn(sid, "one")
                host.run_turn(sid, "two")
                event = host.compact(sid, force=True)
                self.assertIsNotNone(event)
                self.assertEqual(event.kind, "compaction.applied")
            store = Store(root)
            try:
                events = store.read_session(sid)
            finally:
                store.close()
            self.assertTrue(any(e.kind == "user.message" for e in events))

    def test_second_compact_folds_previous_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(text="a" * 4000),
                    ModelReply(text="b" * 4000),
                    ModelReply(text="third"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow(), token_budget=80) as host:
                sid = host.create_session(str(ws))
                host.run_turn(sid, "one")
                host.run_turn(sid, "two")
                host.run_turn(sid, "three")
                events = host.events(sid)
            compacts = [e for e in events if e.kind == "compaction.applied"]
            self.assertGreaterEqual(len(compacts), 2)
            self.assertTrue(any(e.payload.get("folded_previous") for e in compacts[1:]))
            self.assertTrue(any("Previous summary:" in text for text in model.summaries))

    def test_tool_output_pruned_in_prompt_not_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            huge = "z" * 20_000
            (ws / "big.txt").write_text(huge, encoding="utf-8")
            model = ScriptModel(
                [
                    ModelReply(
                        text="",
                        tool_calls=[
                            ToolCall(id="c1", name="read_file", arguments={"path": "big.txt"})
                        ],
                    ),
                    ModelReply(text="saw it"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                host.run_turn(sid, "read big")
                events = host.events(sid)
            result = next(e for e in events if e.kind == "tool.result")
            self.assertGreater(len(result.payload.get("output") or ""), 15_000)
            prompt = project_messages(events)
            tool_msg = next(m for m in prompt if m.get("role") == "tool")
            self.assertLess(len(tool_msg["content"]), 9_000)
            self.assertIn("pruned from prompt", tool_msg["content"])
