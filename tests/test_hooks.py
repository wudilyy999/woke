from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from woke.errors import ValidationError
from woke.hooks import load_hooks, matching
from woke.host import Host
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoAllow


def _write_hooks(ws: Path, hooks: list[dict]) -> None:
    (ws / ".woke").mkdir(exist_ok=True)
    (ws / ".woke" / "hooks.json").write_text(
        json.dumps({"hooks": hooks}), encoding="utf-8"
    )


class HookTests(unittest.TestCase):
    def test_pre_tool_hook_blocks_the_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            _write_hooks(
                ws,
                [
                    {
                        "event": "PreToolUse",
                        "match": "run_shell",
                        "command": "printf 'no shell here' >&2; exit 2",
                    }
                ],
            )
            model = ScriptModel(
                [
                    ModelReply(
                        text="",
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="run_shell",
                                arguments={"command": "touch should-not-exist"},
                            )
                        ],
                    ),
                    ModelReply(text="ok, shell stayed off"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws), title="hooks")
                outcome = host.run_turn(sid, "run it")
                hook_events = [e.payload for e in outcome.events if e.kind == "hook.result"]
                result = [e.payload for e in outcome.events if e.kind == "tool.result"][0]
            self.assertEqual(outcome.status, "completed")
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "blocked by hook: no shell here")
            self.assertFalse((ws / "should-not-exist").exists())
            self.assertEqual(hook_events[0]["event"], "PreToolUse")
            self.assertFalse(hook_events[0]["ok"])

    def test_post_tool_hook_output_reaches_the_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            _write_hooks(
                ws,
                [
                    {
                        "event": "post_tool_use",
                        "match": "write_file",
                        "command": "printf 'format-checked'",
                    }
                ],
            )
            model = ScriptModel(
                [
                    ModelReply(
                        text="",
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="write_file",
                                arguments={"path": "out.txt", "content": "hello"},
                            )
                        ],
                    ),
                    ModelReply(text="wrote it"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws), title="hooks")
                outcome = host.run_turn(sid, "write out.txt")
                result = [e.payload for e in outcome.events if e.kind == "tool.result"][0]
            self.assertTrue(result["ok"])
            self.assertIn("format-checked", result["output"])
            self.assertEqual(result["diff"].count("+hello"), 1)

    def test_turn_end_hook_sees_the_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            _write_hooks(ws, [{"event": "TurnEnd", "command": "cat >> hooks.log"}])
            with Host(root, model=ScriptModel([ModelReply(text="done")]), policy=AutoAllow()) as host:
                sid = host.create_session(str(ws), title="hooks")
                outcome = host.run_turn(sid, "just answer")
                events = [e.payload for e in outcome.events if e.kind == "hook.result"]
            payload = json.loads((ws / "hooks.log").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["session_id"], sid)
            self.assertEqual(events[0]["event"], "TurnEnd")
            self.assertTrue(events[0]["ok"])

    def test_hook_config_errors_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            _write_hooks(ws, [{"event": "BeforeEverything", "command": "true"}])
            with self.assertRaises(ValidationError):
                load_hooks(ws)
            _write_hooks(ws, [{"event": "TurnEnd"}])
            with self.assertRaises(ValidationError):
                load_hooks(ws)
            (ws / ".woke" / "hooks.json").write_text("{", encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_hooks(ws)

    def test_match_pattern_limits_hooks_to_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            _write_hooks(
                ws,
                [
                    {"event": "PreToolUse", "match": "mcp__*", "command": "true"},
                    {"event": "PreToolUse", "match": "*", "command": "true"},
                    {"event": "TurnEnd", "match": "run_shell", "command": "true"},
                ],
            )
            hooks = load_hooks(ws)
            self.assertEqual(len(matching(hooks, "pretooluse", "mcp__fake__echo")), 2)
            self.assertEqual(len(matching(hooks, "pretooluse", "read_file")), 1)
            self.assertEqual(matching(hooks, "turnend"), [])


if __name__ == "__main__":
    unittest.main()
