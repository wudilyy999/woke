from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.errors import ValidationError
from woke.host import Host
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoDeny, WaitUser, load_rules
from woke.sandbox import SandboxManager


def _write_rules(ws: Path, body: str) -> None:
    root = ws / ".woke"
    root.mkdir(exist_ok=True)
    (root / "permissions.json").write_text(body, encoding="utf-8")


class RuleFileTests(unittest.TestCase):
    def test_missing_file_yields_no_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rules = load_rules(Path(tmp))
        self.assertFalse(rules.allow("run_shell", {"command": "git status"}))

    def test_rules_match_primary_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            _write_rules(
                ws,
                '{"allow": [{"tool": "run_shell", "match": "git *"},'
                ' {"tool": "write_file", "match": "docs/*"}]}',
            )
            rules = load_rules(ws)
        self.assertTrue(rules.allow("run_shell", {"command": "git status"}))
        self.assertFalse(rules.allow("run_shell", {"command": "rm -rf build"}))
        self.assertTrue(rules.allow("write_file", {"path": "docs/a.md"}))
        self.assertFalse(rules.allow("write_file", {"path": "src/a.py"}))

    def test_wildcard_tool_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            _write_rules(ws, '{"allow": [{"tool": "*", "match": "docs/*"}]}')
            rules = load_rules(ws)
        self.assertTrue(rules.allow("write_file", {"path": "docs/a.md"}))
        self.assertTrue(rules.allow("str_replace", {"path": "docs/a.md"}))
        self.assertFalse(rules.allow("write_file", {"path": "src/a.py"}))

    def test_bad_rule_file_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            _write_rules(ws, "[]")
            with self.assertRaises(ValidationError):
                load_rules(ws)
            _write_rules(ws, "{")
            with self.assertRaises(ValidationError):
                load_rules(ws)
            _write_rules(ws, '{"allow": [{"tool": "run_shell"}]}')
            with self.assertRaises(ValidationError):
                load_rules(ws)


class RuleTurnTests(unittest.TestCase):
    def setUp(self) -> None:
        if not SandboxManager().available:
            self.skipTest("no platform sandbox available")

    def _host_run(self, command: str, rules: str, policy):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "root"
        ws = Path(tmp.name) / "ws"
        ws.mkdir()
        _write_rules(ws, rules)
        model = ScriptModel(
            [
                ModelReply(
                    tool_calls=[
                        ToolCall(id="c1", name="run_shell", arguments={"command": command})
                    ]
                ),
                ModelReply(text="done"),
            ]
        )
        with Host(root, model=model, policy=policy) as host:
            sid = host.create_session(str(ws))
            outcome = host.run_turn(sid, "go")
            events = host.events(sid)
        return outcome, events

    def test_matching_command_is_allowed_by_policy(self) -> None:
        outcome, events = self._host_run(
            "git status",
            '{"allow": [{"tool": "run_shell", "match": "git *"}]}',
            WaitUser(),
        )
        result = [e for e in events if e.kind == "tool.result"][0]
        decided = [e for e in events if e.kind == "permission.decided"]
        self.assertEqual(outcome.status, "completed")
        self.assertTrue(result.payload["ok"])
        self.assertEqual([e.payload["decision"] for e in decided], ["allow"])
        self.assertEqual(decided[0].payload["source"], "policy")

    def test_unmatched_command_still_pauses(self) -> None:
        outcome, events = self._host_run(
            "rm -rf build",
            '{"allow": [{"tool": "run_shell", "match": "git *"}]}',
            WaitUser(),
        )
        self.assertEqual(outcome.status, "paused")
        self.assertTrue(any(e.kind == "permission.requested" for e in events))

    def test_readonly_policy_denies_regardless_of_rules(self) -> None:
        outcome, events = self._host_run(
            "git status",
            '{"allow": [{"tool": "run_shell", "match": "git *"}]}',
            AutoDeny(),
        )
        result = [e for e in events if e.kind == "tool.result"][0]
        self.assertEqual(outcome.status, "completed")
        self.assertFalse(result.payload["ok"])
        self.assertEqual(result.payload["error"], "permission denied")


if __name__ == "__main__":
    unittest.main()
