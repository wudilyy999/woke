from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.commands import expand_command, load_commands
from woke.host import Host
from woke.model import ModelReply, ScriptModel
from woke.policy import AutoAllow
from woke.tui import Tui, filter_slash


class CommandFileTests(unittest.TestCase):
    def test_load_commands_from_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            root = ws / ".woke" / "commands"
            root.mkdir(parents=True)
            (root / "deploy.md").write_text("Deploy $ARGUMENTS carefully.", encoding="utf-8")
            (root / "empty.md").write_text("   ", encoding="utf-8")
            (root / "bad name.md").write_text("nope", encoding="utf-8")
            commands = load_commands(ws)
        self.assertEqual(commands, {"deploy": "Deploy $ARGUMENTS carefully."})

    def test_missing_directory_yields_no_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_commands(Path(tmp)), {})

    def test_expand_command_substitutes_and_appends(self) -> None:
        self.assertEqual(expand_command("Do $ARGUMENTS now", "x"), "Do x now")
        self.assertEqual(expand_command("Do the thing", "x"), "Do the thing\n\nx")
        self.assertEqual(expand_command("Do the thing", "  "), "Do the thing")

    def test_slash_picker_lists_custom_commands(self) -> None:
        self.assertEqual(
            [name for name, _hint in filter_slash("/dep", {"deploy": "x"})],
            ["/deploy"],
        )
        self.assertIn("/quit", [name for name, _hint in filter_slash("/", {"deploy": "x"})])


class CommandDispatchTests(unittest.TestCase):
    def test_custom_command_sends_expanded_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            commands = ws / ".woke" / "commands"
            commands.mkdir(parents=True)
            (commands / "deploy.md").write_text("Ship $ARGUMENTS", encoding="utf-8")
            with Host(root, model=ScriptModel([ModelReply(text="ok")]), policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                tui = Tui(host, sid)
                sent: list[str] = []
                tui.send = sent.append  # type: ignore[method-assign]
                self.assertTrue(tui.command("/deploy prod"))
        self.assertEqual(sent, ["Ship prod"])


if __name__ == "__main__":
    unittest.main()
