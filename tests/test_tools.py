from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.errors import PathEscapes
from woke.tools import contained_path, execute


class ToolTests(unittest.TestCase):
    def test_path_escape_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            with self.assertRaises(PathEscapes):
                contained_path(ws, "../outside.txt")
            ok, output = execute("read_file", {"path": "../outside.txt"}, ws)
            self.assertFalse(ok)
            self.assertIn("escapes", output)

    def test_write_read_grep_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            ok, _ = execute("write_file", {"path": "a/b.txt", "content": "hello woke"}, ws)
            self.assertTrue(ok)
            ok, text = execute("read_file", {"path": "a/b.txt"}, ws)
            self.assertTrue(ok)
            self.assertIn("hello woke", text)
            ok, hits = execute("grep", {"pattern": "woke", "path": "."}, ws)
            self.assertTrue(ok)
            self.assertIn("a/b.txt:1", hits)
            ok, _ = execute("str_replace", {"path": "a/b.txt", "old": "woke", "new": "runtime"}, ws)
            self.assertTrue(ok)
            self.assertEqual((ws / "a/b.txt").read_text(), "hello runtime")

    def test_missing_args_and_empty_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            ok, output = execute("read_file", {}, ws)
            self.assertFalse(ok)
            self.assertIn("missing argument", output)
            ok, output = execute("run_shell", {"command": "   "}, ws)
            self.assertFalse(ok)
            self.assertIn("empty", output)

    def test_binary_file_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "blob.bin").write_bytes(b"hello\x00world")
            ok, output = execute("read_file", {"path": "blob.bin"}, ws)
            self.assertFalse(ok)
            self.assertIn("binary", output)
