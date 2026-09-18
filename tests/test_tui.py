from __future__ import annotations

import unittest
from pathlib import Path

from woke.events import Event
from woke.tui import (
    activity_line,
    composer_frame,
    display_width,
    filter_slash,
    pad_width,
    spinner_frame,
    status_line,
    transcript_lines,
    welcome_lines,
    workspace_candidates,
    wrap_text,
)


def _event(seq: int, kind: str, payload: dict, turn: str = "t") -> Event:
    return Event(
        seq=seq,
        ts="2026-01-01T00:00:00Z",
        session_id="s",
        turn_id=turn,
        run_id="r",
        kind=kind,
        payload=payload,
    )


class TuiRenderTests(unittest.TestCase):
    def test_failed_turn_shows_error(self) -> None:
        rows = transcript_lines([
            _event(1, "turn.terminated", {"status": "failed", "error": "model HTTP 404: model_not_found"})
        ], 80)
        self.assertIn(("err", "model HTTP 404: model_not_found"), rows)

    def test_welcome_looks_like_codex_card(self) -> None:
        lines = [text for _style, text in welcome_lines("gpt-4o-mini", "/Users/me/proj", 50)]
        joined = "\n".join(lines)
        self.assertIn("╭", joined)
        self.assertIn("woke", joined)
        self.assertIn("model:", joined)
        self.assertIn("directory:", joined)
        frame = composer_frame("hello", 40, 4)
        self.assertTrue(any(line.startswith("│") for line in frame))
        self.assertIn("hello", "\n".join(frame))
        for line in frame:
            self.assertEqual(display_width(line), 40)

    def test_transcript_roles(self) -> None:
        events = [
            _event(1, "user.message", {"text": "write it"}),
            _event(2, "model.message", {"text": "sure"}),
            _event(3, "tool.call", {"id": "c1", "name": "write_file", "arguments": {"path": "a.txt"}}),
            _event(4, "tool.result", {"id": "c1", "name": "write_file", "ok": True, "output": "wrote a.txt"}),
        ]
        rows = transcript_lines(events, 60)
        styles = [style for style, _ in rows]
        texts = [text for _style, text in rows]
        self.assertIn("user", styles)
        self.assertIn("brand", styles)
        self.assertTrue(any("you" in t for t in texts))
        self.assertTrue(any("woke" in t for t in texts))
        self.assertTrue(any("write_file" in t for t in texts))

    def test_status_and_wrap(self) -> None:
        self.assertGreater(len(wrap_text("abcd efgh", 5)), 1)
        line = status_line([], 32000, "gpt-4o-mini", False, True)
        self.assertIn("context left", line)
        self.assertIn("auto", line)
        self.assertIn("commands", line)

    def test_cjk_width_and_slash_filter(self) -> None:
        self.assertEqual(display_width("中"), 2)
        self.assertEqual(display_width(pad_width("你好", 8)), 8)
        self.assertGreaterEqual(len(wrap_text("你好世界你好世界", 8)), 2)
        names = [name for name, _hint in filter_slash("/mo")]
        self.assertEqual(names, ["/model"])
        self.assertIn("/quit", [name for name, _hint in filter_slash("/")])
        self.assertIn("/workspace", [name for name, _hint in filter_slash("/")])
        self.assertIn("/workspace", [name for name, _hint in filter_slash("/work")])
        self.assertIn("/permission", [name for name, _hint in filter_slash("/")])
        self.assertIn("/permission", [name for name, _hint in filter_slash("/perm")])
        self.assertIn("/resume", [name for name, _hint in filter_slash("/")])
        self.assertIn("/new", [name for name, _hint in filter_slash("/")])
        self.assertIn("/rewind", [name for name, _hint in filter_slash("/")])
        self.assertIn("/fork", [name for name, _hint in filter_slash("/")])

    def test_activity_spinner_and_workspace_list(self) -> None:
        self.assertNotEqual(spinner_frame(0), spinner_frame(1))
        thinking = activity_line([], True, 0, "waiting for model")
        self.assertIsNotNone(thinking)
        self.assertIn("waiting for model", thinking or "")
        compacting = activity_line([], True, 1, "compacting context")
        self.assertIn("compacting context", compacting or "")
        running = activity_line(
            [
                _event(1, "user.message", {"text": "go"}),
                _event(2, "tool.call", {"id": "c1", "name": "write_file", "arguments": {"path": "a.txt"}}),
            ],
            True,
            3,
        )
        self.assertIn("running write_file", running or "")
        self.assertIsNone(activity_line([], False, 0))
        rows = workspace_candidates(str(Path.home()), "", [])
        self.assertTrue(any(Path(path) == Path.home() for path, _hint in rows))
