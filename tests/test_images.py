from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from woke.errors import PathEscapes, ValidationError
from woke.files import IMAGE_CAP, image_rel
from woke.host import Host, HostClient
from woke.model import ModelReply, ScriptModel
from woke.policy import AutoAllow
from woke.projection import estimate_tokens
from woke.tui import Tui

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _png(path: Path) -> None:
    path.write_bytes(PNG_1X1)


class ImageTests(unittest.TestCase):
    def test_turn_sends_image_parts_to_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            _png(ws / "shot.png")
            model = ScriptModel([ModelReply(text="i see a pixel")])
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws), title="vision")
                outcome = host.run_turn(sid, "what is this?", images=["shot.png"])
                payload = next(
                    event.payload for event in host.events(sid) if event.kind == "user.message"
                )
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(payload["images"], ["shot.png"])
            parts = model.calls[0][-1]["content"]
            self.assertEqual(parts[0], {"type": "text", "text": "what is this?"})
            self.assertEqual(parts[1]["type"], "image_url")
            url = parts[1]["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/png;base64,"))
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), PNG_1X1)

    def test_mention_attaches_image_instead_of_inlining_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            _png(ws / "shot.png")
            (ws / "notes.md").write_text("plain text")
            model = ScriptModel([ModelReply(text="seen")])
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws), title="mentions")
                host.run_turn(sid, "compare @shot.png with @notes.md")
                payload = next(
                    event.payload for event in host.events(sid) if event.kind == "user.message"
                )
            self.assertEqual(payload["images"], ["shot.png"])
            self.assertEqual([item["path"] for item in payload["attachments"]], ["notes.md"])
            parts = model.calls[0][-1]["content"]
            self.assertIn("plain text", parts[0]["text"])
            self.assertNotIn("PNG", parts[0]["text"])

    def test_attach_rejects_bad_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            (ws / "notes.md").write_text("text")
            outside = Path(tmp) / "outside.png"
            _png(outside)
            _png(ws / "big.png")
            with (ws / "big.png").open("ab") as handle:
                handle.truncate(IMAGE_CAP + 1)

            with self.assertRaises(ValidationError):
                image_rel(ws, "notes.md")
            with self.assertRaises(ValidationError):
                image_rel(ws, "missing.png")
            with self.assertRaises(ValidationError):
                image_rel(ws, "big.png")
            with self.assertRaises(PathEscapes):
                image_rel(ws, str(outside))

    def test_tui_image_command_queues_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            _png(ws / "shot.png")
            (ws / "notes.md").write_text("text")
            with Host(root, model=ScriptModel([ModelReply(text="ok")]), policy=AutoAllow()) as host:
                sid = host.create_session(str(ws), title="tui")
                tui = Tui(host, sid)
                self.assertTrue(tui.command("/image shot.png"))
                self.assertEqual(tui.pending_images, ["shot.png"])
                self.assertIn("queued", tui.notice or "")
                tui.command("/image shot.png")
                self.assertEqual(tui.pending_images, ["shot.png"])
                tui.command("/image notes.md")
                self.assertIn("unsupported image type", tui.error or "")
                tui.command("/new")
                self.assertEqual(tui.pending_images, [])

    def test_image_over_http(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            _png(ws / "shot.png")
            model = ScriptModel([ModelReply(text="remote ok")])
            with Host(root, model=model, policy=AutoAllow()) as host:
                host.serve_background()
                client = HostClient(root)
                sid = client.create_session(str(ws), title="remote")
                outcome = client.run_turn(sid, "look", images=["shot.png"])
            self.assertEqual(outcome.status, "completed")
            parts = model.calls[0][-1]["content"]
            self.assertEqual(parts[1]["type"], "image_url")

    def test_token_estimate_accounts_for_images(self) -> None:
        text_only = estimate_tokens([{"role": "user", "content": "look here"}])
        with_image = estimate_tokens(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look here"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                }
            ]
        )
        self.assertGreater(with_image - text_only, 500)


if __name__ == "__main__":
    unittest.main()
