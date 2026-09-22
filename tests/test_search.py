from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.host import Host, HostClient
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoAllow
from woke.registry import SPAWN_NAME
from woke.tui import Tui


class SessionSearchTests(unittest.TestCase):
    def test_search_finds_transcript_text_across_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(text="the cache lives in redis"),
                    ModelReply(text="nothing about caching here"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                first = host.create_session(str(ws), title="one")
                second = host.create_session(str(ws), title="two")
                host.run_turn(first, "where does the cache live?")
                host.run_turn(second, "unrelated question")

                hits = host.search_sessions("cache")
                self.assertEqual([hit["id"] for hit in hits], [first])
                self.assertEqual(hits[0]["title"], "one")
                self.assertEqual(hits[0]["matches"], 2)
                self.assertIn("cache", hits[0]["snippet"].lower())

                self.assertEqual(host.search_sessions("   "), [])
                self.assertNotIn(second, [hit["id"] for hit in host.search_sessions("cache")])

    def test_search_ignores_child_sessions_and_like_wildcards(self) -> None:
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
                                id="s1",
                                name=SPAWN_NAME,
                                arguments={"task": "study the zygote pipeline", "label": "child"},
                            )
                        ],
                    ),
                    ModelReply(text="child saw a zygote"),
                    ModelReply(text="parent done"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                parent = host.create_session(str(ws), title="parent")
                host.run_turn(parent, "100% done, delegate the rest")

                self.assertEqual(host.search_sessions("zygote"), [])
                self.assertEqual(
                    [hit["id"] for hit in host.search_sessions("100%")], [parent]
                )
                self.assertEqual(host.search_sessions("100_"), [])

    def test_search_over_http(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel([ModelReply(text="the quota is 42 units")])
            with Host(root, model=model, policy=AutoAllow()) as host:
                host.serve_background()
                client = HostClient(root)
                sid = client.create_session(str(ws), title="quota")
                client.run_turn(sid, "how big is the quota?")
                hits = client.search_sessions("quota")
            self.assertEqual([hit["id"] for hit in hits], [sid])
            self.assertEqual(hits[0]["matches"], 2)

    def test_tui_resume_uses_transcript_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel([ModelReply(text="only this session talks about aardvarks")])
            with Host(root, model=model, policy=AutoAllow()) as host:
                older = host.create_session(str(ws), title="older")
                host.run_turn(older, "tell me about the aardvarks")
                current = host.create_session(str(ws), title="current")
                tui = Tui(host, current)
                tui.input = "/resume"
                rows = tui.session_items()
                self.assertEqual([row[0] for row in rows], [current, older])

                tui.input = "/resume only this session"
                rows = tui.session_items()
                self.assertEqual([row[0] for row in rows], [older])
                self.assertIn("hits", rows[0][1])

                self.assertTrue(tui.command("/resume only this session"))
                self.assertEqual(tui.session_id, older)
                self.assertEqual(tui.error, None)


if __name__ == "__main__":
    unittest.main()
