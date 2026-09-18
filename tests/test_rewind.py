from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.host import Host
from woke.model import ModelReply, ScriptModel
from woke.policy import AutoAllow


class RewindTests(unittest.TestCase):
    def test_rewind_forks_prefix_and_keeps_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [ModelReply(text="first"), ModelReply(text="second"), ModelReply(text="third")]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws), title="demo")
                host.run_turn(sid, "one")
                host.run_turn(sid, "two")
                host.run_turn(sid, "three")
                users = [e for e in host.events(sid) if e.kind == "user.message"]
                self.assertEqual([e.payload["text"] for e in users], ["one", "two", "three"])
                two = users[1]
                new_id, prefill = host.fork_before_user_seq(sid, two.seq)
                self.assertEqual(prefill, "two")
                self.assertNotEqual(new_id, sid)
                new_users = [e.payload["text"] for e in host.events(new_id) if e.kind == "user.message"]
                self.assertEqual(new_users, ["one"])
                old_users = [e.payload["text"] for e in host.events(sid) if e.kind == "user.message"]
                self.assertEqual(old_users, ["one", "two", "three"])
                targets = host.rewind_targets(sid)
                self.assertEqual([t["text"] for t in targets], ["three", "two", "one"])

    def test_fork_copies_entire_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel([ModelReply(text="ok")])
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                host.run_turn(sid, "hello")
                forked = host.fork_session(sid)
                self.assertNotEqual(forked, sid)
                kinds = [e.kind for e in host.events(forked) if e.kind != "session.created"]
                orig = [e.kind for e in host.events(sid) if e.kind != "session.created"]
                self.assertEqual(kinds, orig)
