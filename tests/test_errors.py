from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.host import Host
from woke.model import ModelReply, ScriptModel
from woke.policy import AutoAllow


class ErrorTests(unittest.TestCase):
    def test_model_failure_fails_turn_not_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()

            def boom(_messages, _tools=None):
                raise RuntimeError("provider down")

            model = ScriptModel([boom])
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws))
                outcome = host.run_turn(sid, "hello")
                self.assertEqual(outcome.status, "failed")
                self.assertIn("provider down", outcome.error or "")
                kinds = [e.kind for e in host.events(sid)]
                self.assertIn("turn.terminated", kinds)
                term = [e for e in host.events(sid) if e.kind == "turn.terminated"][0]
                self.assertEqual(term.payload["status"], "failed")
                # Host still serves other sessions.
                model.replies.append(ModelReply(text="recovered"))
                sid2 = host.create_session(str(ws), title="other")
                ok = host.run_turn(sid2, "again")
                self.assertEqual(ok.status, "completed")
