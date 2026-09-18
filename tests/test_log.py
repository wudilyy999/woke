from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.errors import HostLocked, UnknownKind
from woke.store import Store, acquire_lock, release_lock, workspace_state_root


class LogTests(unittest.TestCase):
    def test_unknown_kind_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            with self.assertRaises(UnknownKind):
                store.append("s1", "agent.chat", {"text": "nope"})
            store.close()

    def test_append_and_read_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root)
            created = store.append(
                "s1",
                "session.created",
                {"workspace": "/tmp/ws", "title": "demo"},
            )
            self.assertEqual(created.seq, 1)
            store.append(
                "s1",
                "turn.started",
                {},
                turn_id="t1",
            )
            store.close()
            again = Store(root)
            events = again.read_session("s1")
            self.assertEqual([e.kind for e in events], ["session.created", "turn.started"])
            self.assertEqual(events[0].payload["title"], "demo")
            again.close()

    def test_workspace_roots_are_isolated(self) -> None:
        a = workspace_state_root(Path("/tmp/woke-a"))
        b = workspace_state_root(Path("/tmp/woke-b"))
        self.assertNotEqual(a, b)
        self.assertEqual(workspace_state_root(Path("/tmp/woke-a")), a)

    def test_second_writer_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fd = acquire_lock(root)
            try:
                with self.assertRaises(HostLocked):
                    acquire_lock(root)
            finally:
                release_lock(fd)
