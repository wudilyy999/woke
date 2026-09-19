from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.host import Host, HostClient
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoAllow


class GapTests(unittest.TestCase):
    def test_streaming_callback_preserves_delta_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            model = ScriptModel([ModelReply(text="streamed")])
            deltas: list[str] = []
            with Host(root, model=model, policy=AutoAllow()) as host:
                session = host.create_session(str(workspace))
                outcome = host.run_turn(session, "hello", on_delta=deltas.append)
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(deltas, ["streamed"])

    def test_file_mention_is_attached_to_model_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            (workspace / "notes.txt").write_text("remember this", encoding="utf-8")
            model = ScriptModel([ModelReply(text="seen")])
            with Host(root, model=model, policy=AutoAllow()) as host:
                session = host.create_session(str(workspace))
                host.run_turn(session, "inspect @notes.txt")
            user = next(message for message in model.calls[0] if message["role"] == "user")
            self.assertIn("@notes.txt", user["content"])
            self.assertIn("remember this", user["content"])

    def test_write_result_contains_unified_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            (workspace / "out.txt").write_text("old\n", encoding="utf-8")
            model = ScriptModel(
                [
                    ModelReply(
                        tool_calls=[
                            ToolCall(
                                id="write",
                                name="write_file",
                                arguments={"path": "out.txt", "content": "new\n"},
                            )
                        ]
                    ),
                    ModelReply(text="done"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                session = host.create_session(str(workspace))
                host.run_turn(session, "change it")
                result = next(event for event in host.events(session) if event.kind == "tool.result")
            self.assertIn("-old", result.payload["diff"])
            self.assertIn("+new", result.payload["diff"])

    def test_rewind_preserves_file_changes_after_cut(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            model = ScriptModel(
                [
                    ModelReply(text="first"),
                    ModelReply(
                        tool_calls=[
                            ToolCall(
                                id="write",
                                name="write_file",
                                arguments={"path": "out.txt", "content": "changed"},
                            )
                        ]
                    ),
                    ModelReply(text="done"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                session = host.create_session(str(workspace))
                host.run_turn(session, "first")
                host.run_turn(session, "change")
                target = next(event for event in host.events(session) if event.kind == "user.message" and event.payload["text"] == "change")
                self.assertEqual((workspace / "out.txt").read_text(encoding="utf-8"), "changed")
                host.fork_before_user_seq(session, target.seq)
            self.assertEqual((workspace / "out.txt").read_text(encoding="utf-8"), "changed")

    def test_existing_host_client_uses_host_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            with Host(root, model=ScriptModel([ModelReply(text="hello")]), policy=AutoAllow()) as host:
                host.serve_background()
                client = HostClient(root)
                session = client.create_session(str(workspace))
                outcome = client.run_turn(session, "hello", yes=True)
                self.assertEqual(outcome.status, "completed")
                self.assertEqual(client.get_session(session)["id"], session)


if __name__ == "__main__":
    unittest.main()
