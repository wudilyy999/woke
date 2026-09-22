from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Callable

from woke.acp import AcpPolicy, _prompt_parts

REPO = Path(__file__).resolve().parents[1]


class AcpClient:
    """Talks to `woke acp` the way an editor does: JSON-RPC, one line each."""

    def __init__(
        self,
        root: Path,
        ws: Path,
        fake: str,
        answer: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        env = {**os.environ, "PYTHONPATH": str(REPO), "WOKE_FAKE_MODEL": fake}
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "woke",
                "--root",
                str(root),
                "acp",
                "--workspace",
                str(ws),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(REPO),
        )
        self.answer = answer
        self.seen: list[dict[str, Any]] = []
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 1
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue
            self.seen.append(message)
            if message.get("method") == "session/request_permission" and "id" in message:
                option = self.answer(message) if self.answer else "reject_once"
                self.reply(message["id"], {"outcome": {"outcome": "selected", "optionId": option}})
            self._queue.put(message)

    def send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        with self._lock:
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def reply(self, msg_id: Any, result: dict[str, Any]) -> None:
        self.send({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        msg_id = f"r{self._next_id}"
        self._next_id += 1
        self.send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        return self.wait_for(lambda message: message.get("id") == msg_id)

    def prompt(self, session_id: str, text: str) -> dict[str, Any]:
        msg_id = f"r{self._next_id}"
        self._next_id += 1
        self.send(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": "session/prompt",
                "params": {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]},
            }
        )
        return self.wait_for(lambda message: message.get("id") == msg_id, timeout=60)

    def wait_for(
        self, predicate: Callable[[dict[str, Any]], bool], timeout: float = 20
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"timed out waiting; saw {self.seen}")
            try:
                message = self._queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if predicate(message):
                return message

    def updates(self) -> list[dict[str, Any]]:
        return [
            message["params"]["update"]
            for message in self.seen
            if message.get("method") == "session/update"
        ]

    def stop(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)


class AcpTests(unittest.TestCase):
    def test_initialize_new_session_prompt_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            client = AcpClient(root, ws, "hello")
            try:
                init = client.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
                self.assertEqual(init["result"]["protocolVersion"], 1)
                self.assertTrue(init["result"]["agentCapabilities"]["loadSession"])
                self.assertTrue(
                    init["result"]["agentCapabilities"]["promptCapabilities"]["image"]
                )

                created = client.request("session/new", {"cwd": str(ws), "mcpServers": []})
                sid = created["result"]["sessionId"]
                done = client.prompt(sid, "say hello")
                self.assertEqual(done["result"]["stopReason"], "end_turn")

                chunks = [
                    update["content"]["text"]
                    for update in client.updates()
                    if update["sessionUpdate"] == "agent_message_chunk"
                ]
                self.assertIn("hello from woke", "".join(chunks))

                client.seen.clear()
                loaded = client.request("session/load", {"sessionId": sid, "cwd": str(ws)})
                self.assertEqual(loaded["result"], {})
                replayed = [update["sessionUpdate"] for update in client.updates()]
                self.assertIn("user_message_chunk", replayed)
                self.assertIn("agent_message_chunk", replayed)
            finally:
                client.stop()

    def test_approved_tool_call_streams_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            client = AcpClient(root, ws, "write_then_done", answer=lambda _m: "allow_once")
            try:
                client.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
                sid = client.request("session/new", {"cwd": str(ws), "mcpServers": []})["result"][
                    "sessionId"
                ]
                done = client.prompt(sid, "write out.txt")
                self.assertEqual(done["result"]["stopReason"], "end_turn")
                self.assertEqual((ws / "out.txt").read_text(encoding="utf-8"), "hello-woke")

                updates = client.updates()
                calls = [
                    update
                    for update in updates
                    if update["sessionUpdate"] == "tool_call"
                ]
                results = [
                    update
                    for update in updates
                    if update["sessionUpdate"] == "tool_call_update"
                ]
                self.assertEqual(calls[0]["title"], "write_file")
                self.assertEqual(calls[0]["kind"], "edit")
                self.assertEqual(calls[0]["status"], "in_progress")
                self.assertEqual(results[0]["status"], "completed")
                self.assertEqual(results[0]["toolCallId"], calls[0]["toolCallId"])

                asked = [
                    message
                    for message in client.seen
                    if message.get("method") == "session/request_permission"
                ]
                self.assertEqual(len(asked), 1)
                tool_call = asked[0]["params"]["toolCall"]
                self.assertEqual(tool_call["title"], "write_file")
                self.assertEqual(tool_call["toolCallId"], calls[0]["toolCallId"])
                options = {option["optionId"] for option in asked[0]["params"]["options"]}
                self.assertEqual(
                    options, {"allow_once", "allow_always", "reject_once"}
                )
            finally:
                client.stop()

    def test_denied_tool_call_never_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            client = AcpClient(root, ws, "write_then_done", answer=lambda _m: "reject_once")
            try:
                client.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
                sid = client.request("session/new", {"cwd": str(ws), "mcpServers": []})["result"][
                    "sessionId"
                ]
                done = client.prompt(sid, "write out.txt")
                self.assertEqual(done["result"]["stopReason"], "end_turn")
                self.assertFalse((ws / "out.txt").exists())
                failed = [
                    update
                    for update in client.updates()
                    if update["sessionUpdate"] == "tool_call_update"
                ]
                self.assertEqual(failed[0]["status"], "failed")
                self.assertIn("permission denied", failed[0]["content"][0]["content"]["text"])
            finally:
                client.stop()

    def test_cancel_notification_ends_the_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            client = AcpClient(root, ws, "write_then_done")
            try:
                client.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
                sid = client.request("session/new", {"cwd": str(ws), "mcpServers": []})["result"][
                    "sessionId"
                ]

                def answer(_message: dict[str, Any]) -> str:
                    client.notify("session/cancel", {"sessionId": sid})
                    return "reject_once"

                client.answer = answer
                done = client.prompt(sid, "write out.txt")
                self.assertEqual(done["result"]["stopReason"], "cancelled")
                self.assertFalse((ws / "out.txt").exists())
            finally:
                client.stop()

    def test_prompt_parts_store_images_in_the_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            data = base64.b64encode(b"fake-png-bytes").decode("ascii")
            text, images = _prompt_parts(
                [
                    {"type": "text", "text": "look at this"},
                    {"type": "image", "data": data, "mimeType": "image/png"},
                ],
                ws,
            )
            self.assertEqual(text, "look at this")
            self.assertEqual(len(images), 1)
            stored = ws / images[0]
            self.assertTrue(stored.is_file())
            self.assertEqual(stored.read_bytes(), b"fake-png-bytes")
            self.assertTrue(images[0].startswith(".woke/attachments/"))

    def test_policy_maps_client_answers(self) -> None:
        class StubServer:
            def __init__(self, options: list[str]) -> None:
                self.options = list(options)
                self.asked: list[str] = []

            def request_permission(self, _session: str, name: str, _arguments: dict) -> str:
                self.asked.append(name)
                return self.options.pop(0)

        server = StubServer(["allow_once", "allow_always", "reject_once"])
        policy = AcpPolicy(server, "s")  # type: ignore[arg-type]
        self.assertEqual(policy.decide("run_shell", {}), "allow")
        self.assertEqual(policy.decide("run_shell", {}), "allow")
        self.assertEqual(server.asked, ["run_shell", "run_shell"])
        # allow_always is remembered, so this one never reaches the client.
        self.assertEqual(policy.decide("run_shell", {}), "allow")
        self.assertEqual(server.asked, ["run_shell", "run_shell"])
        self.assertEqual(policy.decide("write_file", {}), "deny")
        self.assertEqual(server.asked, ["run_shell", "run_shell", "write_file"])


if __name__ == "__main__":
    unittest.main()
