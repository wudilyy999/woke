"""Agent Client Protocol server: an editor drives a Host session over stdio.

JSON-RPC 2.0, one message per line. The agent streams ``session/update``
notifications while it works and asks the client through
``session/request_permission`` before a dangerous tool.
"""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, TextIO

from woke import __version__
from woke.errors import HostLocked, NotFound, WokeError
from woke.events import TURN_STATUSES, Event
from woke.host import Host
from woke.policy import WaitUser

PROTOCOL_VERSION = 1
PERMISSION_TIMEOUT = 600.0
ATTACHMENT_DIR = Path(".woke") / "attachments"

TOOL_KINDS = {
    "read_file": "read",
    "list_dir": "read",
    "memory_read": "read",
    "grep": "search",
    "web_search": "search",
    "web_fetch": "fetch",
    "write_file": "edit",
    "str_replace": "edit",
    "memory_write": "edit",
    "run_shell": "execute",
    "spawn_agent": "think",
}

PERMISSION_OPTIONS = [
    {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
    {"optionId": "allow_always", "name": "Allow this tool for the session", "kind": "allow_always"},
    {"optionId": "reject_once", "name": "Deny", "kind": "reject_once"},
]


class AcpError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AcpServer:
    def __init__(
        self,
        host: Host,
        workspace: Path,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.host = host
        self.workspace = Path(workspace).expanduser().resolve()
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self._write_lock = threading.Lock()
        self._client_lock = threading.Lock()
        self._requests: dict[str, threading.Event] = {}
        self._responses: dict[str, dict[str, Any]] = {}
        self._open_call: dict[str, str] = {}
        self._next_request = 1

    def serve(self) -> None:
        for raw in self.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict):
                self.handle(message)

    def handle(self, message: dict[str, Any]) -> None:
        if "method" not in message:
            self._resolve(message)
            return
        method = str(message["method"])
        msg_id = message.get("id")
        try:
            if method == "initialize":
                result = self._initialize()
            elif method == "authenticate":
                result = {}
            elif method == "session/new":
                result = self._new_session(message.get("params") or {})
            elif method == "session/load":
                result = self._load_session(message.get("params") or {})
            elif method == "session/prompt":
                self._prompt(msg_id, message.get("params") or {})
                return
            elif method == "session/cancel":
                params = message.get("params") or {}
                self.host.cancel_turn(str(params.get("sessionId") or ""))
                return
            else:
                raise AcpError(-32601, f"unknown method: {method}")
        except AcpError as exc:
            self._error(msg_id, exc.code, exc.message)
            return
        except WokeError as exc:
            self._error(msg_id, -32602, str(exc))
            return
        if msg_id is not None:
            self._send({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def request_permission(self, session_id: str, name: str, arguments: dict[str, Any]) -> str:
        """Ask the client about one dangerous call. Blocks the turn thread."""
        request_id = f"perm-{self._next_request}"
        self._next_request += 1
        waiter = threading.Event()
        with self._client_lock:
            self._requests[request_id] = waiter
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "session/request_permission",
                "params": {
                    "sessionId": session_id,
                    "toolCall": {
                        "toolCallId": self._open_call.get(session_id) or request_id,
                        "title": name,
                        "kind": TOOL_KINDS.get(name, "other"),
                        "status": "pending",
                        "rawInput": arguments,
                    },
                    "options": PERMISSION_OPTIONS,
                },
            }
        )
        if not waiter.wait(timeout=PERMISSION_TIMEOUT):
            with self._client_lock:
                self._requests.pop(request_id, None)
            return "reject_once"
        with self._client_lock:
            response = self._responses.pop(request_id, {})
        outcome = response.get("outcome")
        if not isinstance(outcome, dict) or outcome.get("outcome") != "selected":
            return "reject_once"
        return str(outcome.get("optionId") or "reject_once")

    def _resolve(self, message: dict[str, Any]) -> None:
        request_id = str(message.get("id") or "")
        with self._client_lock:
            waiter = self._requests.pop(request_id, None)
            if waiter is None:
                return
            result = message.get("result")
            self._responses[request_id] = result if isinstance(result, dict) else {}
        waiter.set()

    def _initialize(self) -> dict[str, Any]:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentCapabilities": {
                "loadSession": True,
                "promptCapabilities": {"image": True, "audio": False, "embeddedContext": True},
            },
            "agentInfo": {"name": "woke", "version": __version__},
            "authMethods": [],
        }

    def _new_session(self, params: dict[str, Any]) -> dict[str, Any]:
        cwd = Path(str(params.get("cwd") or self.workspace)).expanduser()
        if not cwd.is_dir():
            raise AcpError(-32602, f"cwd is not a directory: {cwd}")
        session_id = self.host.create_session(str(cwd), title=cwd.name)
        return {"sessionId": session_id}

    def _load_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = str(params.get("sessionId") or "")
        for event in self.host.events(session_id):
            if event.kind == "user.message":
                self._update(
                    session_id,
                    {
                        "sessionUpdate": "user_message_chunk",
                        "content": {"type": "text", "text": str(event.payload.get("text") or "")},
                    },
                )
            elif event.kind == "model.message":
                text = str(event.payload.get("text") or "")
                if text:
                    self._update(
                        session_id,
                        {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": text},
                        },
                    )
            else:
                self._on_event(session_id, event)
        return {}

    def _prompt(self, msg_id: Any, params: dict[str, Any]) -> None:
        session_id = str(params.get("sessionId") or "")
        info = self.host.get_session(session_id)
        text, images = _prompt_parts(
            params.get("prompt") or [], Path(str(info["workspace"]))
        )
        if not text.strip() and not images:
            raise AcpError(-32602, "prompt is empty")

        def work() -> None:
            try:
                outcome = self.host.run_turn(
                    session_id,
                    text,
                    images=images,
                    policy=AcpPolicy(self, session_id),
                    on_delta=lambda chunk: self._update(
                        session_id,
                        {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": chunk},
                        },
                    ),
                    on_event=lambda event: self._on_event(session_id, event),
                )
            except NotFound as exc:
                self._error(msg_id, -32602, str(exc))
                return
            except WokeError as exc:
                self._error(msg_id, -32603, str(exc))
                return
            if outcome.status not in TURN_STATUSES:
                self._error(msg_id, -32603, f"unexpected turn status {outcome.status}")
                return
            if outcome.status == "failed":
                self._error(msg_id, -32603, outcome.error or "turn failed")
                return
            stop = "cancelled" if outcome.status == "cancelled" else "end_turn"
            self._send({"jsonrpc": "2.0", "id": msg_id, "result": {"stopReason": stop}})

        threading.Thread(target=work, name=f"woke-acp-{session_id[:8]}", daemon=True).start()

    def _on_event(self, session_id: str, event: Event) -> None:
        if event.kind == "tool.call":
            name = str(event.payload.get("name") or "")
            call_id = str(event.payload.get("id") or "")
            self._open_call[session_id] = call_id
            self._update(
                session_id,
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": call_id,
                    "title": name,
                    "kind": TOOL_KINDS.get(name, "other"),
                    "status": "in_progress",
                    "rawInput": event.payload.get("arguments") or {},
                },
            )
        elif event.kind == "tool.result":
            ok = bool(event.payload.get("ok"))
            body = event.payload.get("output") if ok else event.payload.get("error")
            self._update(
                session_id,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": str(event.payload.get("id") or ""),
                    "status": "completed" if ok else "failed",
                    "content": [
                        {
                            "type": "content",
                            "content": {"type": "text", "text": str(body or "")},
                        }
                    ],
                },
            )
        elif event.kind == "todo.updated":
            self._update(
                session_id,
                {
                    "sessionUpdate": "plan",
                    "entries": [
                        {
                            "content": str(item["text"]),
                            "priority": "medium",
                            "status": str(item["status"]),
                        }
                        for item in event.payload.get("items") or []
                    ],
                },
            )

    def _update(self, session_id: str, update: dict[str, Any]) -> None:
        self._send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": session_id, "update": update},
            }
        )

    def _error(self, msg_id: Any, code: int, message: str) -> None:
        if msg_id is None:
            return
        self._send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}})

    def _send(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, ensure_ascii=False)
        with self._write_lock:
            self.stdout.write(line + "\n")
            self.stdout.flush()


class AcpPolicy:
    """Dangerous calls go to the editor; nothing is silently allowed."""

    mode = "ask"

    def __init__(self, server: AcpServer, session_id: str) -> None:
        self.server = server
        self.session_id = session_id
        self.granted: set[str] = set()

    def decide(
        self,
        name: str,
        arguments: dict[str, Any],
        grants: frozenset[str] = frozenset(),
    ) -> str:
        if name in grants or name in self.granted:
            return "allow"
        option = self.server.request_permission(self.session_id, name, arguments)
        if option == "allow_once":
            return "allow"
        if option == "allow_always":
            self.granted.add(name)
            return "allow"
        return "deny"


def _prompt_parts(prompt: Any, workspace: Path) -> tuple[str, list[str]]:
    texts: list[str] = []
    images: list[str] = []
    for part in prompt if isinstance(prompt, list) else []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            texts.append(str(part.get("text") or ""))
        elif kind == "resource_link":
            texts.append(f"@{part.get('uri')}")
        elif kind == "image":
            images.append(_store_image(workspace, part))
    return "\n".join(texts).strip(), images


def _store_image(workspace: Path, part: dict[str, Any]) -> str:
    data = str(part.get("data") or "")
    mime = str(part.get("mimeType") or "image/png")
    try:
        blob = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AcpError(-32602, f"image part is not valid base64: {exc}") from exc
    suffix = mimetypes.guess_extension(mime) or ".png"
    folder = Path(workspace).resolve() / ATTACHMENT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(blob)
    return str(path.relative_to(Path(workspace).resolve()))


def run_acp(
    root: Path,
    workspace: Path,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    from woke.model import build_model
    from woke.store import init_root

    init_root(root)
    try:
        host = Host(root, model=build_model(), policy=WaitUser())
    except HostLocked as exc:
        print(f"woke acp: {exc}", file=sys.stderr)
        return 1
    try:
        AcpServer(host, workspace, stdin=stdin, stdout=stdout).serve()
    finally:
        host.close()
    return 0
