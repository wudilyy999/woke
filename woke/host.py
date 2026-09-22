from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from woke.errors import AuthError, HostDown, NotFound, SessionBusy, WokeError
from woke.events import Event
from woke.mcp import McpHub
from woke.model import Model, build_model
from woke.policy import AutoAllow, AutoDeny, Policy, WaitUser, parse_policy
from woke.projection import last_model_message, open_run_id, open_turn_id, pending_permission_id
from woke.registry import ToolRegistry
from woke.runner import DEFAULT_BUDGET, Engine, TurnOutcome
from woke.sandbox import SandboxManager
from woke.store import (
    Store,
    acquire_lock,
    clear_host_meta,
    init_root,
    read_root_meta,
    release_lock,
    write_host_meta,
)


class Host:
    """Single-writer execution authority for one state root."""

    def __init__(
        self,
        root: Path,
        model: Model | None = None,
        policy: Policy | None = None,
        token_budget: int = DEFAULT_BUDGET,
        crash_after_tool_call: bool = False,
    ) -> None:
        self.root = Path(root)
        init_root(self.root)
        self.meta = read_root_meta(self.root)
        self._lock_fd = acquire_lock(self.root)
        self.store = Store(self.root)
        self.model = model or build_model()
        self.policy = policy or WaitUser()
        self.token_budget = token_budget
        self.crash_after_tool_call = crash_after_tool_call
        self._session_locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._active_turns: set[str] = set()
        self._httpd: ThreadingHTTPServer | None = None
        self.port: int | None = None
        self._closed = False
        self.sandbox = SandboxManager()
        self.mcp = McpHub.load(self.root, sandbox=self.sandbox, workspace=self.root)
        self.mcp_errors = self.mcp.start()
        self.registry = ToolRegistry(hub=self.mcp, allow_spawn=True, sandbox=self.sandbox)
        self.phase = ""
        self.recover_all()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        httpd = self._httpd
        self._httpd = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        clear_host_meta(self.root)
        self.mcp.close()
        self.store.close()
        release_lock(self._lock_fd)

    def __enter__(self) -> Host:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _engine(
        self,
        policy: Policy | None = None,
        on_event: Any = None,
        on_delta: Any = None,
        on_tool_output: Any = None,
        should_cancel: Any = None,
        depth: int = 0,
    ) -> Engine:
        registry = self.registry if depth == 0 else self.registry.child()
        return Engine(
            store=self.store,
            model=self.model,
            policy=policy or self.policy,
            token_budget=self.token_budget,
            crash_after_tool_call=self.crash_after_tool_call,
            on_event=on_event,
            on_delta=on_delta,
            on_tool_output=on_tool_output,
            should_cancel=should_cancel,
            registry=registry,
            depth=depth,
            spawn_child=self._spawn_agent if depth == 0 else None,
            on_phase=self._set_phase,
        )

    def _set_phase(self, text: str) -> None:
        self.phase = text

    def _spawn_agent(
        self,
        parent_id: str,
        arguments: dict[str, Any],
        workspace: Path,
        depth: int,
    ) -> tuple[bool, str]:
        task = str(arguments.get("task") or "").strip()
        if not task:
            return False, "task is required"
        label = str(arguments.get("label") or "subagent")
        child_id = self.create_session(str(workspace), title=label, parent_session_id=parent_id)
        outcome = self._engine(policy=AutoAllow(), depth=depth + 1).start_turn(child_id, task)
        events = self.store.read_session(child_id)
        turn_id = None
        for event in events:
            if event.kind == "turn.started":
                turn_id = event.turn_id
        last = last_model_message(events, turn_id) if turn_id else None
        digest = (last.payload.get("text") or "") if last else ""
        return True, f"child_session={child_id}\nstatus={outcome.status}\n{digest}".rstrip()

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._meta_lock:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            return lock

    def _begin_turn(self, session_id: str) -> threading.Event:
        cancel = threading.Event()
        with self._meta_lock:
            self._cancel_events[session_id] = cancel
            self._active_turns.add(session_id)
        return cancel

    def _end_turn(self, session_id: str) -> None:
        with self._meta_lock:
            self._cancel_events.pop(session_id, None)
            self._active_turns.discard(session_id)

    def create_session(
        self,
        workspace: str,
        title: str = "",
        parent_session_id: str | None = None,
    ) -> str:
        session_id = str(uuid.uuid4())
        path = str(Path(workspace).expanduser().resolve())
        Path(path).mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"workspace": path, "title": title or Path(path).name}
        if parent_session_id:
            payload["parent_session_id"] = parent_session_id
        self.store.append(session_id, "session.created", payload)
        return session_id

    def session_for_workspace(self, workspace: str) -> str:
        path = str(Path(workspace).expanduser().resolve())
        Path(path).mkdir(parents=True, exist_ok=True)
        for session in self.list_sessions():
            if session["workspace"] == path and not session.get("parent_session_id"):
                return session["id"]
        return self.create_session(path)

    def get_session(self, session_id: str) -> dict[str, Any]:
        events = self.store.read_session(session_id)
        if not events or events[0].kind != "session.created":
            raise NotFound(session_id)
        preview = ""
        for event in reversed(events):
            if event.kind == "user.message":
                preview = str(event.payload.get("text") or "").replace("\n", " ")[:72]
                break
        return {
            "id": session_id,
            "title": events[0].payload["title"],
            "workspace": events[0].payload["workspace"],
            "parent_session_id": events[0].payload.get("parent_session_id"),
            "open_turn": open_turn_id(events),
            "event_count": len(events),
            "updated": events[-1].ts,
            "preview": preview,
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        return [self.get_session(event.session_id) for event in self.store.list_sessions()]

    def list_root_sessions(self) -> list[dict[str, Any]]:
        items = [item for item in self.list_sessions() if not item.get("parent_session_id")]
        return list(reversed(items))

    def rewind_targets(self, session_id: str) -> list[dict[str, Any]]:
        """User turns that can be forked, newest first."""
        events = self.store.read_session(session_id)
        out: list[dict[str, Any]] = []
        for event in events:
            if event.kind != "user.message":
                continue
            text = str(event.payload.get("text") or "").replace("\n", " ").strip()
            if not text:
                continue
            out.append({"seq": event.seq, "turn_id": event.turn_id, "text": text[:80]})
        return list(reversed(out))

    def fork_before_user_seq(self, source_id: str, user_seq: int) -> tuple[str, str]:
        """New session with events strictly before this user.message. Original untouched.

        Returns (new_session_id, prefill_prompt). Filesystem state is left to git.
        """
        source = self.store.read_session(source_id)
        if not source or source[0].kind != "session.created":
            raise NotFound(source_id)
        target = next((e for e in source if e.kind == "user.message" and e.seq == user_seq), None)
        if target is None:
            raise NotFound(f"user.message seq={user_seq}")
        cut = user_seq
        for event in source:
            if event.kind == "turn.started" and event.turn_id == target.turn_id:
                cut = event.seq
                break
        created = source[0]
        title = str(created.payload.get("title") or "session")
        new_id = self.create_session(
            str(created.payload["workspace"]),
            title=f"{title} (rewind)",
        )
        # Annotate fork on the new session.created we just wrote.
        # create_session already appended; copy remaining prefix after it.
        seq_map: dict[int, int] = {}
        for event in source:
            if event.kind == "session.created":
                continue
            if event.seq >= cut:
                break
            payload = dict(event.payload)
            if event.kind == "compaction.applied":
                mapped = _remap_compact_seq(payload, seq_map)
                if mapped is None:
                    continue
                payload = mapped
            copied = self.store.append(
                new_id,
                event.kind,
                payload,
                turn_id=event.turn_id,
                run_id=event.run_id,
            )
            seq_map[event.seq] = copied.seq
        prefill = str(target.payload.get("text") or "")
        return new_id, prefill

    def fork_session(self, source_id: str) -> str:
        """Copy the whole session into a new id. Original untouched."""
        source = self.store.read_session(source_id)
        if not source or source[0].kind != "session.created":
            raise NotFound(source_id)
        created = source[0]
        title = str(created.payload.get("title") or "session")
        new_id = self.create_session(str(created.payload["workspace"]), title=f"{title} (fork)")
        seq_map: dict[int, int] = {}
        for event in source:
            if event.kind == "session.created":
                continue
            payload = dict(event.payload)
            if event.kind == "compaction.applied":
                mapped = _remap_compact_seq(payload, seq_map)
                if mapped is None:
                    continue
                payload = mapped
            copied = self.store.append(
                new_id,
                event.kind,
                payload,
                turn_id=event.turn_id,
                run_id=event.run_id,
            )
            seq_map[event.seq] = copied.seq
        return new_id

    def events(self, session_id: str, after: int = 0) -> list[Event]:
        self.get_session(session_id)
        return self.store.read_session(session_id, after=after)

    def run_turn(
        self,
        session_id: str,
        text: str,
        yes: bool = False,
        on_delta: Any = None,
        on_tool_output: Any = None,
        mode: str = "execute",
    ) -> TurnOutcome:
        self.get_session(session_id)
        policy = AutoDeny() if mode == "plan" else (parse_policy(yes) if yes else self.policy)
        with self._lock_for(session_id):
            events = self.store.read_session(session_id)
            if open_turn_id(events):
                raise SessionBusy(f"session {session_id} already has an open turn")
            cancel = self._begin_turn(session_id)
            try:
                return self._engine(
                    policy=policy,
                    on_delta=on_delta,
                    on_tool_output=on_tool_output,
                    should_cancel=cancel.is_set,
                ).start_turn(session_id, text, mode=mode)
            finally:
                self._end_turn(session_id)
                self.phase = ""

    def decide_permission(
        self,
        session_id: str,
        call_id: str,
        decision: str,
        scope: str = "once",
    ) -> TurnOutcome:
        if decision not in {"allow", "deny"}:
            raise WokeError(f"bad decision: {decision}")
        if scope not in {"once", "session"}:
            raise WokeError(f"bad scope: {scope}")
        self.get_session(session_id)
        with self._lock_for(session_id):
            events = self.store.read_session(session_id)
            turn_id = open_turn_id(events)
            if turn_id is None:
                raise SessionBusy("no open turn")
            pending = pending_permission_id(events, turn_id)
            if pending != call_id:
                raise WokeError(f"pending permission is {pending}, not {call_id}")
            run_id = open_run_id(events, turn_id)
            if run_id is None:
                raise SessionBusy("no open run")
            name = ""
            for event in events:
                if event.kind == "permission.requested" and event.payload.get("id") == call_id:
                    name = str(event.payload.get("name") or "")
            cancel = self._begin_turn(session_id)
            try:
                engine = self._engine(should_cancel=cancel.is_set)
                engine.emit(
                    session_id,
                    "permission.decided",
                    {
                        "id": call_id,
                        "decision": decision,
                        "source": "user",
                        "name": name,
                        "scope": scope if decision == "allow" else "once",
                    },
                    turn_id=turn_id,
                    run_id=run_id,
                )
                return engine.continue_turn(session_id, turn_id, run_id)
            finally:
                self._end_turn(session_id)
                self.phase = ""

    def cancel_turn(self, session_id: str) -> bool:
        """Stop the open turn in this session. Returns False when no turn is open."""
        self.get_session(session_id)
        with self._meta_lock:
            running = self._cancel_events.get(session_id) if session_id in self._active_turns else None
        if running is not None:
            running.set()
            return True
        with self._lock_for(session_id):
            with self._meta_lock:
                running = (
                    self._cancel_events.get(session_id)
                    if session_id in self._active_turns
                    else None
                )
            if running is not None:
                running.set()
                return True
            events = self.store.read_session(session_id)
            turn_id = open_turn_id(events)
            if turn_id is None:
                return False
            run_id = open_run_id(events, turn_id)
            if run_id is not None:
                self.store.append(
                    session_id,
                    "run.terminated",
                    {"status": "aborted", "error": "cancelled"},
                    turn_id=turn_id,
                    run_id=run_id,
                )
            self.store.append(
                session_id,
                "turn.terminated",
                {"status": "cancelled", "error": "cancelled"},
                turn_id=turn_id,
            )
            return True

    def compact(self, session_id: str, force: bool = True) -> Event | None:
        self.get_session(session_id)
        with self._lock_for(session_id):
            events = self.store.read_session(session_id)
            try:
                return self._engine().compact(
                    session_id,
                    force=force,
                    turn_id=open_turn_id(events),
                    run_id=open_run_id(events, open_turn_id(events) or ""),
                )
            finally:
                self.phase = ""

    def recover_all(self) -> list[TurnOutcome]:
        outcomes = []
        for session_id in self.store.session_ids():
            with self._lock_for(session_id):
                info = self.get_session(session_id)
                depth = 1 if info.get("parent_session_id") else 0
                policy = AutoAllow() if depth else self.policy
                engine = self._engine(policy=policy, depth=depth)
                try:
                    outcome = engine.recover_session(session_id)
                except Exception:
                    continue
                if outcome is not None:
                    outcomes.append(outcome)
        return outcomes

    def serve(self, port: int = 0) -> None:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler(self))
        self._httpd = httpd
        self.port = int(httpd.server_address[1])
        write_host_meta(self.root, os.getpid(), self.port)
        print(f"woke host on 127.0.0.1:{self.port} root={self.root}", flush=True)
        httpd.serve_forever()

    def serve_background(self, port: int = 0) -> int:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler(self))
        self._httpd = httpd
        self.port = int(httpd.server_address[1])
        write_host_meta(self.root, os.getpid(), self.port)
        thread = threading.Thread(target=httpd.serve_forever, name="woke-host", daemon=True)
        thread.start()
        return self.port


class HostClient:
    """Client facade used when another process already owns the Host lock."""

    def __init__(self, root: Path) -> None:
        from woke.store import read_host_meta

        meta = read_host_meta(root)
        if meta is None:
            raise HostDown(f"no host is running in {root}")
        self.root = Path(root)
        self.base = f"http://127.0.0.1:{meta['port']}"
        self.token = read_root_meta(root)["token"]
        self.phase = ""
        self.token_budget = DEFAULT_BUDGET
        self.mcp = type("McpState", (), {"servers": []})()
        self.mcp_errors: list[str] = []

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        import urllib.request

        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"X-Woke-Token": self.token, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/sessions/{session_id}")

    def list_sessions(self) -> list[dict[str, Any]]:
        return self._request("GET", "/sessions")["sessions"]

    def list_root_sessions(self) -> list[dict[str, Any]]:
        return list(reversed([item for item in self.list_sessions() if not item.get("parent_session_id")]))

    def session_for_workspace(self, workspace: str) -> str:
        path = str(Path(workspace).expanduser().resolve())
        for item in self.list_root_sessions():
            if item["workspace"] == path:
                return item["id"]
        return self.create_session(path)

    def create_session(self, workspace: str, title: str = "", parent_session_id: str | None = None) -> str:
        body = {"workspace": workspace, "title": title}
        if parent_session_id:
            body["parent_session_id"] = parent_session_id
        return self._request("POST", "/sessions", body)["id"]

    def events(self, session_id: str, after: int = 0) -> list[Event]:
        return [Event(**item) for item in self._request("GET", f"/sessions/{session_id}/events?after={after}")["events"]]

    def run_turn(
        self,
        session_id: str,
        text: str,
        yes: bool = False,
        on_delta: Any = None,
        on_tool_output: Any = None,
        mode: str = "execute",
    ) -> TurnOutcome:
        data = self._request(
            "POST",
            f"/sessions/{session_id}/turns",
            {"text": text, "yes": yes, "mode": mode},
        )
        outcome = _outcome_from_json(data)
        if on_delta:
            for event in outcome.events:
                if event.kind == "model.message" and event.payload.get("text"):
                    on_delta(event.payload["text"])
        return outcome

    def cancel_turn(self, session_id: str) -> bool:
        data = self._request("POST", f"/sessions/{session_id}/cancel", {})
        return bool(data.get("cancelled"))

    def decide_permission(self, session_id: str, call_id: str, decision: str, scope: str = "once") -> TurnOutcome:
        return _outcome_from_json(
            self._request(
                "POST",
                f"/sessions/{session_id}/permissions",
                {"id": call_id, "decision": decision, "scope": scope},
            )
        )

    def compact(self, session_id: str, force: bool = True) -> Event | None:
        data = self._request("POST", f"/sessions/{session_id}/compact", {})
        raw = data.get("event")
        return Event(**raw) if raw else None

    def fork_session(self, session_id: str) -> str:
        raise WokeError("fork is unavailable while attached to a remote Host")

    def rewind_targets(self, session_id: str) -> list[dict[str, Any]]:
        events = self.events(session_id)
        return [
            {"seq": event.seq, "turn_id": event.turn_id, "text": str(event.payload.get("text") or "")[:80]}
            for event in reversed([event for event in events if event.kind == "user.message"])
        ]

    def fork_before_user_seq(self, source_id: str, user_seq: int) -> tuple[str, str]:
        raise WokeError("rewind is unavailable while attached to a remote Host")

    def close(self) -> None:
        return


def _remap_compact_seq(payload: dict[str, Any], seq_map: dict[int, int]) -> dict[str, Any] | None:
    old_to = int(payload["to_seq"])
    mapped_to = seq_map.get(old_to)
    if mapped_to is None:
        return None
    old_from = int(payload["from_seq"])
    mapped_from = seq_map.get(old_from, min(seq_map.values()) if seq_map else mapped_to)
    out = dict(payload)
    out["from_seq"] = mapped_from
    out["to_seq"] = mapped_to
    return out


def _handler(host: Host) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            try:
                self._auth()
                parts, query = self._path()
                if parts == ["health"]:
                    return self._json(200, {"ok": True, "root_id": host.meta["root_id"]})
                if parts == ["sessions"]:
                    return self._json(200, {"sessions": host.list_sessions()})
                if len(parts) == 2 and parts[0] == "sessions":
                    return self._json(200, host.get_session(parts[1]))
                if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "events":
                    after = int((query.get("after") or ["0"])[0])
                    events = [event.to_dict() for event in host.events(parts[1], after=after)]
                    return self._json(200, {"events": events})
                self._json(404, {"error": "not found"})
            except Exception as exc:
                self._handle_error(exc)

        def do_POST(self) -> None:
            try:
                self._auth()
                parts, _query = self._path()
                body = self._body()
                if parts == ["shutdown"]:
                    def _stop() -> None:
                        time.sleep(0.05)
                        host.close()

                    threading.Thread(target=_stop, daemon=True).start()
                    return self._json(200, {"ok": True})
                if parts == ["sessions"]:
                    session_id = host.create_session(
                        str(body.get("workspace") or ""),
                        title=str(body.get("title") or ""),
                        parent_session_id=str(body.get("parent_session_id") or "") or None,
                    )
                    return self._json(201, host.get_session(session_id))
                if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "turns":
                    yes = bool(body.get("yes"))
                    mode = str(body.get("mode") or "execute")
                    outcome = host.run_turn(
                        parts[1], str(body.get("text") or ""), yes=yes, mode=mode
                    )
                    return self._json(200, _outcome_json(outcome))
                if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "permissions":
                    outcome = host.decide_permission(
                        parts[1],
                        str(body.get("id") or ""),
                        str(body.get("decision") or ""),
                        scope=str(body.get("scope") or "once"),
                    )
                    return self._json(200, _outcome_json(outcome))
                if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "compact":
                    event = host.compact(parts[1], force=True)
                    return self._json(200, {"event": None if event is None else event.to_dict()})
                if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "cancel":
                    return self._json(200, {"cancelled": host.cancel_turn(parts[1])})
                self._json(404, {"error": "not found"})
            except Exception as exc:
                self._handle_error(exc)

        def _auth(self) -> None:
            got = self.headers.get("X-Woke-Token") or ""
            auth = self.headers.get("Authorization") or ""
            if auth.startswith("Bearer "):
                got = auth[7:]
            import hmac

            if not hmac.compare_digest(got, host.meta["token"]):
                raise AuthError("bad token")

        def _path(self) -> tuple[list[str], dict[str, list[str]]]:
            parsed = urlparse(self.path)
            parts = [part for part in parsed.path.split("/") if part]
            return parts, parse_qs(parsed.query)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            if not raw:
                return {}
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}

        def _json(self, code: int, obj: dict[str, Any]) -> None:
            blob = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(blob)

        def _handle_error(self, exc: Exception) -> None:
            if isinstance(exc, AuthError):
                return self._json(401, {"error": str(exc)})
            if isinstance(exc, NotFound):
                return self._json(404, {"error": str(exc)})
            if isinstance(exc, SessionBusy):
                return self._json(409, {"error": str(exc)})
            if isinstance(exc, WokeError):
                return self._json(400, {"error": str(exc)})
            return self._json(500, {"error": str(exc)})

    return Handler


def _outcome_json(outcome: TurnOutcome) -> dict[str, Any]:
    return {
        "status": outcome.status,
        "pending_call_id": outcome.pending_call_id,
        "error": outcome.error,
        "events": [event.to_dict() for event in outcome.events],
    }


def _outcome_from_json(data: dict[str, Any]) -> TurnOutcome:
    return TurnOutcome(
        status=str(data.get("status") or "failed"),
        pending_call_id=data.get("pending_call_id"),
        error=data.get("error"),
        events=[Event(**item) for item in data.get("events") or []],
    )
