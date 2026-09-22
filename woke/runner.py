from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from woke.errors import SimulatedCrash, ToolCancelled, TurnCancelled
from woke.events import Event, TODO_STATUSES
from woke.files import expand_mentions, snapshot_before, unified_diff
from woke.memory import load_briefing
from woke.model import Model, ModelReply
from woke.policy import Policy, load_rules
from woke.projection import (
    call_ids,
    compact_cut_seq,
    estimate_tokens,
    last_model_message,
    latest_compaction,
    open_run_id,
    project_messages,
    render_for_summary,
    result_ids,
    session_grants,
    workspace_of,
)
from woke.registry import MAX_SPAWN_DEPTH, SPAWN_NAME, TODO_NAME, ToolRegistry
from woke.store import Store
from woke.tools import cap_output

MAX_STEPS = 20
DEFAULT_BUDGET = 32_000
MAX_COMPACT_ROUNDS = 4
PARALLEL_SAFE = frozenset(
    {"read_file", "list_dir", "grep", "memory_read", "web_fetch", "web_search", SPAWN_NAME}
)

SYSTEM = """You are woke, a coding agent.
The workspace is {workspace}.
Use tools to read and change files. Prefer the smallest change that finishes the task.
The event log is durable; do not claim work is done unless a tool result shows it.
If a tool fails, read the error and try a different approach. Do not repeat the same failing call.
Use memory_read/memory_write for notes that should survive compaction (.woke/memory.md).
MCP tools are named mcp__<server>__<tool>. spawn_agent runs a nested agent with its own session log; do not nest further.
Independent calls in one reply run together, so batch parallel reads and subagents into a single reply.
"""

PLAN_MODE = """
You are in plan mode. Inspect the workspace with read-only tools, then answer with a concrete,
ordered plan. Do not modify files, run state-changing commands, or start sub-agents until the
user leaves plan mode.
"""

OnEvent = Callable[[Event], None]
OnPhase = Callable[[str], None]
OnDelta = Callable[[str], None]
OnToolOutput = Callable[[str], None]


def _never_cancel() -> bool:
    return False


@dataclass
class TurnOutcome:
    status: str
    events: list[Event] = field(default_factory=list)
    pending_call_id: str | None = None
    error: str | None = None


class Engine:
    def __init__(
        self,
        store: Store,
        model: Model,
        policy: Policy,
        token_budget: int = DEFAULT_BUDGET,
        crash_after_tool_call: bool = False,
        on_event: OnEvent | None = None,
        registry: ToolRegistry | None = None,
        depth: int = 0,
        spawn_child: Callable[..., tuple[bool, str]] | None = None,
        on_phase: OnPhase | None = None,
        on_delta: OnDelta | None = None,
        on_tool_output: OnToolOutput | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        self.store = store
        self.model = model
        self.policy = policy
        self.token_budget = token_budget
        self.crash_after_tool_call = crash_after_tool_call or os.environ.get(
            "WOKE_CRASH_AFTER_TOOL_CALL"
        ) == "1"
        self._crash_exit = os.environ.get("WOKE_CRASH_AFTER_TOOL_CALL") == "1"
        self.on_event = on_event
        self.registry = registry or ToolRegistry()
        self.depth = depth
        self.spawn_child = spawn_child
        self.on_phase = on_phase
        self.on_delta = on_delta
        self.on_tool_output = on_tool_output
        self.should_cancel = should_cancel if should_cancel is not None else _never_cancel
        self.mode = "execute"

    def _phase(self, text: str) -> None:
        if self.on_phase:
            self.on_phase(text)

    def _model_delta(self, text: str) -> None:
        if self.should_cancel():
            raise TurnCancelled()
        if self.on_delta:
            self.on_delta(text)

    def emit(
        self,
        session_id: str,
        kind: str,
        payload: dict[str, Any],
        turn_id: str | None = None,
        run_id: str | None = None,
    ) -> Event:
        event = self.store.append(session_id, kind, payload, turn_id=turn_id, run_id=run_id)
        if self.on_event:
            self.on_event(event)
        return event

    def start_turn(self, session_id: str, text: str, mode: str = "execute") -> TurnOutcome:
        turn_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        self.mode = mode
        collected: list[Event] = []
        previous = self.on_event

        def capture(event: Event) -> None:
            collected.append(event)
            if previous:
                previous(event)

        self.on_event = capture
        try:
            self.emit(session_id, "turn.started", {"mode": mode}, turn_id=turn_id)
            self.emit(session_id, "run.started", {"reason": "fresh"}, turn_id=turn_id, run_id=run_id)
            self.emit(
                session_id,
                "user.message",
                {
                    "text": text,
                    "attachments": expand_mentions(text, Path(workspace_of(self.store.read_session(session_id)))),
                },
                turn_id=turn_id,
                run_id=run_id,
            )
            outcome = self._loop(session_id, turn_id, run_id)
            outcome.events = collected
            return outcome
        finally:
            self.on_event = previous

    def continue_turn(self, session_id: str, turn_id: str, run_id: str) -> TurnOutcome:
        collected: list[Event] = []
        previous = self.on_event

        def capture(event: Event) -> None:
            collected.append(event)
            if previous:
                previous(event)

        self.on_event = capture
        try:
            paused = self._dispatch_pending_tools(session_id, turn_id, run_id)
            if paused is not None:
                paused.events = collected
                return paused
            outcome = self._loop(session_id, turn_id, run_id)
            outcome.events = collected
            return outcome
        finally:
            self.on_event = previous

    def recover_session(self, session_id: str) -> TurnOutcome | None:
        events = self.store.read_session(session_id)
        turn_id = None
        for event in events:
            if event.kind == "turn.started":
                turn_id = event.turn_id
            elif event.kind == "turn.terminated":
                turn_id = None
        if turn_id is None:
            return None

        dangling = call_ids(events, turn_id) - result_ids(events, turn_id)
        open_run = open_run_id(events, turn_id)
        for event in events:
            if (
                event.turn_id == turn_id
                and event.kind == "tool.call"
                and event.payload["id"] in dangling
            ):
                call_id = event.payload["id"]
                requested = False
                decided = False
                for item in events:
                    if item.turn_id != turn_id:
                        continue
                    if item.kind == "permission.requested" and item.payload["id"] == call_id:
                        requested = True
                    if item.kind == "permission.decided" and item.payload["id"] == call_id:
                        decided = True
                if requested and not decided:
                    self.emit(
                        session_id,
                        "permission.decided",
                        {"id": call_id, "decision": "deny", "source": "recovery"},
                        turn_id=turn_id,
                        run_id=event.run_id,
                    )
                self.emit(
                    session_id,
                    "tool.result",
                    {
                        "id": call_id,
                        "name": event.payload["name"],
                        "ok": False,
                        "output": "",
                        "error": "interrupted_by_crash",
                    },
                    turn_id=turn_id,
                    run_id=event.run_id,
                )
        if open_run:
            self.emit(
                session_id,
                "run.terminated",
                {"status": "aborted", "error": "crash"},
                turn_id=turn_id,
                run_id=open_run,
            )

        run_id = str(uuid.uuid4())
        self.emit(
            session_id,
            "run.started",
            {"reason": "recovery"},
            turn_id=turn_id,
            run_id=run_id,
        )
        return self._loop(session_id, turn_id, run_id)

    def compact(self, session_id: str, force: bool = False, turn_id: str | None = None, run_id: str | None = None) -> Event | None:
        last: Event | None = None
        rounds = 1 if force else MAX_COMPACT_ROUNDS
        for _ in range(rounds):
            events = self.store.read_session(session_id)
            if not events:
                return last
            over = estimate_tokens(_with_system(events)) > self.token_budget
            if not force and not over:
                return last
            prev = latest_compaction(events)
            start = int(prev.payload["to_seq"]) + 1 if prev else events[0].seq
            to_seq = compact_cut_seq(events, turn_id)
            if to_seq is None or to_seq < start:
                return last
            chunk = [event for event in events if start <= event.seq <= to_seq]
            if not chunk:
                return last
            prior = str(prev.payload.get("summary") or "") if prev else ""
            blob = ""
            if prior:
                blob += "Previous summary:\n" + prior + "\n\n"
            blob += render_for_summary(chunk)
            self._phase("compacting context")
            try:
                summary = self.model.summarize(blob)
            except Exception:
                self._phase("")
                return last
            self._phase("thinking")
            if not str(summary).strip():
                return last
            last = self.emit(
                session_id,
                "compaction.applied",
                {
                    "summary": str(summary),
                    "from_seq": start,
                    "to_seq": to_seq,
                    "folded_previous": bool(prior),
                },
                turn_id=turn_id,
                run_id=run_id,
            )
            if force:
                return last
        return last

    def _loop(self, session_id: str, turn_id: str, run_id: str) -> TurnOutcome:
        for _ in range(MAX_STEPS):
            if self.should_cancel():
                return self._cancel(session_id, turn_id, run_id)
            try:
                self.compact(session_id, turn_id=turn_id, run_id=run_id)
            except Exception:
                pass
            events = self.store.read_session(session_id)
            self._phase("waiting for model")
            try:
                reply = self.model.complete(
                    _with_system(events, self.mode), self.registry.specs(), on_delta=self._model_delta
                )
            except SimulatedCrash:
                raise
            except TurnCancelled:
                return self._cancel(session_id, turn_id, run_id)
            except Exception as exc:
                self._phase("")
                return self._fail(session_id, turn_id, run_id, str(exc))
            self._phase("thinking")
            if self.should_cancel():
                return self._cancel(session_id, turn_id, run_id)
            if not reply.text and not reply.tool_calls:
                reply = ModelReply(text="(empty model response)")
            self._emit_model(session_id, turn_id, run_id, reply)
            if not reply.tool_calls:
                self.emit(
                    session_id,
                    "run.terminated",
                    {"status": "completed"},
                    turn_id=turn_id,
                    run_id=run_id,
                )
                self.emit(
                    session_id,
                    "turn.terminated",
                    {"status": "completed"},
                    turn_id=turn_id,
                )
                return TurnOutcome(status="completed")
            paused = self._dispatch_pending_tools(session_id, turn_id, run_id)
            if paused is not None:
                return paused
        return self._fail(session_id, turn_id, run_id, "max_steps")

    def _fail(self, session_id: str, turn_id: str, run_id: str, error: str) -> TurnOutcome:
        self.emit(
            session_id,
            "run.terminated",
            {"status": "failed", "error": error},
            turn_id=turn_id,
            run_id=run_id,
        )
        self.emit(
            session_id,
            "turn.terminated",
            {"status": "failed", "error": error},
            turn_id=turn_id,
        )
        return TurnOutcome(status="failed", error=error)

    def _cancel(self, session_id: str, turn_id: str, run_id: str) -> TurnOutcome:
        self.emit(
            session_id,
            "run.terminated",
            {"status": "aborted", "error": "cancelled"},
            turn_id=turn_id,
            run_id=run_id,
        )
        self.emit(
            session_id,
            "turn.terminated",
            {"status": "cancelled", "error": "cancelled"},
            turn_id=turn_id,
        )
        return TurnOutcome(status="cancelled", error="cancelled")

    def _emit_model(self, session_id: str, turn_id: str, run_id: str, reply: ModelReply) -> None:
        payload: dict[str, Any] = {"text": reply.text or ""}
        if reply.usage:
            payload["usage"] = reply.usage
        if reply.tool_calls:
            payload["tool_calls"] = [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in reply.tool_calls
            ]
        self.emit(session_id, "model.message", payload, turn_id=turn_id, run_id=run_id)

    def _dispatch_pending_tools(self, session_id: str, turn_id: str, run_id: str) -> TurnOutcome | None:
        events = self.store.read_session(session_id)
        model_event = last_model_message(events, turn_id)
        if model_event is None:
            return None
        done = result_ids(events, turn_id)
        written_calls = call_ids(events, turn_id)
        workspace = Path(workspace_of(events))
        grants = session_grants(events)
        rules = load_rules(workspace)
        pending: list[tuple[str, str, dict[str, Any]]] = []
        for call in model_event.payload.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or uuid.uuid4())
            name = str(call.get("name") or "")
            arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            if call_id in done:
                continue
            if call_id not in written_calls:
                call_payload: dict[str, Any] = {
                    "id": call_id,
                    "name": name,
                    "arguments": arguments,
                }
                if name in {"write_file", "str_replace"}:
                    call_payload["before"] = snapshot_before(workspace, str(arguments["path"]))
                self.emit(
                    session_id,
                    "tool.call",
                    call_payload,
                    turn_id=turn_id,
                    run_id=run_id,
                )
                self._maybe_crash()
            events = self.store.read_session(session_id)
            if self.should_cancel():
                self.emit(
                    session_id,
                    "tool.result",
                    {
                        "id": call_id,
                        "name": name,
                        "ok": False,
                        "output": "",
                        "error": "cancelled",
                    },
                    turn_id=turn_id,
                    run_id=run_id,
                )
                return self._cancel(session_id, turn_id, run_id)
            if self.registry.is_dangerous(name):
                decided = None
                requested = False
                for event in events:
                    if event.turn_id != turn_id:
                        continue
                    if event.kind == "permission.requested" and event.payload["id"] == call_id:
                        requested = True
                    if event.kind == "permission.decided" and event.payload["id"] == call_id:
                        decided = event.payload["decision"]
                if not requested:
                    self.emit(
                        session_id,
                        "permission.requested",
                        {"id": call_id, "name": name, "arguments": arguments},
                        turn_id=turn_id,
                        run_id=run_id,
                    )
                if decided is None:
                    decision = self.policy.decide(name, arguments, grants=grants)
                    if decision == "wait" and rules.allow(name, arguments):
                        decision = "allow"
                    source = "grant" if decision == "allow" and name in grants else "policy"
                    if decision == "wait":
                        return TurnOutcome(status="paused", pending_call_id=call_id)
                    self.emit(
                        session_id,
                        "permission.decided",
                        {
                            "id": call_id,
                            "decision": decision,
                            "source": source,
                            "name": name,
                            "scope": "session" if source == "grant" else "once",
                        },
                        turn_id=turn_id,
                        run_id=run_id,
                    )
                    decided = decision
                if decided == "deny":
                    self.emit(
                        session_id,
                        "tool.result",
                        {
                            "id": call_id,
                            "name": name,
                            "ok": False,
                            "output": "",
                            "error": "permission denied",
                        },
                        turn_id=turn_id,
                        run_id=run_id,
                    )
                    continue
            pending.append((call_id, name, arguments))
        results = self._run_tools(session_id, turn_id, run_id, pending, workspace)
        for (call_id, name, _arguments), (ok, output, diff) in zip(pending, results):
            output, truncated = cap_output(output)
            payload: dict[str, Any] = {
                "id": call_id,
                "name": name,
                "ok": ok,
                "output": output,
            }
            if truncated:
                payload["truncated"] = True
            if not ok:
                payload["error"] = output
            if diff is not None:
                payload["diff"] = diff
            self.emit(session_id, "tool.result", payload, turn_id=turn_id, run_id=run_id)
            if self.should_cancel():
                return self._cancel(session_id, turn_id, run_id)
        return None

    def _run_tools(
        self,
        session_id: str,
        turn_id: str,
        run_id: str,
        pending: list[tuple[str, str, dict[str, Any]]],
        workspace: Path,
    ) -> list[tuple[bool, str, str | None]]:
        """Run the approved calls. Read-only batches go wide, the rest keep their order."""
        if len(pending) > 1 and all(name in PARALLEL_SAFE for _, name, _ in pending):
            with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                futures = [
                    pool.submit(
                        self._execute_tool, session_id, turn_id, run_id, call_id, name, arguments, workspace
                    )
                    for call_id, name, arguments in pending
                ]
                return [future.result() for future in futures]
        return [
            self._execute_tool(session_id, turn_id, run_id, call_id, name, arguments, workspace)
            for call_id, name, arguments in pending
        ]

    def _execute_tool(
        self,
        session_id: str,
        turn_id: str,
        run_id: str,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        workspace: Path,
    ) -> tuple[bool, str, str | None]:
        try:
            self._phase(f"running {name}")
            if name == TODO_NAME:
                ok, output = self._todo(session_id, arguments, turn_id, run_id)
            elif name == SPAWN_NAME:
                ok, output = self._spawn(session_id, arguments, workspace)
            else:
                ok, output = self.registry.execute(
                    name,
                    arguments,
                    workspace,
                    on_output=self.on_tool_output,
                    should_cancel=self.should_cancel,
                )
        except ToolCancelled:
            ok, output = False, "cancelled"
        except SimulatedCrash:
            raise
        except Exception as exc:
            ok, output = False, f"{type(exc).__name__}: {exc}"
        self._phase("thinking")
        diff = None
        if ok and name in {"write_file", "str_replace"}:
            before = next(
                event.payload.get("before")
                for event in reversed(self.store.read_session(session_id))
                if event.kind == "tool.call" and event.payload.get("id") == call_id
            )
            diff = unified_diff(
                str(arguments["path"]), before, snapshot_before(workspace, str(arguments["path"]))
            )
        return ok, output, diff

    def _spawn(self, parent_id: str, arguments: dict[str, Any], workspace: Path) -> tuple[bool, str]:
        if self.depth >= MAX_SPAWN_DEPTH:
            return False, "max spawn depth"
        if self.spawn_child is None:
            return False, "spawn_agent is not available"
        return self.spawn_child(parent_id, arguments, workspace, self.depth)

    def _todo(
        self,
        session_id: str,
        arguments: dict[str, Any],
        turn_id: str,
        run_id: str,
    ) -> tuple[bool, str]:
        items = arguments.get("items")
        if not isinstance(items, list) or not items:
            return False, "items must be a non-empty list"
        normalized: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                return False, "each item must be an object"
            text = str(item.get("text") or "").strip()
            status = str(item.get("status") or "")
            if not text:
                return False, "item text is required"
            if status not in TODO_STATUSES:
                return False, f"bad item status: {status}"
            normalized.append({"text": text, "status": status})
        self.emit(
            session_id,
            "todo.updated",
            {"items": normalized},
            turn_id=turn_id,
            run_id=run_id,
        )
        done = sum(1 for item in normalized if item["status"] == "completed")
        return True, f"{done}/{len(normalized)} todos complete"

    def _maybe_crash(self) -> None:
        if not self.crash_after_tool_call:
            return
        if self._crash_exit:
            os._exit(2)
        raise SimulatedCrash("crash after tool.call")


def _render_todos(events: list[Event]) -> str:
    latest = None
    for event in events:
        if event.kind == "todo.updated":
            latest = event
    if latest is None:
        return ""
    marks = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}
    return "\n".join(f"{marks[item['status']]} {item['text']}" for item in latest.payload["items"])


def _with_system(events: list[Event], mode: str = "execute") -> list[dict[str, Any]]:
    workspace = workspace_of(events)
    content = SYSTEM.format(workspace=workspace)
    if mode == "plan":
        content += "\n" + PLAN_MODE
    briefing = load_briefing(Path(workspace))
    if briefing:
        content += "\n\n# Workspace memory\n" + briefing
    todos = _render_todos(events)
    if todos:
        content += "\n\n# Task list\n" + todos
    return [{"role": "system", "content": content}] + project_messages(events)
