from __future__ import annotations

import json
from typing import Any

from woke.events import Event

PROMPT_TOOL_CAP = 8_000


def latest_compaction(events: list[Event]) -> Event | None:
    found = None
    for event in events:
        if event.kind == "compaction.applied":
            found = event
    return found


def projected_events(events: list[Event]) -> list[Event]:
    """Events that still appear in the model prompt after the latest compaction."""
    compact = latest_compaction(events)
    if compact is None:
        return list(events)
    cut = int(compact.payload["to_seq"])
    return [event for event in events if event.seq > cut]


def project_messages(events: list[Event], prune_tools: bool = True) -> list[dict[str, Any]]:
    """OpenAI-shaped conversation reconstructed from the log (a projection)."""
    compact = latest_compaction(events)
    messages: list[dict[str, Any]] = []
    if compact is not None:
        messages.append(
            {
                "role": "system",
                "content": "Prior work summary:\n" + compact.payload["summary"],
            }
        )
        visible = [event for event in events if event.seq > int(compact.payload["to_seq"])]
    else:
        visible = events

    for event in visible:
        if event.kind == "user.message":
            content = event.payload["text"]
            for attachment in event.payload.get("attachments") or []:
                content += f"\n\n@{attachment['path']}\n```\n{attachment['content']}\n```"
            messages.append({"role": "user", "content": content})
        elif event.kind == "model.message":
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": event.payload.get("text") or None,
            }
            calls = event.payload.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                        },
                    }
                    for call in calls
                ]
            messages.append(msg)
        elif event.kind == "tool.result":
            content = event.payload.get("output") or ""
            if not event.payload.get("ok"):
                err = event.payload.get("error") or "tool failed"
                content = f"ERROR: {err}\n{content}".rstrip()
            if prune_tools:
                content = prune_tool_content(content)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": event.payload["id"],
                    "content": content,
                }
            )
    return messages


def prune_tool_content(text: str, limit: int = PROMPT_TOOL_CAP) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…[pruned from prompt; full text remains in the log]"


def session_grants(events: list[Event]) -> frozenset[str]:
    names: set[str] = set()
    for event in events:
        if event.kind != "permission.decided":
            continue
        if event.payload.get("decision") != "allow":
            continue
        if event.payload.get("scope") != "session":
            continue
        name = event.payload.get("name")
        if name:
            names.add(str(name))
    return frozenset(names)


def compact_cut_seq(events: list[Event], turn_id: str | None) -> int | None:
    """Last seq that can enter a summary without dropping an open tool pair."""
    if not events:
        return None
    compact = latest_compaction(events)
    start = int(compact.payload["to_seq"]) + 1 if compact else events[0].seq
    current_start = None
    if turn_id:
        for event in events:
            if event.kind == "turn.started" and event.turn_id == turn_id:
                current_start = event.seq
                break
    last_term = None
    for event in events:
        if event.kind != "turn.terminated" or event.seq < start:
            continue
        if current_start is not None and event.seq >= current_start:
            continue
        last_term = event.seq
    if last_term is not None:
        return last_term
    last_safe = None
    for event in events:
        if event.seq < start:
            continue
        if event.kind in {"tool.result", "turn.terminated"}:
            last_safe = event.seq
        elif event.kind == "model.message" and not event.payload.get("tool_calls"):
            last_safe = event.seq
    return last_safe


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += len(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            total += len(json.dumps(call, ensure_ascii=False))
    return max(1, total // 4)


def render_for_summary(events: list[Event]) -> str:
    lines: list[str] = []
    for event in events:
        payload = json.dumps(event.payload, ensure_ascii=False)
        if len(payload) > 2000:
            payload = payload[:2000] + "…"
        lines.append(f"#{event.seq} {event.kind} {payload}")
    return "\n".join(lines)


def workspace_of(events: list[Event]) -> str:
    for event in events:
        if event.kind == "session.created":
            return str(event.payload["workspace"])
    raise ValueError("session.created is missing")


def open_turn_id(events: list[Event]) -> str | None:
    started: str | None = None
    for event in events:
        if event.kind == "turn.started":
            started = event.turn_id
        elif event.kind == "turn.terminated":
            started = None
    return started


def open_run_id(events: list[Event], turn_id: str) -> str | None:
    started: str | None = None
    for event in events:
        if event.turn_id != turn_id:
            continue
        if event.kind == "run.started":
            started = event.run_id
        elif event.kind == "run.terminated":
            started = None
    return started


def last_model_message(events: list[Event], turn_id: str) -> Event | None:
    found = None
    for event in events:
        if event.turn_id == turn_id and event.kind == "model.message":
            found = event
    return found


def result_ids(events: list[Event], turn_id: str) -> set[str]:
    return {
        event.payload["id"]
        for event in events
        if event.turn_id == turn_id and event.kind == "tool.result"
    }


def call_ids(events: list[Event], turn_id: str) -> set[str]:
    return {
        event.payload["id"]
        for event in events
        if event.turn_id == turn_id and event.kind == "tool.call"
    }


def pending_permission_id(events: list[Event], turn_id: str) -> str | None:
    requested: set[str] = set()
    decided: set[str] = set()
    for event in events:
        if event.turn_id != turn_id:
            continue
        if event.kind == "permission.requested":
            requested.add(event.payload["id"])
        elif event.kind == "permission.decided":
            decided.add(event.payload["id"])
    open_ids = requested - decided
    return next(iter(open_ids), None)
