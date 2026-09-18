from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from woke.errors import UnknownKind, ValidationError

KINDS = frozenset(
    {
        "session.created",
        "turn.started",
        "turn.terminated",
        "run.started",
        "run.terminated",
        "user.message",
        "model.message",
        "tool.call",
        "tool.result",
        "permission.requested",
        "permission.decided",
        "compaction.applied",
    }
)

TURN_STATUSES = frozenset({"completed", "failed", "aborted", "cancelled"})
RUN_REASONS = frozenset({"fresh", "recovery"})
RUN_STATUSES = frozenset({"completed", "failed", "aborted"})
PERMISSION_DECISIONS = frozenset({"allow", "deny"})
PERMISSION_SOURCES = frozenset({"policy", "user", "recovery", "grant"})
PERMISSION_SCOPES = frozenset({"once", "session"})

_REQUIRED: dict[str, tuple[str, ...]] = {
    "session.created": ("workspace", "title"),
    "turn.started": (),
    "turn.terminated": ("status",),
    "run.started": ("reason",),
    "run.terminated": ("status",),
    "user.message": ("text",),
    "model.message": (),
    "tool.call": ("id", "name", "arguments"),
    "tool.result": ("id", "name", "ok"),
    "permission.requested": ("id", "name", "arguments"),
    "permission.decided": ("id", "decision", "source"),
    "compaction.applied": ("summary", "from_seq", "to_seq"),
}


@dataclass(frozen=True)
class Event:
    seq: int
    ts: str
    session_id: str
    turn_id: str | None
    run_id: str | None
    kind: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_event(
    kind: str,
    payload: dict[str, Any],
    turn_id: str | None,
    run_id: str | None,
) -> None:
    if kind not in KINDS:
        raise UnknownKind(kind)
    if not isinstance(payload, dict):
        raise ValidationError("payload must be an object")
    for key in _REQUIRED[kind]:
        if key not in payload:
            raise ValidationError(f"{kind} missing field {key}")

    if kind != "session.created" and kind != "compaction.applied" and not turn_id:
        raise ValidationError(f"{kind} requires turn_id")
    if kind.startswith("run.") or kind in {
        "user.message",
        "model.message",
        "tool.call",
        "tool.result",
        "permission.requested",
        "permission.decided",
    }:
        if not run_id:
            raise ValidationError(f"{kind} requires run_id")

    if kind == "session.created":
        _require_str(payload, "workspace")
        _require_str(payload, "title")
    elif kind == "turn.terminated":
        if payload["status"] not in TURN_STATUSES:
            raise ValidationError(f"bad turn status: {payload['status']}")
    elif kind == "run.started":
        if payload["reason"] not in RUN_REASONS:
            raise ValidationError(f"bad run reason: {payload['reason']}")
    elif kind == "run.terminated":
        if payload["status"] not in RUN_STATUSES:
            raise ValidationError(f"bad run status: {payload['status']}")
    elif kind == "user.message":
        _require_str(payload, "text")
    elif kind == "model.message":
        if "text" in payload and not isinstance(payload["text"], str):
            raise ValidationError("model.message text must be a string")
        if "tool_calls" in payload and not isinstance(payload["tool_calls"], list):
            raise ValidationError("model.message tool_calls must be a list")
        if not payload.get("text") and not payload.get("tool_calls"):
            raise ValidationError("model.message needs text or tool_calls")
    elif kind == "tool.call":
        _require_str(payload, "id")
        _require_str(payload, "name")
        if not isinstance(payload["arguments"], dict):
            raise ValidationError("tool.call arguments must be an object")
    elif kind == "tool.result":
        _require_str(payload, "id")
        _require_str(payload, "name")
        if not isinstance(payload["ok"], bool):
            raise ValidationError("tool.result ok must be a bool")
    elif kind == "permission.requested":
        _require_str(payload, "id")
        _require_str(payload, "name")
        if not isinstance(payload["arguments"], dict):
            raise ValidationError("permission.requested arguments must be an object")
    elif kind == "permission.decided":
        _require_str(payload, "id")
        if payload["decision"] not in PERMISSION_DECISIONS:
            raise ValidationError(f"bad permission decision: {payload['decision']}")
        if payload["source"] not in PERMISSION_SOURCES:
            raise ValidationError(f"bad permission source: {payload['source']}")
        if "scope" in payload and payload["scope"] not in PERMISSION_SCOPES:
            raise ValidationError(f"bad permission scope: {payload['scope']}")
    elif kind == "compaction.applied":
        _require_str(payload, "summary")
        if not isinstance(payload["from_seq"], int) or not isinstance(payload["to_seq"], int):
            raise ValidationError("compaction seq range must be int")
        if payload["from_seq"] < 1 or payload["to_seq"] < payload["from_seq"]:
            raise ValidationError("compaction seq range is invalid")


def _require_str(payload: dict[str, Any], key: str) -> None:
    if not isinstance(payload.get(key), str):
        raise ValidationError(f"{key} must be a string")
