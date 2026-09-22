"""User-configured hooks around tool calls and turn end."""

from __future__ import annotations

import fnmatch
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from woke.errors import ValidationError

HOOK_FILE = Path(".woke") / "hooks.json"
HOOK_TIMEOUT = 30
EVENT_NAMES = {
    "pretooluse": "PreToolUse",
    "posttooluse": "PostToolUse",
    "turnend": "TurnEnd",
}


@dataclass(frozen=True)
class Hook:
    event: str
    match: str
    command: str


def load_hooks(workspace: Path) -> list[Hook]:
    path = Path(workspace) / HOOK_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: {exc}") from exc
    raw = data.get("hooks") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise ValidationError(f"{path}: hooks must be a list")
    hooks: list[Hook] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValidationError(f"{path}: each hook must be an object")
        event = str(item.get("event") or "").replace("_", "").lower()
        if event not in EVENT_NAMES:
            raise ValidationError(f"{path}: unknown hook event {item.get('event')!r}")
        command = str(item.get("command") or "").strip()
        if not command:
            raise ValidationError(f"{path}: hook command is required")
        hooks.append(
            Hook(event=event, match=str(item.get("match") or "*"), command=command)
        )
    return hooks


def matching(hooks: list[Hook], event: str, tool: str = "") -> list[Hook]:
    out = []
    for hook in hooks:
        if hook.event != event:
            continue
        if tool:
            if fnmatch.fnmatch(tool, hook.match):
                out.append(hook)
        elif hook.match == "*":
            out.append(hook)
    return out


def run_hook(hook: Hook, payload: dict[str, Any], workspace: Path) -> tuple[bool, str]:
    """Feed the payload as JSON on stdin; a non-zero exit marks the hook failed."""
    try:
        proc = subprocess.run(
            hook.command,
            shell=True,
            cwd=workspace,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=HOOK_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, f"hook timed out after {HOOK_TIMEOUT}s"
    output = (proc.stdout or "").strip()
    error = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return False, error or output or f"hook exited {proc.returncode}"
    return True, output
