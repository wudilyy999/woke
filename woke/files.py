from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from typing import Any

from woke.events import Event
from woke.tools import SKIP_DIRS, contained_path

MENTION = re.compile(r"(^|\s)@([^\s@]+)")
ATTACH_CAP = 80_000


def expand_mentions(text: str, workspace: Path) -> list[dict[str, str]]:
    attachments: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in MENTION.finditer(text):
        rel = match.group(2).lstrip("./")
        if not rel or rel in seen:
            continue
        seen.add(rel)
        try:
            path = contained_path(workspace, rel)
        except Exception:
            continue
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(raw) > ATTACH_CAP:
            raw = raw[:ATTACH_CAP] + "\n…[truncated]"
        attachments.append({"path": rel, "content": raw})
    return attachments


def list_workspace_files(workspace: Path, prefix: str = "", limit: int = 40) -> list[str]:
    root = Path(workspace).resolve()
    prefix = prefix.lstrip("./")
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            full = Path(dirpath) / name
            try:
                rel = str(full.relative_to(root))
            except ValueError:
                continue
            if prefix and prefix.lower() not in rel.lower() and not rel.lower().startswith(prefix.lower()):
                continue
            out.append(rel)
            if len(out) >= limit:
                return sorted(out)
    return sorted(out)


def unified_diff(path: str, before: str | None, after: str | None, limit: int = 80) -> str:
    old = (before or "").splitlines()
    new = (after or "").splitlines()
    lines = list(
        difflib.unified_diff(old, new, fromfile="a/" + path, tofile="b/" + path, lineterm="")
    )
    if len(lines) > limit:
        lines = lines[:limit] + [f"… ({len(lines) - limit} more diff lines)"]
    return "\n".join(lines)


def snapshot_before(workspace: Path, rel: str) -> str | None:
    try:
        path = contained_path(workspace, rel)
    except Exception:
        return None
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def restore_files_after_cut(workspace: Path, events: list[Event], cut_seq: int) -> list[str]:
    """Restore write_file/str_replace paths to their content at cut_seq.

    Uses `before` captured on the first mutating tool.call at or after the cut.
    Shell mutations are not restored. Returns restored relative paths.
    """
    first: dict[str, Any] = {}
    for event in events:
        if event.seq < cut_seq or event.kind != "tool.call":
            continue
        name = str(event.payload.get("name") or "")
        if name not in {"write_file", "str_replace"}:
            continue
        rel = str((event.payload.get("arguments") or {}).get("path") or "")
        if not rel or rel in first:
            continue
        if "before" not in event.payload:
            continue
        first[rel] = event.payload.get("before")
    restored: list[str] = []
    for rel, before in first.items():
        try:
            dest = contained_path(workspace, rel)
        except Exception:
            continue
        if before is None:
            if dest.exists() and dest.is_file():
                dest.unlink()
                restored.append(rel)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(str(before), encoding="utf-8")
        restored.append(rel)
    return restored
