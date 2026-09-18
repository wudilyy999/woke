from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from woke.errors import PathEscapes, ValidationError

DANGEROUS = frozenset({"write_file", "str_replace", "run_shell"})
SKIP_DIRS = frozenset({".git", "node_modules", ".venv", "__pycache__", ".woke"})
OUTPUT_CAP = 100_000
READ_MAX_BYTES = 1_000_000
GREP_MAX_FILE_BYTES = 1_000_000
SHELL_TIMEOUT = 30
REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "read_file": ("path",),
    "write_file": ("path", "content"),
    "str_replace": ("path", "old", "new"),
    "grep": ("pattern",),
    "run_shell": ("command",),
    "memory_write": ("text",),
}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file in the workspace. Lines are 1-based.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "1-based start line"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a text file in the workspace, creating parents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": "Replace exactly one occurrence of old in a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List a directory relative to the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search workspace files with a Python regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command with cwd set to the workspace. Not a sandbox.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read durable notes in .woke/memory.md.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Append (default) or replace durable notes in .woke/memory.md.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "mode": {"type": "string", "description": "append or replace"},
                },
                "required": ["text"],
            },
        },
    },
]


def contained_path(workspace: Path, rel: str) -> Path:
    ws = workspace.resolve()
    raw = Path(rel)
    path = raw.resolve() if raw.is_absolute() else (ws / rel).resolve()
    try:
        path.relative_to(ws)
    except ValueError as exc:
        raise PathEscapes(rel) from exc
    return path


def execute(name: str, arguments: dict[str, Any], workspace: Path) -> tuple[bool, str]:
    if not isinstance(arguments, dict):
        return False, "arguments must be an object"
    missing = [key for key in REQUIRED_ARGS.get(name, ()) if key not in arguments]
    if missing:
        return False, f"missing argument: {', '.join(missing)}"
    try:
        if name == "read_file":
            return True, _read_file(workspace, arguments)
        if name == "write_file":
            return True, _write_file(workspace, arguments)
        if name == "str_replace":
            return True, _str_replace(workspace, arguments)
        if name == "list_dir":
            return True, _list_dir(workspace, arguments)
        if name == "grep":
            return True, _grep(workspace, arguments)
        if name == "run_shell":
            return True, _run_shell(workspace, arguments)
        if name == "memory_read":
            from woke.memory import read_memory

            return True, read_memory(workspace)
        if name == "memory_write":
            from woke.memory import write_memory

            mode = str(arguments.get("mode") or "append")
            if mode not in {"append", "replace"}:
                return False, "mode must be append or replace"
            return True, write_memory(workspace, str(arguments.get("text") or ""), mode=mode)
        return False, f"unknown tool: {name}"
    except PathEscapes as exc:
        return False, str(exc)
    except (OSError, ValidationError, re.error, subprocess.SubprocessError, ValueError, TypeError, KeyError) as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001 — tool failures must not kill the turn
        return False, f"{type(exc).__name__}: {exc}"


def cap_output(text: str) -> tuple[str, bool]:
    if len(text) <= OUTPUT_CAP:
        return text, False
    return text[:OUTPUT_CAP] + "\n…[truncated]", True


def _read_file(workspace: Path, arguments: dict[str, Any]) -> str:
    path = contained_path(workspace, str(arguments["path"]))
    if not path.is_file():
        raise ValidationError(f"not a file: {arguments['path']}")
    size = path.stat().st_size
    offset = int(arguments.get("offset") or 1)
    limit = arguments.get("limit")
    if limit is None and size > READ_MAX_BYTES:
        raise ValidationError(
            f"file too large ({size} bytes); pass offset and limit to read a slice"
        )
    head = path.read_bytes()[:4096]
    if b"\x00" in head:
        raise ValidationError("binary file")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(offset, 1) - 1
    chunk = lines[start : start + int(limit)] if limit is not None else lines[start:]
    numbered = [f"{i + start + 1}\t{line}" for i, line in enumerate(chunk)]
    return "\n".join(numbered)


def _write_file(workspace: Path, arguments: dict[str, Any]) -> str:
    path = contained_path(workspace, str(arguments["path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    content = str(arguments.get("content") or "")
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(workspace.resolve())} ({len(content)} bytes)"


def _str_replace(workspace: Path, arguments: dict[str, Any]) -> str:
    path = contained_path(workspace, str(arguments["path"]))
    old = str(arguments["old"])
    new = str(arguments["new"])
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        raise ValidationError("old string not found")
    if count != 1:
        raise ValidationError(f"old string is not unique ({count} matches)")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"replaced 1 occurrence in {path.relative_to(workspace.resolve())}"


def _list_dir(workspace: Path, arguments: dict[str, Any]) -> str:
    path = contained_path(workspace, str(arguments.get("path") or "."))
    if not path.is_dir():
        raise ValidationError("not a directory")
    names = []
    for entry in sorted(path.iterdir(), key=lambda p: p.name):
        names.append(entry.name + ("/" if entry.is_dir() else ""))
    return "\n".join(names)


def _grep(workspace: Path, arguments: dict[str, Any]) -> str:
    pattern = re.compile(str(arguments["pattern"]))
    root = contained_path(workspace, str(arguments.get("path") or "."))
    hits: list[str] = []
    files = [root] if root.is_file() else _walk_files(root)
    for file_path in files:
        try:
            if file_path.stat().st_size > GREP_MAX_FILE_BYTES:
                continue
            sample = file_path.read_bytes()[:4096]
            if b"\x00" in sample:
                continue
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        rel = file_path.relative_to(workspace.resolve())
        for i, line in enumerate(lines, start=1):
            if pattern.search(line):
                hits.append(f"{rel}:{i}:{line}")
                if len(hits) >= 100:
                    return "\n".join(hits)
    return "\n".join(hits) if hits else "(no matches)"


def _walk_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for name in filenames:
            out.append(Path(dirpath) / name)
    return out


def _run_shell(workspace: Path, arguments: dict[str, Any]) -> str:
    command = str(arguments["command"]).strip()
    if not command:
        raise ValidationError("command is empty")
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=workspace.resolve(),
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationError(f"timed out after {SHELL_TIMEOUT}s") from exc
    parts = [
        f"exit {completed.returncode}",
        completed.stdout.rstrip(),
        completed.stderr.rstrip(),
    ]
    return "\n".join(part for part in parts if part)
