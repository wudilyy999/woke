from __future__ import annotations

import html
import os
import queue
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from woke.errors import PathEscapes, ToolCancelled, ValidationError
from woke.sandbox import SandboxManager, SandboxUnavailable

DANGEROUS = frozenset({"write_file", "str_replace", "run_shell", "web_fetch", "web_search"})
SKIP_DIRS = frozenset({".git", "node_modules", ".venv", "__pycache__", ".woke"})
OUTPUT_CAP = 100_000
READ_MAX_BYTES = 1_000_000
GREP_MAX_FILE_BYTES = 1_000_000
SHELL_TIMEOUT = 30
WEB_TIMEOUT = 20
WEB_MAX_BYTES = 2_000_000
WEB_USER_AGENT = "Mozilla/5.0 (compatible; woke/0.1)"
SEARCH_URL = "https://lite.duckduckgo.com/lite/?q="
REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "read_file": ("path",),
    "write_file": ("path", "content"),
    "str_replace": ("path", "old", "new"),
    "grep": ("pattern",),
    "run_shell": ("command",),
    "web_fetch": ("url",),
    "web_search": ("query",),
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
            "description": (
                "Run a shell command with cwd set to the workspace. The command runs in a "
                "platform sandbox that confines writes to the workspace and tmp directories."
            ),
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
            "name": "web_fetch",
            "description": "Fetch an http(s) URL and return its text content.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return the top results with snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "description": "Number of results, default 5"},
                },
                "required": ["query"],
            },
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


def execute(
    name: str,
    arguments: dict[str, Any],
    workspace: Path,
    sandbox: SandboxManager | None = None,
    on_output: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[bool, str]:
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
            return True, _run_shell(workspace, arguments, sandbox, on_output, should_cancel)
        if name == "web_fetch":
            return True, _web_fetch(arguments)
        if name == "web_search":
            return True, _web_search(arguments)
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


def _run_shell(
    workspace: Path,
    arguments: dict[str, Any],
    sandbox: SandboxManager | None,
    on_output: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> str:
    command = str(arguments["command"]).strip()
    if not command:
        raise ValidationError("command is empty")
    if sandbox is None:
        raise SandboxUnavailable("sandbox is required for run_shell")
    argv = sandbox.wrap_command(command, workspace, allow_write=True)
    proc = subprocess.Popen(
        argv,
        cwd=str(workspace.resolve()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    chunks: list[str] = []
    deadline = time.monotonic() + SHELL_TIMEOUT
    try:
        while True:
            if should_cancel is not None and should_cancel():
                raise ToolCancelled("cancelled")
            if time.monotonic() > deadline:
                raise ValidationError(f"timed out after {SHELL_TIMEOUT}s")
            try:
                line = lines.get(timeout=0.05)
            except queue.Empty:
                continue
            if line is None:
                break
            chunks.append(line)
            if on_output is not None:
                on_output(line)
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    parts = [
        f"exit {code}",
        "".join(chunks).rstrip(),
    ]
    return "\n".join(part for part in parts if part)


def _http_get(url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValidationError(f"unsupported url: {url}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": WEB_USER_AGENT, "Accept": "text/html,text/plain,*/*"},
    )
    with urllib.request.urlopen(request, timeout=WEB_TIMEOUT) as response:
        raw = response.read(WEB_MAX_BYTES)
        charset = response.headers.get_content_charset() or "utf-8"
        final_url = response.geturl()
    return raw.decode(charset, errors="replace"), final_url


def _web_fetch(arguments: dict[str, Any]) -> str:
    page, final_url = _http_get(str(arguments["url"]).strip())
    body = _html_to_text(page)
    return f"{final_url}\n\n{body}" if body else f"{final_url}\n\n(no text content)"


def _web_search(arguments: dict[str, Any]) -> str:
    query = str(arguments["query"]).strip()
    if not query:
        raise ValidationError("query is empty")
    limit = int(arguments.get("limit") or 5)
    page, _final_url = _http_get(SEARCH_URL + urllib.parse.quote(query))
    results = _search_results(page)
    if not results:
        raise ValidationError("search returned no parseable results")
    lines: list[str] = []
    for index, (url, title, snippet) in enumerate(results[:limit], start=1):
        lines.append(f"{index}. {title}\n{url}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


def _search_results(page: str) -> list[tuple[str, str, str]]:
    snippets = [
        _html_to_text(block)
        for block in re.findall(
            r"<td[^>]*class=['\"]result-snippet['\"][^>]*>(.*?)</td>", page, re.S
        )
    ]
    results: list[tuple[str, str, str]] = []
    for tag, title in re.findall(
        r"(<a[^>]*class=['\"]result-link['\"][^>]*>)(.*?)</a>", page, re.S
    ):
        href = re.search(r"href=['\"]([^'\"]+)['\"]", tag)
        if href is None:
            continue
        index = len(results)
        snippet = " ".join(snippets[index].split()) if index < len(snippets) else ""
        results.append((_unwrap_search_url(href.group(1)), _html_to_text(title), snippet))
    return results


def _unwrap_search_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return href


def _html_to_text(markup: str) -> str:
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", markup)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|tr|h[1-6]|td|table)\s*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
