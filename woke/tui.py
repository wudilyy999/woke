"""Codex-style interactive terminal: transcript, bottom composer, slash picker.

Layout and color roles follow OpenAI Codex CLI's published TUI style notes.
This is an independent curses UI, not a port of the Rust TUI.
"""

from __future__ import annotations

import curses
import locale
import select
import sys
import threading
from pathlib import Path
from typing import Any

from woke.commands import expand_command, load_commands
from woke.events import Event
from woke.files import image_rel
from woke.host import Host, HostClient
from woke.errors import HostLocked
from woke.model import active_model_label, build_model
from woke.policy import AutoAllow, GradedPolicy, WaitUser
from woke.projection import estimate_tokens, pending_permission_id, project_messages

COMMANDS: list[tuple[str, str]] = [
    ("/model", "switch model"),
    ("/workspace", "switch project directory"),
    ("/image", "attach an image to the next message"),
    ("/resume", "resume or search a previous session"),
    ("/rewind", "fork from an earlier user message"),
    ("/fork", "copy this session and keep going"),
    ("/new", "start a new conversation"),
    ("/plan", "toggle plan mode (read-only)"),
    ("/help", "keyboard shortcuts"),
    ("/clear", "new session in this workspace"),
    ("/permission", "readonly / ask / edits / auto"),
    ("/yes", "allow all tools (auto)"),
    ("/edits", "allow file edits, ask for shell"),
    ("/wait", "ask before each dangerous tool"),
    ("/readonly", "deny writes and shell"),
    ("/compact", "compact context"),
    ("/status", "session and MCP"),
    ("/quit", "exit"),
]

HELP = """/            command picker
/model       switch model (↑↓ then Enter)
/workspace   switch directory (↑↓ then Enter)
/image PATH  attach an image to the next message
/resume      resume a previous session; text searches every transcript
/rewind      fork from an earlier user turn (Esc Esc)
/fork        copy this session
/new         start a new conversation
/plan        toggle plan mode; planning turns stay read-only
/help        this list
/clear       new session
/permission  readonly | ask | edits | auto
/yes         allow all tools
/edits       allow file writes, ask for shell
/wait        ask every dangerous tool
/readonly    deny writes and shell
/compact     compact context
/quit        exit
custom       .woke/commands/<name>.md becomes /<name>
y / a / n    allow once / session / deny
Esc          close picker
"""

SPINNER = "|/-\\"
PICKER_PATHS = {
    "/model": "models",
    "/workspace": "workspace",
    "/permission": "permission",
    "/resume": "sessions",
    "/rewind": "rewind",
}

PERMISSION_CHOICES: list[tuple[str, str]] = [
    ("readonly", "read-only; deny writes and shell"),
    ("ask", "ask before each write, shell, or spawn"),
    ("edits", "auto-allow file edits; still ask for shell"),
    ("auto", "allow all tools"),
]


def char_width(ch: str) -> int:
    code = ord(ch)
    if code == 0 or code < 32 or 0x7F <= code < 0xA0:
        return 0
    if 0x0300 <= code <= 0x036F:
        return 0
    if (
        0x1100 <= code <= 0x115F
        or 0x2329 <= code <= 0x232A
        or 0x2E80 <= code <= 0xA4CF
        or 0xAC00 <= code <= 0xD7A3
        or 0xF900 <= code <= 0xFAFF
        or 0xFE10 <= code <= 0xFE19
        or 0xFE30 <= code <= 0xFE6F
        or 0xFF00 <= code <= 0xFF60
        or 0xFFE0 <= code <= 0xFFE6
        or 0x1F300 <= code <= 0x1FAFF
        or 0x20000 <= code <= 0x3FFFD
    ):
        return 2
    return 1


def display_width(text: str) -> int:
    return sum(char_width(ch) for ch in text)


def slice_width(text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""
    out: list[str] = []
    used = 0
    for ch in text:
        width = char_width(ch)
        if used + width > max_width:
            break
        out.append(ch)
        used += width
    return "".join(out)


def pad_width(text: str, width: int) -> str:
    clipped = slice_width(text, width)
    return clipped + " " * max(0, width - display_width(clipped))


def wrap_text(text: str, width: int) -> list[str]:
    width = max(8, width)
    lines: list[str] = []
    for paragraph in (text or "").split("\n"):
        if paragraph == "":
            lines.append("")
            continue
        rest = paragraph
        while display_width(rest) > width:
            acc = ""
            acc_w = 0
            cut = 0
            last_space = -1
            for i, ch in enumerate(rest):
                w = char_width(ch)
                if acc_w + w > width:
                    break
                acc += ch
                acc_w += w
                cut = i + 1
                if ch == " ":
                    last_space = i
            if last_space > 0 and last_space >= cut // 2:
                lines.append(rest[:last_space])
                rest = rest[last_space + 1 :]
            else:
                lines.append(rest[:cut])
                rest = rest[cut:]
        lines.append(rest)
    return lines or [""]


def filter_slash(prefix: str, custom: dict[str, str] | None = None) -> list[tuple[str, str]]:
    needle = prefix.strip().lower()
    items = list(COMMANDS)
    for name in sorted(custom or {}):
        items.append((f"/{name}", "custom command"))
    if needle == "/":
        return items
    return [item for item in items if item[0].startswith(needle)]


def _short(value: Any, limit: int = 80) -> str:
    if isinstance(value, dict):
        parts = []
        for key, item in list(value.items())[:4]:
            parts.append(f"{key}={item!s}"[:40])
        text = " ".join(parts)
    else:
        text = str(value)
    text = text.replace("\n", " ")
    return text if display_width(text) <= limit else slice_width(text, max(1, limit - 1)) + "…"


def diff_rows(diff: str, limit: int = 200) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in diff.splitlines()[:limit]:
        if line.startswith(("+++", "---", "@@")):
            style = "dim"
        elif line.startswith("+"):
            style = "ok"
        elif line.startswith("-"):
            style = "err"
        else:
            style = "default"
        rows.append((style, line))
    return rows


def transcript_lines(events: list[Event], width: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    inner = max(8, width - 2)

    def add(style: str, text: str) -> None:
        for line in wrap_text(text, inner):
            rows.append((style, line))

    for event in events:
        if event.kind == "user.message":
            add("dim", "you")
            add("user", event.payload.get("text") or "")
            rows.append(("default", ""))
        elif event.kind == "model.message":
            add("brand", "woke")
            text = event.payload.get("text") or ""
            if text:
                add("default", text)
            rows.append(("default", ""))
        elif event.kind == "tool.call":
            name = event.payload.get("name") or "tool"
            args = _short(event.payload.get("arguments") or {})
            add("dim", f"> {name}  {args}")
        elif event.kind == "tool.result":
            ok = bool(event.payload.get("ok"))
            body = event.payload.get("error") if not ok else (event.payload.get("output") or "")
            add("ok" if ok else "err", "  " + _short(body, inner - 2))
            diff = event.payload.get("diff")
            if ok and diff:
                for style, line in diff_rows(str(diff)):
                    add(style, "  " + line)
        elif event.kind == "permission.requested":
            add("user", f"  approval needed: {event.payload.get('name')}")
        elif event.kind == "permission.decided":
            decision = event.payload.get("decision")
            add("ok" if decision == "allow" else "err", f"  {decision}")
        elif event.kind == "compaction.applied":
            add("dim", f"  compacted seq {event.payload.get('from_seq')}-{event.payload.get('to_seq')}")
        elif event.kind == "todo.updated":
            add("dim", "tasks")
            for item in event.payload.get("items") or []:
                status = item.get("status")
                mark = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}.get(status, "[ ]")
                add("dim" if status == "completed" else "default", f"  {mark} {item.get('text')}")
        elif event.kind == "turn.terminated" and event.payload.get("status") != "completed":
            add("err", f"  turn {event.payload.get('status')}")
            if event.payload.get("error"):
                add("err", str(event.payload["error"]))
    return rows


def spinner_frame(tick: int) -> str:
    return SPINNER[tick % len(SPINNER)]


def activity_line(events: list[Event], busy: bool, tick: int, phase: str = "") -> str | None:
    if not busy and not phase:
        return None
    spin = spinner_frame(tick)
    text = (phase or "").strip()
    if not text:
        last_call = None
        last_result_ids: set[str] = set()
        for event in events:
            if event.kind == "tool.call":
                last_call = event
            elif event.kind == "tool.result":
                last_result_ids.add(str(event.payload.get("id") or ""))
            elif event.kind in {"model.message", "user.message", "turn.terminated"}:
                last_call = None
        if last_call is not None and str(last_call.payload.get("id") or "") not in last_result_ids:
            name = str(last_call.payload.get("name") or "tool")
            args = _short(last_call.payload.get("arguments") or {}, 40)
            text = f"running {name}  {args}"
        else:
            text = "waiting for model"
    return f"{spin}  {text}"


def welcome_lines(model: str, workspace: str, width: int) -> list[tuple[str, str]]:
    width = max(40, min(width, 62))
    inner = width - 4
    top = "╭" + "─" * (width - 2) + "╮"
    bot = "╰" + "─" * (width - 2) + "╯"

    def row(text: str) -> str:
        return "│ " + pad_width(text, inner) + " │"

    home = str(Path.home())
    shown = workspace
    if shown.startswith(home):
        shown = "~" + shown[len(home) :]
    return [
        ("dim", top),
        ("brand", row("woke")),
        ("dim", row("")),
        ("default", row(f"model: {model}")),
        ("default", row(f"directory: {shown}")),
        ("dim", bot),
        ("dim", ""),
        ("dim", "Enter send  / commands  ? shortcuts"),
    ]


def composer_frame(text: str, width: int, height: int) -> list[str]:
    width = max(20, width)
    inner = width - 4
    body = wrap_text("> " + text, inner)
    if not body:
        body = ["> "]
    height = max(3, height)
    usable = height - 2
    if len(body) > usable:
        body = body[-usable:]
    while len(body) < usable:
        body.append("")
    top = "╭" + "─" * (width - 2) + "╮"
    bot = "╰" + "─" * (width - 2) + "╯"
    lines = [top]
    for line in body:
        lines.append("│ " + pad_width(line, inner) + " │")
    lines.append(bot)
    return lines


def status_line(
    events: list[Event],
    budget: int,
    model: str,
    busy: bool,
    yes: bool,
    tick: int = 0,
    phase: str = "",
    perm: str = "",
    plan: bool = False,
) -> str:
    tokens = context_tokens(events)
    left = max(0, 100 - int(tokens * 100 / max(budget, 1)))
    perm_label = perm or ("auto" if yes else "ask")
    access = "plan" if plan else f"perm:{perm_label}"
    if busy or phase:
        act = activity_line(events, True, tick, phase) or f"{spinner_frame(tick)}  working"
        return f"{left}% context left  {model}  {access}  {act}  / commands"
    return f"{left}% context left  {model}  {access}  / commands"


def context_tokens(events: list[Event]) -> int:
    """Total tokens reported by the provider, else a chars/4 estimate of the prompt."""
    for event in reversed(events):
        if event.kind != "model.message":
            continue
        usage = event.payload.get("usage")
        if not usage:
            continue
        total = usage.get("total_tokens") or (
            (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
        )
        if total:
            return int(total)
    return estimate_tokens(project_messages(events))


def pending_call(events: list[Event]) -> tuple[str, str] | None:
    from woke.projection import open_turn_id

    turn_id = open_turn_id(events)
    if not turn_id:
        return None
    call_id = pending_permission_id(events, turn_id)
    if not call_id:
        return None
    name = ""
    for event in reversed(events):
        if event.kind == "permission.requested" and event.payload.get("id") == call_id:
            name = str(event.payload.get("name") or "")
            break
    return call_id, name


def _model_rows() -> list[tuple[str, str]]:
    from woke.config import load_config

    cfg = load_config()
    return [(spec.id, spec.display_name) for spec in cfg.models.values()]


def workspace_candidates(current: str, typed: str, recent: list[str]) -> list[tuple[str, str]]:
    home = Path.home()
    typed = typed.strip()
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []

    def add(path: Path, hint: str = "") -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        if not resolved.is_dir():
            return
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        label = key
        if label.startswith(str(home)):
            label = "~" + label[len(str(home)) :]
        rows.append((key, hint or label))

    if typed:
        raw = Path(typed).expanduser()
        if raw.is_dir():
            add(raw, "use this folder")
            add(raw.parent, "parent")
            try:
                children = sorted((p for p in raw.iterdir() if p.is_dir()), key=lambda p: p.name.lower())
            except OSError:
                children = []
            for child in children[:40]:
                if child.name.startswith(".") and child.name not in {".", ".."}:
                    continue
                add(child)
            return rows
        parent = raw.parent if str(raw.parent) != "" else Path.cwd()
        prefix = raw.name.lower()
        add(parent, "parent")
        try:
            children = sorted((p for p in parent.expanduser().iterdir() if p.is_dir()), key=lambda p: p.name.lower())
        except OSError:
            children = []
        for child in children:
            if prefix and prefix not in child.name.lower():
                continue
            add(child)
        return rows[:40]

    add(Path(current), "current")
    add(Path.cwd(), "cwd")
    add(home, "home")
    add(home / "Downloads")
    add(home / "Desktop")
    add(home / "Documents")
    for item in recent:
        add(Path(item), "recent")
    return rows


class Tui:
    def __init__(self, host: Host, session_id: str, yes: bool = False, model_id: str | None = None) -> None:
        self.host = host
        self.session_id = session_id
        self.yes = yes
        self.perm_mode = "auto" if yes else "ask"
        self.model_id = model_id
        self.input = ""
        self.scroll = 0
        self.busy = False
        self.error: str | None = None
        self.help = False
        self.notice: str | None = None
        self.plan_mode = False
        self.pending_images: list[str] = []
        self.live_text = ""
        self.live_tool_output = ""
        self.picker: str | None = None  # slash | models | workspace | permission
        self.menu_index = 0
        self.tick = 0
        self._esc_armed = False
        self._lock = threading.Lock()
        self._color = {}
        self.custom_commands: dict[str, str] = {}
        self.reload_commands()

    @property
    def model_name(self) -> str:
        return active_model_label(self.model_id)

    def events(self) -> list[Event]:
        return self.host.events(self.session_id)

    def workspace(self) -> str:
        return str(self.host.get_session(self.session_id)["workspace"])

    def reload_commands(self) -> None:
        self.custom_commands = load_commands(Path(self.workspace()))

    def send(self, text: str) -> None:
        if self.busy or not text.strip():
            return
        images = self.pending_images
        self.pending_images = []
        self.live_text = ""
        self.live_tool_output = ""
        self._run_bg(
            "waiting for model",
            lambda: self.host.run_turn(
                self.session_id,
                text,
                on_delta=self._append_delta,
                on_tool_output=self._append_tool_output,
                mode="plan" if self.plan_mode else "execute",
                images=images,
            ),
        )

    def _append_delta(self, text: str) -> None:
        self.live_text += text

    def _append_tool_output(self, text: str) -> None:
        self.live_tool_output = (self.live_tool_output + text)[-4000:]

    def _cancel_current(self) -> None:
        try:
            self.host.cancel_turn(self.session_id)
        except Exception as exc:  # noqa: BLE001 — surface the failure in the UI
            self.error = str(exc)
            return
        self.notice = "cancelling"

    def _run_bg(self, phase: str, fn: Any) -> None:
        if self.busy:
            return
        self.busy = True
        self.error = None
        self.host.phase = phase

        def work() -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
            finally:
                self.busy = False
                self.host.phase = ""
                if self.notice == "cancelling":
                    self.notice = None

        threading.Thread(target=work, daemon=True).start()

    def decide(self, decision: str, scope: str = "once") -> None:
        events = self.events()
        pending = pending_call(events)
        if pending is None or self.busy:
            return
        call_id, _name = pending
        self._run_bg(
            "waiting for model",
            lambda: self.host.decide_permission(self.session_id, call_id, decision, scope=scope),
        )

    def slash_items(self) -> list[tuple[str, str]]:
        return filter_slash(
            self.input if self.input.startswith("/") else "/",
            self.custom_commands,
        )

    def model_items(self) -> list[tuple[str, str]]:
        rows = _model_rows()
        needle = self._picker_query("model")
        if needle:
            rows = [row for row in rows if needle in row[0].lower() or needle in row[1].lower()]
        out = []
        current = self.model_id or ""
        label = self.model_name
        for mid, name in rows:
            mark = "*" if mid == current or name == label else " "
            out.append((mid, f"{mark} {name}"))
        return out

    def workspace_items(self) -> list[tuple[str, str]]:
        recent = []
        for session in self.host.list_sessions():
            if session.get("parent_session_id"):
                continue
            path = session.get("workspace")
            if path:
                recent.append(str(path))
        return workspace_candidates(self.workspace(), self._picker_query("workspace"), recent)

    def _picker_query(self, name: str) -> str:
        needle = self.input
        if needle.startswith("/"):
            rest = needle[1:]
            if rest.lower().startswith(name):
                needle = rest[len(name) :].lstrip()
            else:
                needle = ""
        return needle.strip()

    def command(self, raw: str) -> bool:
        cmd = raw.strip().split()[0].lower()
        if cmd in {"/quit", "/exit", "/q"}:
            return False
        if cmd in {"/help", "?"}:
            self.help = not self.help
            self.picker = None
            return True
        if cmd in {"/clear", "/new"}:
            return self._new_session()
        if cmd == "/resume":
            parts = raw.split(maxsplit=1)
            if len(parts) == 1:
                self.picker = "sessions"
                self.input = ""
                self.menu_index = 0
                return True
            return self._resume_session(parts[1].strip())
        if cmd == "/rewind":
            parts = raw.split(maxsplit=1)
            if len(parts) == 1:
                return self._open_rewind()
            try:
                return self._rewind_to(int(parts[1].strip()))
            except ValueError:
                self.error = "rewind needs a user.message seq"
                return True
        if cmd == "/fork":
            new_id = self.host.fork_session(self.session_id)
            self.session_id = new_id
            self.notice = f"forked {new_id[:8]}  (original kept; files unchanged)"
            self.picker = None
            self.input = ""
            return True
        if cmd == "/plan":
            self.plan_mode = not self.plan_mode
            self.notice = "plan mode on (read-only)" if self.plan_mode else "plan mode off"
            self.picker = None
            self.input = ""
            return True
        if cmd == "/yes":
            return self._set_permission("auto")
        if cmd == "/edits":
            return self._set_permission("edits")
        if cmd == "/wait":
            return self._set_permission("ask")
        if cmd == "/readonly":
            return self._set_permission("readonly")
        if cmd == "/permission":
            parts = raw.split(maxsplit=1)
            if len(parts) == 1:
                self.picker = "permission"
                self.input = ""
                self.menu_index = 0
                return True
            return self._set_permission(parts[1].strip())
        if cmd == "/compact":
            self._run_bg("compacting context", lambda: self.host.compact(self.session_id, force=True))
            return True
        if cmd == "/status":
            self.help = True
            self.picker = None
            return True
        if cmd == "/model":
            parts = raw.split(maxsplit=1)
            if len(parts) == 1:
                self.picker = "models"
                self.input = ""
                self.menu_index = 0
                return True
            return self._set_model(parts[1].strip())
        if cmd == "/workspace":
            parts = raw.split(maxsplit=1)
            if len(parts) == 1:
                self.picker = "workspace"
                self.input = ""
                self.menu_index = 0
                return True
            return self._set_workspace(parts[1].strip())
        if cmd == "/image":
            parts = raw.split(maxsplit=1)
            if len(parts) == 1:
                self.error = "usage: /image PATH"
                return True
            try:
                rel = image_rel(Path(self.workspace()), parts[1].strip())
            except Exception as exc:  # noqa: BLE001 — bad paths belong in the composer
                self.error = str(exc)
                return True
            if rel not in self.pending_images:
                self.pending_images.append(rel)
            self.notice = f"attached {rel} ({len(self.pending_images)} queued)"
            self.error = None
            self.input = ""
            return True
        template = self.custom_commands.get(cmd.lstrip("/"))
        if template is not None:
            arguments = raw.strip()[len(cmd) :].strip()
            self.send(expand_command(template, arguments))
            return True
        self.error = f"unknown command {cmd}"
        return True

    def _set_model(self, model_id: str) -> bool:
        try:
            from woke.config import load_config

            self.host.model = build_model(model_id)
            display = model_id
            try:
                _provider, spec = load_config().resolve(model_id)
                self.model_id = spec.id
                display = spec.display_name
                if spec.max_context_size:
                    self.host.token_budget = min(32_000, max(8_000, spec.max_context_size // 8))
            except RuntimeError:
                self.model_id = model_id
            self.notice = f"model -> {display}  (same session, log unchanged)"
            self.error = None
            self.picker = None
            self.input = ""
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
        return True

    def _set_workspace(self, path: str) -> bool:
        target = Path(path).expanduser()
        try:
            target = target.resolve()
        except OSError as exc:
            self.error = str(exc)
            return True
        if not target.is_dir():
            self.error = f"not a directory: {path}"
            return True
        self.session_id = self.host.create_session(str(target))
        self.pending_images = []
        self.reload_commands()
        self.notice = f"new session in {target}"
        self.error = None
        self.picker = None
        self.input = ""
        self.help = False
        return True

    def _new_session(self) -> bool:
        self.session_id = self.host.create_session(self.workspace())
        self.pending_images = []
        self.reload_commands()
        self.notice = "new conversation"
        self.picker = None
        self.input = ""
        self.help = False
        self.error = None
        return True

    def _resume_session(self, token: str) -> bool:
        token = token.strip()
        matches = [
            item
            for item in self.host.list_root_sessions()
            if item["id"] == token or item["id"].startswith(token)
        ]
        if not matches:
            hits = self.host.search_sessions(token)
            if len(hits) == 1:
                matches = hits
            elif hits:
                self.picker = "sessions"
                self.input = f"/resume {token}"
                self.menu_index = 0
                self.error = None
                return True
        if not matches:
            self.error = f"no session matching {token}"
            return True
        item = matches[0]
        self.session_id = item["id"]
        self.pending_images = []
        self.reload_commands()
        self.notice = f"resumed {item['id'][:8]}  {item['title']}"
        self.picker = None
        self.input = ""
        self.help = False
        self.error = None
        return True

    def session_items(self) -> list[tuple[str, str]]:
        needle = self._picker_query("resume").lower()
        rows: list[tuple[str, str]] = []
        listed: set[str] = set()
        home = str(Path.home())
        for item in self.host.list_root_sessions():
            path = item["workspace"]
            if path.startswith(home):
                path = "~" + path[len(home) :]
            preview = item.get("preview") or "(empty)"
            label = f"{item['id'][:8]}  {item['title']}  {path}  {preview}"
            if needle and needle not in label.lower() and needle not in item["id"]:
                continue
            mark = "*" if item["id"] == self.session_id else " "
            rows.append((item["id"], f"{mark} {label}"))
            listed.add(item["id"])
        if needle:
            for hit in self.host.search_sessions(needle):
                if hit["id"] in listed:
                    continue
                mark = "*" if hit["id"] == self.session_id else " "
                label = f"{hit['id'][:8]}  {hit['title']}  {hit['matches']} hits  {hit['snippet']}"
                rows.append((hit["id"], f"{mark} {label}"))
        return rows

    def rewind_items(self) -> list[tuple[str, str]]:
        needle = self._picker_query("rewind").lower()
        rows: list[tuple[str, str]] = []
        for item in self.host.rewind_targets(self.session_id):
            label = f"seq {item['seq']}  {item['text']}"
            if needle and needle not in label.lower():
                continue
            rows.append((str(item["seq"]), label))
        return rows

    def _open_rewind(self) -> bool:
        if not self.host.rewind_targets(self.session_id):
            self.error = "no previous user message to rewind"
            self.picker = None
            return True
        self.picker = "rewind"
        self.input = ""
        self.menu_index = 0
        self.help = False
        self.error = None
        return True

    def _rewind_to(self, user_seq: int) -> bool:
        try:
            new_id, prefill = self.host.fork_before_user_seq(self.session_id, user_seq)
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            return True
        self.session_id = new_id
        self.input = prefill
        self.picker = None
        self.help = False
        self.notice = (
            f"rewound to seq {user_seq} as {new_id[:8]}  "
            "(original kept; workspace files not reverted)"
        )
        self.error = None
        return True

    def _set_permission(self, mode: str) -> bool:
        try:
            policy = GradedPolicy(mode)
        except ValueError as exc:
            self.error = str(exc)
            return True
        self.host.policy = policy
        self.perm_mode = mode
        self.yes = mode == "auto"
        self.picker = None
        self.input = ""
        self.error = None
        return True

    def run(self) -> None:
        try:
            locale.setlocale(locale.LC_ALL, "")
        except locale.Error:
            pass
        curses.wrapper(self._loop)

    def _loop(self, stdscr: curses.window) -> None:
        curses.curs_set(1)
        try:
            curses.use_default_colors()
        except curses.error:
            pass
        _init_colors()
        stdscr.nodelay(True)
        stdscr.keypad(True)
        while True:
            self._draw(stdscr)
            key = _read_key(stdscr)
            if key is None:
                continue
            if not self._handle(key):
                break

    def _handle(self, key: str | int) -> bool:
        if key == curses.KEY_RESIZE:
            return True
        if key in (3,):  # Ctrl+C
            return False
        if key in (27,):  # Esc: close picker; Esc Esc while idle opens rewind
            if self.picker is not None or self.help or self.input:
                self.picker = None
                self.help = False
                self.input = ""
                self.menu_index = 0
                self._esc_armed = False
                return True
            if self.busy:
                self._esc_armed = False
                self._cancel_current()
                return True
            if self._esc_armed:
                self._esc_armed = False
                return self._open_rewind()
            self._esc_armed = True
            self.notice = "Esc again to rewind"
            return True
        if key == 12:
            self.scroll = 0
            return True

        menu = self._menu()
        if menu and key in (curses.KEY_UP, curses.KEY_DOWN):
            if key == curses.KEY_UP:
                self.menu_index = max(0, self.menu_index - 1)
            else:
                self.menu_index = min(len(menu) - 1, self.menu_index + 1)
            return True
        if not menu and key == curses.KEY_UP:
            self.scroll += 1
            return True
        if not menu and key == curses.KEY_DOWN:
            self.scroll = max(0, self.scroll - 1)
            return True

        if key in (curses.KEY_BACKSPACE, 127, 8) or key == "\x7f":
            self.input = self.input[:-1]
            if self.picker == "slash" and not self.input.startswith("/"):
                self.picker = None
            self.menu_index = 0
            return True

        pending = pending_call(self.events())
        if pending and not self.busy and self.input == "" and self.picker is None:
            if key in ("y", "Y"):
                self.decide("allow", scope="once")
                return True
            if key in ("a", "A"):
                self.decide("allow", scope="session")
                return True
            if key in ("n", "N"):
                self.decide("deny")
                return True

        enter = key in (10, 13, curses.KEY_ENTER, "\n", "\r")
        if enter:
            return self._enter()

        if key == 9:  # Tab completes slash / opens nested picker
            items = self._menu()
            if items:
                chosen = items[min(self.menu_index, len(items) - 1)]
                if self.picker in {None, "slash"}:
                    dest = PICKER_PATHS.get(chosen[0])
                    if dest:
                        self.picker = dest
                        self.input = ""
                        self.menu_index = 0
                    else:
                        self.input = chosen[0]
                    return True
            return True

        if isinstance(key, str) and key.isprintable():
            if self.busy:
                return True
            self._esc_armed = False
            if self.notice == "Esc again to rewind":
                self.notice = None
            self.input += key
            if self.input == "/":
                self.picker = "slash"
                self.help = False
                self.menu_index = 0
            elif self.picker == "slash" and not self.input.startswith("/"):
                self.picker = None
            self.menu_index = min(self.menu_index, max(0, len(self._menu()) - 1))
        return True

    def _menu(self) -> list[tuple[str, str]]:
        if self.picker == "models":
            return self.model_items()
        if self.picker == "workspace":
            return self.workspace_items()
        if self.picker == "permission":
            return list(PERMISSION_CHOICES)
        if self.picker == "sessions":
            return self.session_items()
        if self.picker == "rewind":
            return self.rewind_items()
        if self.picker == "slash" or self.input.startswith("/"):
            return self.slash_items()
        return []

    def _enter(self) -> bool:
        items = self._menu()
        if self.picker == "models":
            if items:
                chosen = items[min(self.menu_index, len(items) - 1)]
                return self._set_model(chosen[0])
            if self.input.strip():
                return self._set_model(self.input.strip())
            return True
        if self.picker == "workspace":
            if items:
                chosen = items[min(self.menu_index, len(items) - 1)]
                return self._set_workspace(chosen[0])
            if self.input.strip():
                return self._set_workspace(self.input.strip())
            return True
        if self.picker == "permission":
            if items:
                chosen = items[min(self.menu_index, len(items) - 1)]
                return self._set_permission(chosen[0])
            if self.input.strip():
                return self._set_permission(self.input.strip())
            return True
        if self.picker == "sessions":
            if items:
                chosen = items[min(self.menu_index, len(items) - 1)]
                return self._resume_session(chosen[0])
            if self.input.strip():
                return self._resume_session(self.input.strip())
            return True
        if self.picker == "rewind":
            if items:
                chosen = items[min(self.menu_index, len(items) - 1)]
                return self._rewind_to(int(chosen[0]))
            return True
        if (self.picker == "slash" or self.input.startswith("/")) and items and " " not in self.input.strip():
            chosen = items[min(self.menu_index, len(items) - 1)]
            self.input = ""
            self.picker = None
            return self.command(chosen[0])
        text = self.input
        self.input = ""
        self.picker = None
        if text.startswith("/"):
            return self.command(text)
        if text.strip() == "?":
            self.help = not self.help
            return True
        self.send(text)
        return True

    def _draw(self, stdscr: curses.window) -> None:
        h, w = stdscr.getmaxyx()
        stdscr.erase()
        self.tick += 1
        events = self.events()
        styles = _styles()
        pending = pending_call(events)
        menu = self._menu()
        menu_h = min(8, len(menu)) if menu else 0
        composer_h = 4
        perm_h = 2 if pending else 0
        live_h = 1 if (self.busy or getattr(self.host, "phase", "")) else 0
        trans_h = max(1, h - composer_h - 1 - perm_h - menu_h - live_h)

        user_seen = any(e.kind == "user.message" for e in events)
        if self.help and self.picker is None:
            rows = [("dim", line) for line in HELP.split("\n")]
            info = self.host.get_session(self.session_id)
            mcp = ", ".join(s.name for s in self.host.mcp.servers) or "none"
            rows.append(("dim", f"session {info['id'][:8]}  mcp {mcp}"))
            rows.append(
                ("dim", f"tokens {context_tokens(events)}  budget {self.host.token_budget}")
            )
            if self.host.mcp_errors:
                rows.extend(("err", err) for err in self.host.mcp_errors)
        elif not user_seen and self.picker is None:
            rows = welcome_lines(self.model_name, self.workspace(), min(w, 62))
        else:
            rows = transcript_lines(events, w)

        live = activity_line(events, self.busy, self.tick, getattr(self.host, "phase", ""))
        if self.pending_images:
            rows.append(("dim", "attached " + ", ".join(self.pending_images)))
        if self.notice:
            rows.append(("ok", self.notice))
        if self.error:
            rows.append(("err", self.error))
        if self.busy and self.live_tool_output:
            for line in self.live_tool_output.splitlines()[-8:]:
                rows.append(("dim", line))
        if self.busy and self.live_text:
            rows.append(("default", self.live_text))

        visible = rows
        if len(visible) > trans_h:
            start = max(0, len(visible) - trans_h - self.scroll)
            visible = visible[start : start + trans_h]
        for i, (style, text) in enumerate(visible):
            _add(stdscr, i, 0, text, styles.get(style, 0))

        y = trans_h
        if pending:
            _add(
                stdscr,
                y,
                0,
                f" allow {pending[1]}?  y once / a session / n deny ",
                styles["warn"] | curses.A_BOLD,
            )
            y += 2

        if menu:
            start = 0
            if self.menu_index >= menu_h:
                start = self.menu_index - menu_h + 1
            view = menu[start : start + menu_h]
            for i, (name, hint) in enumerate(view):
                selected = start + i == self.menu_index
                prefix = "▸" if selected else " "
                label = f"{prefix} {name}  {hint}"
                _add(stdscr, y + i, 0, pad_width(label, w - 1), styles["sel"] if selected else styles["dim"])
            y += menu_h

        if live:
            _add(stdscr, y, 0, pad_width(" " + live + " ", w - 1), styles["warn"] | curses.A_BOLD)
            y += 1

        frame = composer_frame(self.input, w, composer_h)
        for i, line in enumerate(frame):
            _add(stdscr, y + i, 0, line, 0)
        _add(
            stdscr,
            h - 1,
            0,
            status_line(
                events,
                self.host.token_budget,
                self.model_name,
                self.busy,
                self.yes,
                self.tick,
                getattr(self.host, "phase", ""),
                self.perm_mode,
                self.plan_mode,
            ),
            styles["dim"],
        )
        inner = max(1, w - 4)
        typed = wrap_text("> " + self.input, inner)
        last = typed[-1] if typed else "> "
        cursor_x = 2 + display_width(last)
        stdscr.move(min(h - 2, y + 1 + min(len(typed) - 1, composer_h - 3)), min(w - 2, cursor_x))
        stdscr.refresh()


def _init_colors() -> None:
    try:
        curses.use_default_colors()
    except curses.error:
        pass
    curses.start_color()
    # Matcha/ink: keep the terminal background, avoid cyan-on-magenta reverse.
    if curses.COLORS >= 256:
        curses.init_pair(1, 180, -1)  # user: warm sand
        curses.init_pair(2, 108, -1)  # ok/brand: matcha
        curses.init_pair(3, 167, -1)  # error: clay red
        curses.init_pair(4, 144, -1)  # dim text
        curses.init_pair(5, 186, -1)  # activity/warn: straw
        curses.init_pair(6, 150, -1)  # selection
    else:
        curses.init_pair(1, curses.COLOR_YELLOW, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_WHITE, -1)
        curses.init_pair(5, curses.COLOR_YELLOW, -1)
        curses.init_pair(6, curses.COLOR_GREEN, -1)


def _styles() -> dict[str, int]:
    return {
        "brand": curses.color_pair(2) | curses.A_BOLD,
        "user": curses.color_pair(1) | curses.A_BOLD,
        "dim": curses.color_pair(4) | curses.A_DIM,
        "ok": curses.color_pair(2),
        "err": curses.color_pair(3) | curses.A_BOLD,
        "warn": curses.color_pair(5),
        "sel": curses.color_pair(6) | curses.A_BOLD,
        "default": curses.A_NORMAL,
    }


def _read_key(stdscr: curses.window):
    """Wait briefly for a key so the spinner can keep turning."""
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0.08)
    except (ValueError, OSError):
        ready = [True]
    if not ready:
        return None
    try:
        key = stdscr.get_wch()
    except curses.error:
        return None
    if key in (-1, None):
        return None
    return key


def _add(stdscr: curses.window, y: int, x: int, text: str, attr: int) -> None:
    h, w = stdscr.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    # Never draw into the bottom-right cell; curses raises there.
    max_w = w - x - (1 if y == h - 1 else 1)
    clipped = slice_width(text.replace("\n", " "), max(0, max_w))
    try:
        stdscr.addnstr(y, x, clipped, len(clipped), attr)
    except curses.error:
        pass


def run_tui(root: Path, workspace: str, yes: bool = False, model_id: str | None = None) -> int:
    from woke.store import init_root

    init_root(root)
    policy = AutoAllow() if yes else WaitUser()
    try:
        model = build_model(model_id)
    except RuntimeError:
        from woke.model import ReactiveModel

        model = ReactiveModel("hello")
    try:
        host: Host | HostClient = Host(root, model=model, policy=policy)
        owned = True
        host.serve_background()
    except HostLocked:
        host = HostClient(root)
        owned = False
    try:
        session_id = host.create_session(workspace)
        Tui(host, session_id, yes=yes, model_id=model_id).run()
    finally:
        if owned:
            host.close()
    return 0
