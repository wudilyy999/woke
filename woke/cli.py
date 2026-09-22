from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from woke import __version__
from woke.errors import HostDown, WokeError
from woke.host import Host
from woke.store import default_root, init_root, read_host_meta, read_root_meta, workspace_state_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="woke",
        description="Event-sourced coding-agent runtime. The log is the only fact.",
    )
    parser.add_argument("--root", type=Path, default=None, help="state root (default: ~/.woke)")
    parser.add_argument("--workspace", type=Path, default=None, help="project directory (default: current directory)")
    parser.add_argument("-y", "--yes", action="store_true", help="auto-allow dangerous tools in the TUI")
    parser.add_argument("--model", default=None, help="model id from ~/.kimi-code/config.toml")
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="cmd")

    host_p = sub.add_parser("host", help="run the single-writer Host process")
    host_sub = host_p.add_subparsers(dest="host_cmd")
    start_p = host_sub.add_parser("start", help="start Host in the foreground")
    start_p.add_argument("--port", type=int, default=0)
    start_p.add_argument("--detach", action="store_true")
    start_p.add_argument("--yes", action="store_true", help="auto-allow dangerous tools")
    start_p.add_argument("--budget", type=int, default=32_000)
    host_sub.add_parser("stop", help="stop a running Host")

    sess = sub.add_parser("session", help="create or list sessions")
    sess_sub = sess.add_subparsers(dest="session_cmd")
    new_p = sess_sub.add_parser("new")
    new_p.add_argument("--workspace", required=True)
    new_p.add_argument("--title", default="")
    sess_sub.add_parser("list")

    send_p = sub.add_parser("send", help="run one turn")
    send_p.add_argument("session")
    send_p.add_argument("text")
    send_p.add_argument("--yes", action="store_true")
    send_p.add_argument("--plan", action="store_true", help="read-only planning turn")

    ev_p = sub.add_parser("events", help="print the session log")
    ev_p.add_argument("session")
    ev_p.add_argument("--after", type=int, default=0)

    compact_p = sub.add_parser("compact", help="force a compaction event")
    compact_p.add_argument("session")

    search_p = sub.add_parser("search", help="search every session transcript")
    search_p.add_argument("query")
    search_p.add_argument("--limit", type=int, default=20)

    ap = sub.add_parser("approve")
    ap.add_argument("session")
    ap.add_argument("call_id")
    dp = sub.add_parser("deny")
    dp.add_argument("session")
    dp.add_argument("call_id")
    cp = sub.add_parser("cancel", help="cancel the open turn in a session")
    cp.add_argument("session")

    sub.add_parser("status")
    sub.add_parser("models", help="list models from Kimi Code / woke config")
    sub.add_parser("tui", help="Codex-style interactive terminal")

    args = parser.parse_args(argv)
    launch_cwd = Path.cwd().resolve()
    if args.version:
        print(__version__)
        return 0
    if args.cmd is None:
        if sys.stdin.isatty() and sys.stdout.isatty():
            args.cmd = "tui"
        else:
            parser.print_help()
            return 1

    workspace_hint = Path(args.workspace).expanduser().resolve() if args.workspace else launch_cwd
    root = (args.root or workspace_state_root(workspace_hint)).expanduser().resolve()
    try:
        if args.cmd == "host":
            return _cmd_host(root, args)
        if args.cmd == "session":
            return _cmd_session(root, args)
        if args.cmd == "send":
            return _cmd_send(root, args)
        if args.cmd == "events":
            return _cmd_events(root, args)
        if args.cmd == "search":
            return _cmd_search(root, args)
        if args.cmd == "compact":
            data = _client(root).post(f"/sessions/{args.session}/compact", {})
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return 0
        if args.cmd == "approve":
            return _cmd_decide(root, args.session, args.call_id, "allow")
        if args.cmd == "deny":
            return _cmd_decide(root, args.session, args.call_id, "deny")
        if args.cmd == "cancel":
            data = _client(root).post(f"/sessions/{args.session}/cancel", {})
            print("cancelled" if data.get("cancelled") else "no open turn")
            return 0
        if args.cmd == "status":
            return _cmd_status(root)
        if args.cmd == "models":
            return _cmd_models()
        if args.cmd == "tui":
            from woke.tui import run_tui

            workspace = workspace_hint
            from woke.store import remember_workspace

            init_root(root)
            remember_workspace(root, workspace)
            return run_tui(
                root,
                str(workspace),
                yes=bool(args.yes),
                model_id=args.model,
            )
    except (WokeError, RuntimeError, urllib.error.URLError, OSError) as exc:
        print(f"woke: {exc}", file=sys.stderr)
        return 1
    return 1


def _cmd_host(root: Path, args: argparse.Namespace) -> int:
    if args.host_cmd == "start":
        if args.detach:
            return _detach(root, args)
        from woke.policy import AutoAllow, WaitUser

        init_root(root)
        policy = AutoAllow() if args.yes else WaitUser()
        host = Host(root, policy=policy, token_budget=args.budget)
        try:
            host.serve(port=args.port)
        except KeyboardInterrupt:
            pass
        finally:
            host.close()
        return 0
    if args.host_cmd == "stop":
        client = _client(root)
        try:
            client.post("/shutdown", {})
        except (HostDown, urllib.error.URLError):
            meta = read_host_meta(root)
            if meta and meta.get("pid"):
                os.kill(int(meta["pid"]), 15)
        return 0
    print("woke host start|stop", file=sys.stderr)
    return 1


def _detach(root: Path, args: argparse.Namespace) -> int:
    init_root(root)
    log = root / "host.log"
    cmd = [
        sys.executable,
        "-m",
        "woke",
        "--root",
        str(root),
        "host",
        "start",
        "--port",
        str(args.port),
        "--budget",
        str(args.budget),
    ]
    if args.yes:
        cmd.append("--yes")
    with log.open("ab") as handle:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
            start_new_session=True,
            env=os.environ.copy(),
        )
    for _ in range(50):
        time.sleep(0.05)
        meta = read_host_meta(root)
        if meta is None:
            continue
        try:
            _client(root).get("/health")
            print(f"woke host pid={meta['pid']} port={meta['port']} root={root}")
            return 0
        except (HostDown, urllib.error.URLError, OSError):
            continue
    print("woke: host did not become ready; see host.log", file=sys.stderr)
    return 1


def _cmd_session(root: Path, args: argparse.Namespace) -> int:
    client = _client(root)
    if args.session_cmd == "new":
        data = client.post("/sessions", {"workspace": args.workspace, "title": args.title})
        print(data["id"])
        return 0
    if args.session_cmd == "list":
        data = client.get("/sessions")
        for session in data.get("sessions") or []:
            print(f"{session['id']}  {session['title']}  {session['workspace']}")
        return 0
    print("woke session new|list", file=sys.stderr)
    return 1


def _cmd_send(root: Path, args: argparse.Namespace) -> int:
    data = _client(root).post(
        f"/sessions/{args.session}/turns",
        {"text": args.text, "yes": bool(args.yes), "mode": "plan" if args.plan else "execute"},
    )
    _print_events(data.get("events") or [])
    status = data.get("status")
    if status == "paused":
        print(
            f"paused: approve with  woke --root {root} approve {args.session} {data.get('pending_call_id')}",
            file=sys.stderr,
        )
        return 2
    return 0 if status == "completed" else 1


def _cmd_events(root: Path, args: argparse.Namespace) -> int:
    data = _client(root).get(f"/sessions/{args.session}/events?after={args.after}")
    _print_events(data.get("events") or [])
    return 0


def _cmd_search(root: Path, args: argparse.Namespace) -> int:
    from urllib.parse import quote

    data = _client(root).get(f"/search?q={quote(args.query)}&limit={args.limit}")
    sessions = data.get("sessions") or []
    if not sessions:
        print(f"no session mentions {args.query!r}", file=sys.stderr)
        return 1
    for item in sessions:
        print(f"{item['id']}  {item['title']}  {item['workspace']}  {item['matches']} hits")
        print(f"      {item['snippet']}")
    return 0


def _cmd_decide(root: Path, session: str, call_id: str, decision: str) -> int:
    data = _client(root).post(
        f"/sessions/{session}/permissions",
        {"id": call_id, "decision": decision},
    )
    _print_events(data.get("events") or [])
    return 0 if data.get("status") == "completed" else 1


def _cmd_models() -> int:
    from woke.config import load_config

    cfg = load_config()
    if cfg.source is None:
        print("no config (expected ~/.woke/config.toml or ~/.kimi-code/config.toml)", file=sys.stderr)
        return 1
    print(f"source   {cfg.source}")
    print(f"default  {cfg.default_model}")
    for spec in cfg.models.values():
        mark = "*" if spec.id == cfg.default_model else " "
        print(f"{mark} {spec.id:40} {spec.display_name}")
    return 0


def _cmd_status(root: Path) -> int:
    init_root(root)
    meta = read_root_meta(root)
    host = read_host_meta(root)
    print(f"root     {root}")
    print(f"root_id  {meta['root_id']}")
    if host is None:
        print("host     down")
        return 0
    print(f"host     pid={host['pid']} port={host['port']}")
    return 0


def _print_events(events: list[dict[str, Any]]) -> None:
    for event in events:
        payload = json.dumps(event.get("payload") or {}, ensure_ascii=False)
        if len(payload) > 240:
            payload = payload[:240] + "…"
        print(f"{event['seq']:>5}  {event['kind']:<22}  {payload}")


class _Client:
    def __init__(self, root: Path) -> None:
        self.root = root
        host = read_host_meta(root)
        if host is None:
            raise HostDown(f"no host is running in {root}; start it with: woke --root {root} host start")
        self.base = f"http://127.0.0.1:{host['port']}"
        self.token = read_root_meta(root)["token"]

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path, None)

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, body)

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={
                "X-Woke-Token": self.token,
                "Content-Type": "application/json",
                "Connection": "close",
            },
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _client(root: Path) -> _Client:
    return _Client(root)
