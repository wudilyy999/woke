from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from woke.errors import HostLocked, ValidationError
from woke.events import Event, validate_event

ROOT_JSON = "root.json"
HOST_JSON = "host.json"
LOCK_FILE = "woke.lock"
DB_FILE = "woke.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  session_id TEXT NOT NULL,
  turn_id TEXT,
  run_id TEXT,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_session_seq ON events(session_id, seq);
"""

SEARCH_KINDS = ("user.message", "model.message")


def like_pattern(needle: str) -> str:
    escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def snippet(text: str, needle: str, width: int = 60) -> str:
    flat = " ".join(text.split())
    index = flat.lower().find(needle.lower())
    if index < 0:
        return flat[: width * 2]
    start = max(0, index - width)
    end = min(len(flat), index + len(needle) + width)
    return f"{'...' if start else ''}{flat[start:end]}{'...' if end < len(flat) else ''}"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def default_root() -> Path:
    return Path.home() / ".woke"


def workspace_state_root(workspace: Path) -> Path:
    """Per-project log dir so two `woke` in different folders do not share a lock."""
    resolved = Path(workspace).expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    return Path.home() / ".woke" / "ws" / digest


def init_root(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / ROOT_JSON
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    meta = {
        "root_id": str(uuid.uuid4()),
        "token": uuid.uuid4().hex + uuid.uuid4().hex,
        "created_at": utc_now(),
    }
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def remember_workspace(root: Path, workspace: Path) -> None:
    (root / "workspace").write_text(str(Path(workspace).resolve()) + "\n", encoding="utf-8")


def read_root_meta(root: Path) -> dict[str, Any]:
    path = root / ROOT_JSON
    if not path.exists():
        raise ValidationError(f"state root not initialized: {root}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_host_meta(root: Path, pid: int, port: int) -> None:
    (root / HOST_JSON).write_text(
        json.dumps({"pid": pid, "port": port, "started_at": utc_now()}, indent=2) + "\n",
        encoding="utf-8",
    )


def read_host_meta(root: Path) -> dict[str, Any] | None:
    path = root / HOST_JSON
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def clear_host_meta(root: Path) -> None:
    path = root / HOST_JSON
    if path.exists():
        path.unlink()


def acquire_lock(root: Path) -> int:
    """Exclusive process lock. Returns a file descriptor that must stay open."""
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        raise HostLocked(
            f"another woke is already running for this workspace (lock {root / LOCK_FILE})"
        ) from exc
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def release_lock(fd: int) -> None:
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class Store:
    """Append-only event log. The committed prefix is the only fact."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        init_root(self.root)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(
            self.root / DB_FILE,
            check_same_thread=False,
            isolation_level=None,
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def append(
        self,
        session_id: str,
        kind: str,
        payload: dict[str, Any],
        turn_id: str | None = None,
        run_id: str | None = None,
    ) -> Event:
        validate_event(kind, payload, turn_id, run_id)
        ts = utc_now()
        blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO events (ts, session_id, turn_id, run_id, kind, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts, session_id, turn_id, run_id, kind, blob),
            )
        return Event(
            seq=int(cur.lastrowid),
            ts=ts,
            session_id=session_id,
            turn_id=turn_id,
            run_id=run_id,
            kind=kind,
            payload=payload,
        )

    def read_session(self, session_id: str, after: int = 0) -> list[Event]:
        with self._lock:
            rows = self._db.execute(
                "SELECT seq, ts, session_id, turn_id, run_id, kind, payload "
                "FROM events WHERE session_id = ? AND seq > ? ORDER BY seq",
                (session_id, after),
            ).fetchall()
        return [self._row(r) for r in rows]

    def read_all(self) -> list[Event]:
        with self._lock:
            rows = self._db.execute(
                "SELECT seq, ts, session_id, turn_id, run_id, kind, payload "
                "FROM events ORDER BY seq"
            ).fetchall()
        return [self._row(r) for r in rows]

    def list_sessions(self) -> list[Event]:
        with self._lock:
            rows = self._db.execute(
                "SELECT seq, ts, session_id, turn_id, run_id, kind, payload "
                "FROM events WHERE kind = 'session.created' ORDER BY seq"
            ).fetchall()
        return [self._row(r) for r in rows]

    def session_ids(self) -> list[str]:
        return [e.session_id for e in self.list_sessions()]

    def search(self, text: str, limit: int = 20) -> list[dict[str, Any]]:
        """Transcript search across sessions, newest hit first."""
        needle = text.strip()
        if not needle:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT seq, session_id, payload FROM events "
                "WHERE kind IN (?, ?) AND payload LIKE ? ESCAPE '\\' ORDER BY seq DESC",
                (*SEARCH_KINDS, like_pattern(needle)),
            ).fetchall()
        hits: dict[str, dict[str, Any]] = {}
        for seq, session_id, payload in rows:
            body = str(json.loads(payload).get("text") or "")
            if needle.lower() not in body.lower():
                continue
            hit = hits.setdefault(
                session_id,
                {
                    "session_id": session_id,
                    "seq": int(seq),
                    "matches": 0,
                    "snippet": snippet(body, needle),
                },
            )
            hit["matches"] += 1
        return sorted(hits.values(), key=lambda item: item["seq"], reverse=True)[:limit]

    @staticmethod
    def _row(row: tuple[Any, ...]) -> Event:
        seq, ts, session_id, turn_id, run_id, kind, payload = row
        return Event(
            seq=seq,
            ts=ts,
            session_id=session_id,
            turn_id=turn_id,
            run_id=run_id,
            kind=kind,
            payload=json.loads(payload),
        )
