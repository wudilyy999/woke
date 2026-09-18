from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from woke.errors import SimulatedCrash
from woke.host import Host
from woke.model import ReactiveModel
from woke.policy import AutoAllow
from woke.store import Store

REPO = Path(__file__).resolve().parents[1]


class RecoveryTests(unittest.TestCase):
    def test_in_process_crash_does_not_rerun_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            host = Host(
                root,
                model=ReactiveModel("write_then_done"),
                policy=AutoAllow(),
                crash_after_tool_call=True,
            )
            sid = host.create_session(str(ws))
            with self.assertRaises(SimulatedCrash):
                host.run_turn(sid, "write a file")
            host.close()
            self.assertFalse((ws / "out.txt").exists())

            with Host(root, model=ReactiveModel("write_then_done"), policy=AutoAllow()) as host2:
                events = host2.events(sid)
            kinds = [e.kind for e in events]
            self.assertIn("tool.call", kinds)
            results = [e for e in events if e.kind == "tool.result"]
            self.assertTrue(results)
            self.assertEqual(results[0].payload["error"], "interrupted_by_crash")
            self.assertFalse((ws / "out.txt").exists())
            self.assertIn("turn.terminated", kinds)
            recoveries = [e for e in events if e.kind == "run.started" and e.payload.get("reason") == "recovery"]
            self.assertTrue(recoveries)

    def test_subprocess_kill_after_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO)
            env["WOKE_FAKE_MODEL"] = "write_then_done"
            env["WOKE_CRASH_AFTER_TOOL_CALL"] = "1"
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "woke",
                    "--root",
                    str(root),
                    "host",
                    "start",
                    "--yes",
                    "--port",
                    "0",
                ],
                cwd=str(REPO),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                _wait_host(root)
                send = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "woke",
                        "--root",
                        str(root),
                        "send",
                        _session(root, ws),
                        "write a file",
                        "--yes",
                    ],
                    cwd=str(REPO),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                proc.wait(timeout=10)
                self.assertNotEqual(proc.returncode, 0)
                self.assertFalse((ws / "out.txt").exists())
                leftover = root / "host.json"
                if leftover.exists():
                    leftover.unlink()
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=3)

            env.pop("WOKE_CRASH_AFTER_TOOL_CALL", None)
            ready = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "woke",
                    "--root",
                    str(root),
                    "host",
                    "start",
                    "--yes",
                    "--port",
                    "0",
                ],
                cwd=str(REPO),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                _wait_host(root)
                time.sleep(0.2)
                events = _events(root)
                results = [e for e in events if e["kind"] == "tool.result"]
                self.assertTrue(results)
                self.assertEqual(results[0]["payload"]["error"], "interrupted_by_crash")
                self.assertFalse((ws / "out.txt").exists())
                kinds = [e["kind"] for e in events]
                self.assertIn("turn.terminated", kinds)
            finally:
                subprocess.run(
                    [sys.executable, "-m", "woke", "--root", str(root), "host", "stop"],
                    cwd=str(REPO),
                    env=env,
                    capture_output=True,
                )
                ready.wait(timeout=5)


def _wait_host(root: Path, tries: int = 50) -> None:
    from woke.store import read_host_meta

    for _ in range(tries):
        meta = read_host_meta(root)
        if meta is not None:
            return
        time.sleep(0.05)
    raise AssertionError("host did not start")


def _session(root: Path, ws: Path) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["WOKE_FAKE_MODEL"] = "write_then_done"
    created = subprocess.run(
        [
            sys.executable,
            "-m",
            "woke",
            "--root",
            str(root),
            "session",
            "new",
            "--workspace",
            str(ws),
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return created.stdout.strip()


def _events(root: Path) -> list[dict]:
    store = Store(root)
    try:
        events = store.read_all()
        return [e.to_dict() for e in events]
    finally:
        store.close()
