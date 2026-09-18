from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from woke.host import Host
from woke.model import ModelReply, ScriptModel
from woke.policy import AutoAllow
from woke.store import read_root_meta


class HttpTests(unittest.TestCase):
    def test_auth_and_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel([ModelReply(text="hello")])
            with Host(root, model=model, policy=AutoAllow()) as host:
                port = host.serve_background()
                token = read_root_meta(root)["token"]
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    _json("GET", port, "/health", token="wrong")
                self.assertEqual(ctx.exception.code, 401)

                health = _json("GET", port, "/health", token=token)
                self.assertTrue(health["ok"])

                created = _json(
                    "POST",
                    port,
                    "/sessions",
                    token=token,
                    body={"workspace": str(ws), "title": "t"},
                )
                sid = created["id"]
                outcome = _json(
                    "POST",
                    port,
                    f"/sessions/{sid}/turns",
                    token=token,
                    body={"text": "hi", "yes": True},
                )
                self.assertEqual(outcome["status"], "completed")
                kinds = [e["kind"] for e in outcome["events"]]
                self.assertEqual(kinds[0], "turn.started")
                self.assertEqual(kinds[-1], "turn.terminated")


def _json(method: str, port: int, path: str, token: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method=method,
        headers={
            "X-Woke-Token": token,
            "Content-Type": "application/json",
            "Connection": "close",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}
    except urllib.error.HTTPError:
        raise
