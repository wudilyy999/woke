from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from woke.host import Host, HostClient
from woke.mcp import McpHub
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoAllow
from woke.tui import Tui, prompt_arguments

FAKE = Path(__file__).resolve().parent / "fake_mcp.py"


class McpTests(unittest.TestCase):
    def test_echo_tool_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mcp.json").write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "fake": {
                                "command": sys.executable,
                                "args": [str(FAKE)],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            hub = McpHub.load(root)
            errors = hub.start()
            self.assertEqual(errors, [])
            tools = hub.tools()
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].qualified, "mcp__fake__echo")
            self.assertEqual(hub.call("mcp__fake__echo", {"text": "hi"}), "echo:hi")
            hub.close()

    def test_host_turn_calls_mcp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            (root / "mcp.json").write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "fake": {
                                "command": sys.executable,
                                "args": [str(FAKE)],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(
                        text="",
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="mcp__fake__echo",
                                arguments={"text": "ping"},
                            )
                        ],
                    ),
                    ModelReply(text="heard ping"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                self.assertEqual(host.mcp_errors, [])
                sid = host.create_session(str(ws))
                outcome = host.run_turn(sid, "echo via mcp")
                self.assertEqual(outcome.status, "completed")
                results = [e for e in host.events(sid) if e.kind == "tool.result"]
                self.assertTrue(results[0].payload["ok"])
                self.assertEqual(results[0].payload["output"], "echo:ping")

    def test_resources_and_prompts_over_stdio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_config(root, {"fake": {"command": sys.executable, "args": [str(FAKE)]}})
            hub = McpHub.load(root)
            self.assertEqual(hub.start(), [])
            self.assertEqual([item.uri for item in hub.resources()], ["fake://notes"])
            self.assertEqual([item.name for item in hub.prompts()], ["review"])
            self.assertEqual(hub.read_resource("fake", "fake://notes"), "notes body")
            self.assertEqual(hub.prompt_text("fake", "review", {"target": "app.py"}), "review app.py")
            hub.close()

    def test_host_uses_resource_tool_and_prompt_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            _write_config(root, {"fake": {"command": sys.executable, "args": [str(FAKE)]}})
            ws = Path(tmp) / "ws"
            ws.mkdir()
            model = ScriptModel(
                [
                    ModelReply(
                        text="",
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="mcp__fake__read_resource",
                                arguments={"uri": "fake://notes"},
                            )
                        ],
                    ),
                    ModelReply(text="read the notes"),
                    ModelReply(text="reviewed"),
                ]
            )
            with Host(root, model=model, policy=AutoAllow()) as host:
                sid = host.create_session(str(ws), title="mcp")
                tools = {tool.name: tool for tool in host.registry.tools()}
                self.assertIn("mcp__fake__read_resource", tools)
                self.assertFalse(tools["mcp__fake__read_resource"].dangerous)
                outcome = host.run_turn(sid, "read the resource")
                self.assertEqual(outcome.status, "completed")
                result = [e.payload for e in outcome.events if e.kind == "tool.result"][0]
                self.assertEqual(result["output"], "notes body")
                self.assertEqual([item["name"] for item in host.mcp_prompts()], ["review"])

                tui = Tui(host, sid)
                self.assertIn("fake:review", tui.mcp_prompt_commands)
                self.assertIn(("/fake:review", "review a target"), tui.slash_items())
                self.assertTrue(tui.command("/fake:review target=app.py"))
                for _ in range(100):
                    if any(
                        event.kind == "user.message" and event.payload["text"] == "review app.py"
                        for event in host.events(sid)
                    ):
                        break
                    time.sleep(0.05)
                texts = [
                    event.payload["text"]
                    for event in host.events(sid)
                    if event.kind == "user.message"
                ]
                self.assertIn("review app.py", texts)

    def test_prompt_arguments(self) -> None:
        self.assertEqual(prompt_arguments(["target"], "app.py"), {"target": "app.py"})
        self.assertEqual(prompt_arguments(["a", "b"], "a=1 b=2"), {"a": "1", "b": "2"})
        with self.assertRaises(Exception):
            prompt_arguments(["a", "b"], "loose text")

    def test_mcp_lists_over_host_http(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            _write_config(root, {"fake": {"command": sys.executable, "args": [str(FAKE)]}})
            ws = Path(tmp) / "ws"
            ws.mkdir()
            with Host(root, model=ScriptModel([ModelReply(text="ok")]), policy=AutoAllow()) as host:
                host.serve_background()
                client = HostClient(root)
                self.assertEqual(client.mcp_server_names(), ["fake"])
                self.assertEqual([item["name"] for item in client.mcp_prompts()], ["review"])
                self.assertEqual(
                    client.mcp_prompt("fake", "review", {"target": "app.py"}), "review app.py"
                )

    def test_streamable_http_transport(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _HttpMcpHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                port = server.server_address[1]
                _write_config(root, {"remote": {"url": f"http://127.0.0.1:{port}/mcp"}})
                hub = McpHub.load(root)
                self.assertEqual(hub.start(), [])
                self.assertEqual([item.qualified for item in hub.tools()], ["mcp__remote__echo"])
                self.assertEqual(hub.call("mcp__remote__echo", {"text": "hi"}), "echo:hi")
                self.assertEqual([item.uri for item in hub.resources()], ["fake://notes"])
                hub.close()
        finally:
            server.shutdown()
            server.server_close()


def _write_config(root: Path, servers: dict) -> None:
    (root / "mcp.json").write_text(
        json.dumps({"mcpServers": servers}), encoding="utf-8"
    )


class _HttpMcpHandler(BaseHTTPRequestHandler):
    """Streamable HTTP server: JSON replies, SSE framing for tool calls."""

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        message = json.loads(self.rfile.read(length).decode("utf-8"))
        if "id" not in message:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        result = _http_result(str(message.get("method")), message.get("params") or {})
        body = json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}).encode("utf-8")
        if message.get("method") == "tools/call":
            payload = b"event: message\ndata: " + body + b"\n\n"
            content_type = "text/event-stream"
        else:
            payload = body
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _http_result(method: str, params: dict) -> dict:
    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}, "resources": {}},
            "serverInfo": {"name": "remote", "version": "0"},
        }
    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "echo",
                    "description": "echo text",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                }
            ]
        }
    if method == "resources/list":
        return {"resources": [{"uri": "fake://notes", "name": "notes"}]}
    if method == "tools/call":
        arguments = params.get("arguments") or {}
        return {"content": [{"type": "text", "text": f"echo:{arguments.get('text')}"}]}
    raise AssertionError(f"unexpected method {method}")
