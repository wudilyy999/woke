from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from woke.host import Host
from woke.mcp import McpHub
from woke.model import ModelReply, ScriptModel, ToolCall
from woke.policy import AutoAllow

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
