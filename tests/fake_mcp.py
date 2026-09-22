"""Minimal MCP stdio server for tests: one echo tool."""

from __future__ import annotations

import json
import sys


def _write(message: dict) -> None:
    raw = json.dumps(message).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii") + raw)
    sys.stdout.buffer.flush()


def _read() -> dict | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("utf-8").strip()
        if ":" not in decoded:
            continue
        key, value = decoded.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length") or "0")
    body = sys.stdin.buffer.read(length)
    data = json.loads(body.decode("utf-8"))
    return data if isinstance(data, dict) else None


def main() -> None:
    while True:
        message = _read()
        if message is None:
            return
        method = message.get("method")
        msg_id = message.get("id")
        if method == "initialize":
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                        "serverInfo": {"name": "fake", "version": "0"},
                    },
                }
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "echo text",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"text": {"type": "string"}},
                                    "required": ["text"],
                                },
                                "annotations": {"readOnlyHint": True},
                            }
                        ]
                    },
                }
            )
        elif method == "tools/call":
            params = message.get("params") or {}
            arguments = params.get("arguments") or {}
            text = str(arguments.get("text") or "")
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": f"echo:{text}"}]},
                }
            )
        elif method == "resources/list":
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "resources": [
                            {
                                "uri": "fake://notes",
                                "name": "notes",
                                "description": "server notes",
                                "mimeType": "text/plain",
                            }
                        ]
                    },
                }
            )
        elif method == "resources/read":
            params = message.get("params") or {}
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "contents": [
                            {"uri": params.get("uri"), "mimeType": "text/plain", "text": "notes body"}
                        ]
                    },
                }
            )
        elif method == "prompts/list":
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "prompts": [
                            {
                                "name": "review",
                                "description": "review a target",
                                "arguments": [{"name": "target", "required": True}],
                            }
                        ]
                    },
                }
            )
        elif method == "prompts/get":
            params = message.get("params") or {}
            arguments = params.get("arguments") or {}
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "messages": [
                            {
                                "role": "user",
                                "content": {
                                    "type": "text",
                                    "text": f"review {arguments.get('target')}",
                                },
                            }
                        ]
                    },
                }
            )


if __name__ == "__main__":
    main()
