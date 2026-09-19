from __future__ import annotations

import json
import os
import select
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from woke.sandbox import SandboxManager


class McpError(RuntimeError):
    pass


@dataclass
class McpTool:
    server: str
    name: str
    description: str
    schema: dict[str, Any]
    read_only: bool

    @property
    def qualified(self) -> str:
        return f"mcp__{self.server}__{self.name}"


class McpServer:
    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        sandbox: SandboxManager | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self.sandbox = sandbox
        self.workspace = workspace
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._next_id = 1
        self.tools: list[McpTool] = []

    def start(self) -> None:
        env = os.environ.copy()
        env.update(self.env)
        argv = [self.command, *self.args]
        if self.sandbox is not None and self.workspace is not None:
            extra = []
            for arg in self.args:
                candidate = Path(arg)
                if candidate.exists():
                    extra.append(candidate.parent)
            argv = self.sandbox.wrap_argv(
                argv,
                self.workspace,
                allow_write=True,
                extra_read_roots=extra or None,
            )
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        result = self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "woke", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized", {})
        listed = self._request("tools/list", {})
        for raw in listed.get("tools") or []:
            annotations = raw.get("annotations") or {}
            schema = raw.get("inputSchema") or {"type": "object", "properties": {}}
            self.tools.append(
                McpTool(
                    server=self.name,
                    name=str(raw.get("name") or ""),
                    description=str(raw.get("description") or ""),
                    schema=schema if isinstance(schema, dict) else {"type": "object"},
                    read_only=bool(annotations.get("readOnlyHint")),
                )
            )
        _ = result

    def call(self, tool: str, arguments: dict[str, Any]) -> str:
        result = self._request("tools/call", {"name": tool, "arguments": arguments})
        if result.get("isError"):
            raise McpError(_content_text(result.get("content")))
        return _content_text(result.get("content"))

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1)
        for stream in (proc.stdout, proc.stderr):
            if stream and stream is not subprocess.DEVNULL:
                try:
                    stream.close()
                except OSError:
                    pass

    def _request(self, method: str, params: dict[str, Any], timeout: float = 15.0) -> dict[str, Any]:
        with self._lock:
            assert self._proc and self._proc.stdin and self._proc.stdout
            msg_id = self._next_id
            self._next_id += 1
            _write(self._proc.stdin, {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
            while True:
                message = _read(self._proc.stdout, timeout=timeout)
                if message is None:
                    raise McpError(f"{self.name} closed during {method}")
                if message.get("id") != msg_id:
                    continue
                if message.get("error"):
                    raise McpError(str(message["error"]))
                result = message.get("result")
                if not isinstance(result, dict):
                    raise McpError(f"{self.name} {method} returned no object")
                return result

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        with self._lock:
            assert self._proc and self._proc.stdin
            _write(self._proc.stdin, {"jsonrpc": "2.0", "method": method, "params": params})


class McpHub:
    def __init__(self, servers: list[McpServer]) -> None:
        self.servers = servers

    @classmethod
    def load(cls, root: Path, sandbox: SandboxManager | None = None, workspace: Path | None = None) -> McpHub:
        path = Path(root) / "mcp.json"
        if not path.exists():
            return cls([])
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls([])
        raw = data.get("mcpServers") or {}
        servers = []
        if isinstance(raw, dict):
            for name, spec in raw.items():
                if not isinstance(spec, dict) or not spec.get("command"):
                    continue
                args = spec.get("args") or []
                env = spec.get("env") or {}
                servers.append(
                    McpServer(
                        name=str(name),
                        command=str(spec["command"]),
                        args=[str(a) for a in args],
                        env={str(k): str(v) for k, v in dict(env).items()},
                        sandbox=sandbox,
                        workspace=workspace,
                    )
                )
        return cls(servers)

    def start(self) -> list[str]:
        errors = []
        alive = []
        for server in self.servers:
            try:
                server.start()
                alive.append(server)
            except (McpError, OSError, subprocess.SubprocessError) as exc:
                errors.append(f"{server.name}: {exc}")
                server.close()
        self.servers = alive
        return errors

    def close(self) -> None:
        for server in self.servers:
            server.close()
        self.servers = []

    def tools(self) -> list[McpTool]:
        out: list[McpTool] = []
        for server in self.servers:
            out.extend(server.tools)
        return out

    def call(self, qualified: str, arguments: dict[str, Any]) -> str:
        for tool in self.tools():
            if tool.qualified == qualified:
                return self._server(tool.server).call(tool.name, arguments)
        raise McpError(f"unknown MCP tool: {qualified}")

    def _server(self, name: str) -> McpServer:
        for server in self.servers:
            if server.name == name:
                return server
        raise McpError(f"unknown MCP server: {name}")


def _write(fp: Any, message: dict[str, Any]) -> None:
    raw = json.dumps(message, ensure_ascii=False).encode("utf-8")
    fp.write(f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii") + raw)
    fp.flush()


def _read(fp: Any, timeout: float) -> dict[str, Any] | None:
    ready, _, _ = select.select([fp], [], [], timeout)
    if not ready:
        raise McpError("MCP timed out")
    headers: dict[str, str] = {}
    while True:
        line = fp.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("utf-8", errors="replace").strip()
        if ":" not in decoded:
            continue
        key, value = decoded.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length") or "0")
    body = b""
    while len(body) < length:
        chunk = fp.read(length - len(body))
        if not chunk:
            break
        body += chunk
    if not body:
        return None
    data = json.loads(body.decode("utf-8"))
    return data if isinstance(data, dict) else None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
        else:
            parts.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(parts)
