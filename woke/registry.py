from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from woke.mcp import McpHub, McpServer, McpTool
from woke.sandbox import SandboxManager
from woke.tools import DANGEROUS, TOOL_SPECS, execute as execute_builtin

SPAWN_NAME = "spawn_agent"
TODO_NAME = "todo_write"
RESOURCE_TOOL = "read_resource"
MAX_SPAWN_DEPTH = 1

SPAWN_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SPAWN_NAME,
        "description": (
            "Run a nested coding agent on a focused subtask. "
            "The child has its own session log in the same workspace and returns a digest. "
            "Use for independent slices, not for simple file edits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The subtask to complete."},
                "label": {"type": "string", "description": "Short name for the child session."},
            },
            "required": ["task"],
        },
    },
}

TODO_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TODO_NAME,
        "description": (
            "Record the task list for this turn so progress stays visible. "
            "Send the full list on every call, with one item in_progress at most."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["text", "status"],
                    },
                }
            },
            "required": ["items"],
        },
    },
}


@dataclass
class BoundTool:
    name: str
    spec: dict[str, Any]
    dangerous: bool
    kind: str  # builtin | mcp | spawn


class ToolRegistry:
    def __init__(
        self,
        hub: McpHub | None = None,
        allow_spawn: bool = True,
        sandbox: SandboxManager | None = None,
    ) -> None:
        self.hub = hub or McpHub([])
        self.allow_spawn = allow_spawn
        self.sandbox = sandbox
        self._spawn: Callable[..., tuple[bool, str]] | None = None

    def set_spawn(self, spawn: Callable[..., tuple[bool, str]] | None) -> None:
        self._spawn = spawn

    def tools(self) -> list[BoundTool]:
        out = [
            BoundTool(name=spec["function"]["name"], spec=spec, dangerous=spec["function"]["name"] in DANGEROUS, kind="builtin")
            for spec in TOOL_SPECS
        ]
        for tool in self.hub.tools():
            out.append(_mcp_bound(tool))
        for server in self.hub.servers:
            if not server.resources:
                continue
            if any(tool.server == server.name and tool.name == RESOURCE_TOOL for tool in server.tools):
                continue
            out.append(_resource_bound(server))
        out.append(BoundTool(name=TODO_NAME, spec=TODO_SPEC, dangerous=False, kind="builtin"))
        if self.allow_spawn:
            out.append(BoundTool(name=SPAWN_NAME, spec=SPAWN_SPEC, dangerous=True, kind="spawn"))
        return out

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec for tool in self.tools()]

    def is_dangerous(self, name: str) -> bool:
        for tool in self.tools():
            if tool.name == name:
                return tool.dangerous
        return True

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        workspace: Path,
        on_output: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[bool, str]:
        if name == TODO_NAME:
            return False, "todo_write is handled by the turn runner"
        if name == SPAWN_NAME:
            if self._spawn is None:
                return False, "spawn_agent is not available"
            try:
                return self._spawn(arguments, workspace)
            except Exception as exc:  # noqa: BLE001 — surface any child failure as a tool error
                return False, str(exc)
        if name.startswith("mcp__"):
            if self.kind_of(name) == "mcp_resource":
                server = name[len("mcp__") : -len(f"__{RESOURCE_TOOL}")]
                try:
                    return True, self.hub.read_resource(server, str(arguments.get("uri") or ""))
                except Exception as exc:  # noqa: BLE001
                    return False, str(exc)
            try:
                return True, self.hub.call(name, arguments)
            except Exception as exc:  # noqa: BLE001
                return False, str(exc)
        return execute_builtin(
            name,
            arguments,
            workspace,
            sandbox=self.sandbox,
            on_output=on_output,
            should_cancel=should_cancel,
        )

    def kind_of(self, name: str) -> str | None:
        for tool in self.tools():
            if tool.name == name:
                return tool.kind
        return None

    def child(self) -> ToolRegistry:
        child = ToolRegistry(hub=self.hub, allow_spawn=False, sandbox=self.sandbox)
        return child


def _mcp_bound(tool: McpTool) -> BoundTool:
    spec = {
        "type": "function",
        "function": {
            "name": tool.qualified,
            "description": f"[MCP {tool.server}] {tool.description}",
            "parameters": tool.schema,
        },
    }
    return BoundTool(name=tool.qualified, spec=spec, dangerous=not tool.read_only, kind="mcp")


def _resource_bound(server: McpServer) -> BoundTool:
    listing = "\n".join(f"- {item.uri} ({item.name})" for item in server.resources[:20])
    name = f"mcp__{server.name}__{RESOURCE_TOOL}"
    spec = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"[MCP {server.name}] Read one of this server's resources by uri.\n{listing}",
            "parameters": {
                "type": "object",
                "properties": {"uri": {"type": "string"}},
                "required": ["uri"],
            },
        },
    }
    return BoundTool(name=name, spec=spec, dangerous=False, kind="mcp_resource")
