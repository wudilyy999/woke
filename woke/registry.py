from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from woke.mcp import McpHub, McpTool
from woke.sandbox import SandboxManager
from woke.tools import DANGEROUS, TOOL_SPECS, execute as execute_builtin

SPAWN_NAME = "spawn_agent"
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

    def execute(self, name: str, arguments: dict[str, Any], workspace: Path) -> tuple[bool, str]:
        if name == SPAWN_NAME:
            if self._spawn is None:
                return False, "spawn_agent is not available"
            try:
                return self._spawn(arguments, workspace)
            except Exception as exc:  # noqa: BLE001 — surface any child failure as a tool error
                return False, str(exc)
        if name.startswith("mcp__"):
            try:
                return True, self.hub.call(name, arguments)
            except Exception as exc:  # noqa: BLE001
                return False, str(exc)
        return execute_builtin(name, arguments, workspace, sandbox=self.sandbox)

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
