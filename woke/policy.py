from __future__ import annotations

from typing import Any, Protocol

# readonly < ask < edits < auto
MODES = ("readonly", "ask", "edits", "auto")

# File mutations auto-allowed in "edits". Shell/spawn/MCP still ask.
EDIT_TOOLS = frozenset({"write_file", "str_replace"})


class Policy(Protocol):
    mode: str

    def decide(
        self,
        name: str,
        arguments: dict[str, Any],
        grants: frozenset[str] = frozenset(),
    ) -> str:
        """Return allow, deny, or wait."""


class GradedPolicy:
    """Four levels: readonly, ask, edits (files only), auto (everything)."""

    def __init__(self, mode: str = "ask") -> None:
        if mode not in MODES:
            raise ValueError(f"unknown permission mode: {mode}")
        self.mode = mode

    def decide(
        self,
        name: str,
        arguments: dict[str, Any],
        grants: frozenset[str] = frozenset(),
    ) -> str:
        if self.mode == "readonly":
            return "deny"
        if name in grants:
            return "allow"
        if self.mode == "auto":
            return "allow"
        if self.mode == "edits" and name in EDIT_TOOLS:
            return "allow"
        return "wait"


class AutoAllow(GradedPolicy):
    def __init__(self) -> None:
        super().__init__("auto")


class AutoDeny(GradedPolicy):
    """Readonly: never run a dangerous tool."""

    def __init__(self) -> None:
        super().__init__("readonly")


class WaitUser(GradedPolicy):
    def __init__(self) -> None:
        super().__init__("ask")


def parse_policy(yes: bool) -> GradedPolicy:
    return AutoAllow() if yes else WaitUser()


def parse_mode(mode: str) -> GradedPolicy:
    return GradedPolicy(mode)
