from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from woke.errors import ValidationError

# readonly < ask < edits < auto
MODES = ("readonly", "ask", "edits", "auto")

# File mutations auto-allowed in "edits". Shell/spawn/MCP still ask.
EDIT_TOOLS = frozenset({"write_file", "str_replace"})

RULES_PATH = ".woke/permissions.json"

# Argument a rule pattern is matched against, per tool.
PRIMARY_ARG = {
    "run_shell": "command",
    "write_file": "path",
    "str_replace": "path",
    "web_fetch": "url",
    "web_search": "query",
}


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


@dataclass(frozen=True)
class PermissionRule:
    tool: str
    match: str


class RuleSet:
    """Workspace rules that turn a `wait` decision into `allow`."""

    def __init__(self, rules: tuple[PermissionRule, ...] = ()) -> None:
        self.rules = rules

    def allow(self, name: str, arguments: dict[str, Any]) -> bool:
        target = _match_target(name, arguments)
        return any(
            (rule.tool == "*" or rule.tool == name) and fnmatch.fnmatchcase(target, rule.match)
            for rule in self.rules
        )


def _match_target(name: str, arguments: dict[str, Any]) -> str:
    key = PRIMARY_ARG.get(name)
    if key is None:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    return str(arguments.get(key) or "")


def load_rules(workspace: Path) -> RuleSet:
    path = Path(workspace) / RULES_PATH
    if not path.is_file():
        return RuleSet()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{RULES_PATH}: {exc}") from exc
    raw = data.get("allow") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise ValidationError(f"{RULES_PATH}: expected an object with an allow list")
    rules: list[PermissionRule] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("tool") or not item.get("match"):
            raise ValidationError(f"{RULES_PATH}: each rule needs tool and match")
        rules.append(PermissionRule(tool=str(item["tool"]), match=str(item["match"])))
    return RuleSet(tuple(rules))
