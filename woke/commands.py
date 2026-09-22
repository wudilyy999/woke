from __future__ import annotations

import re
from pathlib import Path

COMMANDS_DIR = ".woke/commands"
COMMAND_CAP = 20_000
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def load_commands(workspace: Path) -> dict[str, str]:
    """Prompt templates from <workspace>/.woke/commands/<name>.md, keyed by command name."""
    root = Path(workspace) / COMMANDS_DIR
    if not root.is_dir():
        return {}
    out: dict[str, str] = {}
    for path in sorted(root.glob("*.md")):
        name = path.stem.lower()
        if not NAME_RE.match(name):
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        if len(text) > COMMAND_CAP:
            text = text[:COMMAND_CAP] + "\n…[truncated]"
        out[name] = text
    return out


def expand_command(template: str, arguments: str) -> str:
    arguments = arguments.strip()
    if "$ARGUMENTS" in template:
        return template.replace("$ARGUMENTS", arguments)
    return (template + ("\n\n" + arguments if arguments else "")).strip()
