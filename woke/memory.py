from __future__ import annotations

from pathlib import Path

from woke.tools import contained_path

INSTRUCTION_FILES = (
    "AGENTS.md",
    "CLAUDE.md",
    ".woke/instructions.md",
)
MEMORY_REL = ".woke/memory.md"
SECTION_CAP = 8_000
BRIEFING_CAP = 24_000


def memory_path(workspace: Path) -> Path:
    return contained_path(workspace, MEMORY_REL)


def load_briefing(workspace: Path) -> str:
    """Project instructions + durable memory. Missing files are skipped."""
    root = Path(workspace)
    parts: list[str] = []
    for rel in INSTRUCTION_FILES:
        text = _read_cap(root, rel, SECTION_CAP)
        if text:
            parts.append(f"## {rel}\n{text}")
    mem = _read_cap(root, MEMORY_REL, SECTION_CAP)
    if mem:
        parts.append(f"## {MEMORY_REL} (durable memory)\n{mem}")
    briefing = "\n\n".join(parts)
    if len(briefing) > BRIEFING_CAP:
        briefing = briefing[:BRIEFING_CAP] + "\n…[briefing truncated]"
    return briefing


def read_memory(workspace: Path) -> str:
    path = memory_path(workspace)
    if not path.exists():
        return "(empty memory)"
    return path.read_text(encoding="utf-8", errors="replace")


def write_memory(workspace: Path, text: str, mode: str = "append") -> str:
    path = memory_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = text if text.endswith("\n") else text + "\n"
    if mode == "replace":
        path.write_text(body, encoding="utf-8")
        return f"replaced {MEMORY_REL} ({len(body)} bytes)"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + body, encoding="utf-8")
    return f"appended {MEMORY_REL} ({len(body)} bytes)"


def _read_cap(workspace: Path, rel: str, cap: int) -> str:
    try:
        path = contained_path(workspace, rel)
    except Exception:
        return ""
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    text = text.strip()
    if len(text) > cap:
        return text[:cap] + "\n…[truncated]"
    return text
