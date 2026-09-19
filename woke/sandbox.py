from __future__ import annotations

import platform
import shutil
from pathlib import Path

from woke.errors import ValidationError


class SandboxUnavailable(ValidationError):
    pass


class SandboxManager:
    """Minimal platform sandbox for shell and MCP side effects."""

    def __init__(self) -> None:
        self.system = platform.system()
        self.seatbelt = shutil.which("sandbox-exec") if self.system == "Darwin" else None
        self.bwrap = shutil.which("bwrap") if self.system == "Linux" else None

    @property
    def available(self) -> bool:
        return bool(self.seatbelt or self.bwrap)

    def wrap_command(self, command: str, workspace: Path, allow_write: bool) -> list[str]:
        if self.seatbelt:
            return self._seatbelt_argv(command, workspace, allow_write, [workspace])
        if self.bwrap:
            return self._bwrap_argv(command, workspace, allow_write)
        raise SandboxUnavailable(f"sandbox unavailable on {self.system or 'this platform'}")

    def wrap_argv(
        self,
        argv: list[str],
        workspace: Path,
        allow_write: bool,
        extra_read_roots: list[Path] | None = None,
    ) -> list[str]:
        if self.seatbelt:
            return [
                self.seatbelt or "sandbox-exec",
                "-p",
                self._seatbelt_policy(workspace, allow_write, extra_read_roots),
                *argv,
            ]
        if self.bwrap:
            mounts = [
                self.bwrap or "bwrap",
                "--die-with-parent",
                "--new-session",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind",
                "/bin",
                "/bin",
                "--ro-bind",
                "/sbin",
                "/sbin",
                "--ro-bind",
                "/etc",
                "/etc",
            ]
            for root in extra_read_roots or []:
                mounts.extend(["--ro-bind", str(root), str(root)])
            mounts.extend(
                [
                    *( ["--bind", str(workspace), str(workspace)] if allow_write else ["--ro-bind", str(workspace), str(workspace)] ),
                    "--chdir",
                    str(workspace),
                    "--",
                    *argv,
                ]
            )
            return mounts
        raise SandboxUnavailable(f"sandbox unavailable on {self.system or 'this platform'}")

    def _seatbelt_argv(self, command: str, workspace: Path, allow_write: bool) -> list[str]:
        policy = self._seatbelt_policy(workspace, allow_write, [workspace])
        return [self.seatbelt or "sandbox-exec", "-p", policy, "/bin/sh", "-lc", command]

    def _bwrap_argv(self, command: str, workspace: Path, allow_write: bool) -> list[str]:
        argv = [
            self.bwrap or "bwrap",
            "--die-with-parent",
            "--new-session",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/bin",
            "/bin",
            "--ro-bind",
            "/sbin",
            "/sbin",
            "--ro-bind",
            "/etc",
            "/etc",
        ]
        if allow_write:
            argv.extend(["--bind", str(workspace), str(workspace)])
        else:
            argv.extend(["--ro-bind", str(workspace), str(workspace)])
        argv.extend(["--chdir", str(workspace), "--", "/bin/sh", "-lc", command])
        return argv

    def _seatbelt_policy(
        self,
        workspace: Path,
        allow_write: bool,
        extra_read_roots: list[Path] | None = None,
    ) -> str:
        root = str(workspace.resolve())
        parts = [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow signal (target same-sandbox))",
            "(allow sysctl*)",
            "(allow file-read-metadata)",
            '(allow file-read* file-read-metadata (subpath "/System") (subpath "/usr") (subpath "/bin") (subpath "/sbin") (subpath "/etc") (literal "/dev/null") (literal "/dev/zero"))',
            '(allow file-read* (subpath "/Library/Apple") (subpath "/private/etc") (subpath "/var"))',
            '(allow file-read* (subpath "/opt/homebrew") (subpath "/usr/local"))',
            '(allow file-read* (subpath "/opt/anaconda3") (subpath "/opt/anaconda3/lib") (subpath "/opt/anaconda3/bin"))',
            '(allow file-read* file-test-existence (literal "/"))',
            '(allow file-read* file-test-existence file-write-data (literal "/dev/null"))',
            '(allow file-read* file-write-data (subpath "/dev/fd"))',
            f'(allow file-read* file-test-existence (subpath "{_escape_seatbelt(root)}"))',
        ]
        for extra in extra_read_roots or []:
            parts.append(f'(allow file-read* file-test-existence (subpath "{_escape_seatbelt(str(extra.resolve()))}"))')
        if allow_write:
            parts.append(f'(allow file-write* (subpath "{_escape_seatbelt(root)}"))')
        parts.append('(allow file-write* (subpath "/tmp") (subpath "/private/tmp"))')
        return "\n".join(parts)


def _escape_seatbelt(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def sandbox_shell_command(command: str, workspace: Path, allow_write: bool) -> list[str]:
    return SandboxManager().wrap_command(command, workspace, allow_write)
