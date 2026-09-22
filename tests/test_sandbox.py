from __future__ import annotations

import unittest
from pathlib import Path

from woke.sandbox import SandboxManager


class SandboxPolicyTests(unittest.TestCase):
    def test_seatbelt_allows_reads_and_confines_writes(self) -> None:
        workspace = Path("/tmp/ws")
        policy = SandboxManager()._seatbelt_policy(workspace, True)
        self.assertIn("(allow network*)", policy)
        self.assertIn("(allow file-read*)", policy)
        self.assertIn('(allow file-write* (subpath "' + str(workspace.resolve()) + '"))', policy)

    def test_bwrap_reads_host_and_mounts_workspace_writable(self) -> None:
        workspace = Path("/tmp/ws")
        argv = SandboxManager()._bwrap_argv("echo hi", workspace, True)
        self.assertNotIn("--unshare-net", argv)
        root = argv.index("--ro-bind")
        self.assertEqual(argv[root + 1 : root + 3], ["/", "/"])
        bind = argv.index("--bind")
        self.assertEqual(argv[bind + 1 : bind + 3], ["/tmp", "/tmp"])
        writable = argv.index(str(workspace))
        self.assertEqual(argv[writable : writable + 2], [str(workspace), str(workspace)])


if __name__ == "__main__":
    unittest.main()
