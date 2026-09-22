from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from woke.tools import DANGEROUS, _html_to_text, _search_results, _unwrap_search_url, execute


LITE_PAGE = """
<table>
  <tr>
    <td>1.</td>
    <td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone&amp;rut=abc" class='result-link'>Example <b>One</b></a></td>
  </tr>
  <tr><td></td><td class='result-snippet'>First <b>snippet</b> text.</td></tr>
  <tr>
    <td>2.</td>
    <td><a rel="nofollow" href="https://direct.example/two" class='result-link'>Example Two</a></td>
  </tr>
  <tr><td></td><td class='result-snippet'>Second snippet.</td></tr>
</table>
"""


class WebTests(unittest.TestCase):
    def test_search_results_parse_lite_page(self) -> None:
        results = _search_results(LITE_PAGE)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "https://example.com/one")
        self.assertEqual(results[0][1], "Example One")
        self.assertEqual(results[0][2], "First snippet text.")
        self.assertEqual(results[1][0], "https://direct.example/two")
        self.assertEqual(results[1][2], "Second snippet.")

    def test_search_url_unwrapping(self) -> None:
        self.assertEqual(
            _unwrap_search_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.test%2Fb&rut=x"),
            "https://a.test/b",
        )
        self.assertEqual(_unwrap_search_url("https://plain.test/x"), "https://plain.test/x")

    def test_html_to_text_drops_script_and_markup(self) -> None:
        text = _html_to_text("<html><script>bad()</script><p>Hello&nbsp;<b>world</b></p></html>")
        self.assertNotIn("bad", text)
        self.assertNotIn("<", text)
        self.assertIn("Hello world", text)

    def test_fetch_rejects_non_http_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ok, output = execute("web_fetch", {"url": "file:///etc/passwd"}, Path(tmp))
        self.assertFalse(ok)
        self.assertIn("unsupported url", output)

    def test_web_tools_require_approval(self) -> None:
        self.assertIn("web_fetch", DANGEROUS)
        self.assertIn("web_search", DANGEROUS)


if __name__ == "__main__":
    unittest.main()
