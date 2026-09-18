from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from woke.config import load_config
from woke.model import build_model


TOML = """
default_model = "p/demo"

[providers.p]
base_url = "http://127.0.0.1:9/v1"
type = "openai"
api_key = "test-key"

[models."p/demo"]
provider = "p"
model = "demo-model"
display_name = "Demo"
max_context_size = 1000
"""


class ConfigTests(unittest.TestCase):
    def test_env_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.toml"
            path.write_text(TOML.replace("demo-model", "custom-model"))
            with patch.dict("os.environ", {"WOKE_CONFIG": str(path)}, clear=True):
                cfg = load_config()
            self.assertEqual(cfg.models["p/demo"].model, "custom-model")

    def test_model_uses_configured_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(TOML.replace("demo-model", "deepseek-v4.1-flash"))
            cfg = load_config(path)
            with patch.dict("os.environ", {}, clear=True), patch("woke.config.load_config", return_value=cfg):
                model = build_model()
            self.assertEqual(model.model, "deepseek-v4.1-flash")

    def test_parse_kimi_shaped_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(TOML, encoding="utf-8")
            cfg = load_config(path)
            self.assertEqual(cfg.default_model, "p/demo")
            provider, spec = cfg.resolve()
            self.assertEqual(provider.base_url, "http://127.0.0.1:9/v1")
            self.assertEqual(spec.model, "demo-model")
            self.assertEqual(spec.display_name, "Demo")
