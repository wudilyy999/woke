from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Provider:
    name: str
    type: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    model: str
    display_name: str
    max_context_size: int


@dataclass
class AppConfig:
    default_model: str
    providers: dict[str, Provider]
    models: dict[str, ModelSpec]
    source: Path | None

    def resolve(self, model_id: str | None = None) -> tuple[Provider, ModelSpec]:
        mid = (model_id or os.environ.get("WOKE_MODEL") or self.default_model).strip()
        spec = self.models.get(mid)
        if spec is None:
            for item in self.models.values():
                if item.model == mid or item.display_name == mid:
                    spec = item
                    break
        if spec is None:
            raise RuntimeError(f"unknown model {mid!r}; try: woke models")
        provider = self.providers.get(spec.provider)
        if provider is None:
            raise RuntimeError(f"unknown provider {spec.provider!r} for {spec.id}")
        return provider, spec


# Kimi Code ids that the local OpenAI proxy exposes under a slightly different name.
MODEL_ALIASES = {
    "deepseek-v4.1-flash": "deepseek-v4-flash",
}


def kimi_config_path() -> Path:
    return Path.home() / ".kimi-code" / "config.toml"


def woke_config_path() -> Path:
    return Path.home() / ".woke" / "config.toml"


def load_config(path: Path | None = None) -> AppConfig:
    for candidate in (
        [path] if path is not None else [woke_config_path(), kimi_config_path()]
    ):
        if candidate is None or not candidate.is_file():
            continue
        data = tomllib.loads(candidate.read_text(encoding="utf-8"))
        return _parse(data, candidate)
    return AppConfig(default_model="", providers={}, models={}, source=None)


def _parse(data: dict, source: Path) -> AppConfig:
    providers: dict[str, Provider] = {}
    for name, spec in (data.get("providers") or {}).items():
        if not isinstance(spec, dict) or not spec.get("base_url"):
            continue
        key = str(spec.get("api_key") or "")
        if not key and str(spec.get("type") or "") in {"kimi"}:
            key = _kimi_oauth_token()
        providers[str(name)] = Provider(
            name=str(name),
            type=str(spec.get("type") or "openai"),
            base_url=str(spec["base_url"]).rstrip("/"),
            api_key=key,
        )
    models: dict[str, ModelSpec] = {}
    for name, spec in (data.get("models") or {}).items():
        if not isinstance(spec, dict) or not spec.get("model"):
            continue
        models[str(name)] = ModelSpec(
            id=str(name),
            provider=str(spec.get("provider") or ""),
            model=str(spec["model"]),
            display_name=str(spec.get("display_name") or spec["model"]),
            max_context_size=int(spec.get("max_context_size") or 0),
        )
    return AppConfig(
        default_model=str(data.get("default_model") or ""),
        providers=providers,
        models=models,
        source=source,
    )


def _kimi_oauth_token() -> str:
    path = Path.home() / ".kimi-code" / "credentials" / "kimi-code.json"
    if not path.is_file():
        return ""
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(data.get("access_token") or "")
