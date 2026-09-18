from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from woke.errors import ModelError
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol




@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class Model(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelReply: ...

    def summarize(self, text: str) -> str: ...


class ScriptModel:
    """In-process test model. complete() pops scripted replies; summarize() does not."""

    def __init__(self, replies: list[ModelReply | Callable[..., ModelReply]]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, Any]]] = []
        self.summaries: list[str] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelReply:
        self.calls.append(messages)
        if not self.replies:
            return ModelReply(text="(script exhausted)")
        reply = self.replies.pop(0)
        return reply(messages) if callable(reply) else reply

    def summarize(self, text: str) -> str:
        self.summaries.append(text)
        return f"[summary {len(text)} chars]"


class ReactiveModel:
    """Context-shaped fake used across Host restarts (no in-memory queue)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelReply:
        if self.name == "hello":
            return ModelReply(text="hello from woke")
        if _last_tool(messages):
            return ModelReply(text="finished")
        if self.name == "write_then_done":
            return ModelReply(
                text="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "out.txt", "content": "hello-woke"},
                    )
                ],
            )
        if self.name == "write_huge_then_done":
            return ModelReply(
                text="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "out.txt", "content": "x" * 4000},
                    )
                ],
            )
        if self.name == "shell_then_done":
            return ModelReply(
                text="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="run_shell",
                        arguments={"command": "echo pwned > owned.txt"},
                    )
                ],
            )
        return ModelReply(text="hello from woke")

    def summarize(self, text: str) -> str:
        return f"[summary {len(text)} chars]"


def fake_from_env() -> Model | None:
    name = os.environ.get("WOKE_FAKE_MODEL", "").strip()
    if not name:
        return None
    return ReactiveModel(name)


class OpenAICompatModel:
    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelReply:
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = tools
        data = self._post("/chat/completions", body)
        choice = (data.get("choices") or [{}])[0].get("message") or {}
        calls = []
        for raw in choice.get("tool_calls") or []:
            fn = raw.get("function") or {}
            arguments = fn.get("arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = arguments if isinstance(arguments, dict) else {}
            calls.append(
                ToolCall(
                    id=str(raw.get("id") or f"call-{len(calls)+1}"),
                    name=str(fn.get("name") or ""),
                    arguments=parsed,
                )
            )
        return ModelReply(text=choice.get("content") or "", tool_calls=calls)

    def summarize(self, text: str) -> str:
        reply = self.complete(
            [
                {
                    "role": "user",
                    "content": (
                        "Summarize this agent transcript for a later turn. "
                        "Keep file paths, decisions, and unfinished work.\n\n"
                        + text
                    ),
                }
            ],
            tools=[],
        )
        return reply.text or "[empty summary]"

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(3):
            req = urllib.request.Request(
                self.base_url + path,
                data=payload,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = ModelError(f"model HTTP {exc.code}: {detail}")
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = ModelError(f"model transport: {exc.reason}")
                if attempt == 2:
                    raise last_error from exc
            time.sleep(0.4 * (2**attempt))
        raise last_error or ModelError("model request failed")


def build_model(model_id: str | None = None) -> Model:
    fake = fake_from_env()
    if fake is not None:
        return fake
    if os.environ.get("WOKE_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        api_key = os.environ.get("WOKE_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        model = os.environ.get("WOKE_MODEL", "gpt-4o-mini")
        base = os.environ.get("WOKE_API_BASE", "https://api.openai.com/v1")
        return OpenAICompatModel(api_key=api_key, model=model, base_url=base)
    from woke.config import load_config

    cfg = load_config()
    if cfg.models:
        provider, spec = cfg.resolve(model_id)
        if not provider.api_key:
            raise RuntimeError(f"provider {provider.name} has no api_key")
        from woke.config import MODEL_ALIASES

        wire_model = MODEL_ALIASES.get(spec.model, spec.model)
        return OpenAICompatModel(api_key=provider.api_key, model=wire_model, base_url=provider.base_url)
    raise RuntimeError("set WOKE_API_KEY or add ~/.woke/config.toml / ~/.kimi-code/config.toml")


def active_model_label(model_id: str | None = None) -> str:
    if os.environ.get("WOKE_FAKE_MODEL"):
        return os.environ["WOKE_FAKE_MODEL"]
    if os.environ.get("WOKE_MODEL"):
        return os.environ["WOKE_MODEL"]
    from woke.config import load_config

    cfg = load_config()
    if not cfg.models:
        return "gpt-4o-mini"
    try:
        _provider, spec = cfg.resolve(model_id)
        return spec.display_name
    except RuntimeError:
        return model_id or cfg.default_model or "unknown"


def _last_tool(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages):
        if message.get("role") == "tool":
            return True
        if message.get("role") in {"user", "assistant"}:
            return False
    return False



