"""Anthropic Messages API wrapper with a disk cache and tracing."""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic
from anthropic.types import Message

from sqlagent.paths import ROOT
from sqlagent.trace import Tracer

MODELS = ("claude-haiku-4-5", "claude-sonnet-5-5", "claude-opus-5-5")
# Models that accept temperature. Sonnet/Opus 5.5 reject it, so they run at the default.
TEMPERATURE_OK = {"claude-haiku-4-5"}
RESPONSE_CACHE_DIR = ROOT / "cache" / "responses"


@dataclass
class Meter:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    cached_calls: int = 0


class LLM:
    def __init__(
        self,
        model: str,
        run_id: str,
        tracer: Tracer,
        client: anthropic.AsyncAnthropic | None = None,
        use_cache: bool = True,
        cache_dir: Path = RESPONSE_CACHE_DIR,
    ):
        self.model, self.run_id, self.tracer = model, run_id, tracer
        self.client = client or anthropic.AsyncAnthropic(max_retries=4)
        self.use_cache, self.cache_dir = use_cache, cache_dir

    def _cache_path(self, params: dict) -> Path:
        key = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"

    async def create(self, *, meter: Meter, agent: str, question_id=None, **params) -> Message:
        """messages.create, or the saved response if this exact request was made before."""
        params = {"model": self.model, **params}
        if self.model in TEMPERATURE_OK:
            # SDK 1.x dropped the keyword, but the API still takes it for these models
            params.setdefault("extra_body", {"temperature": 0})
        path = self._cache_path(params)
        t0 = time.monotonic()
        if self.use_cache and path.exists():
            msg, cached = Message.model_validate_json(path.read_text()), True
        else:
            msg, cached = await self.client.messages.create(**params), False
            if self.use_cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(msg.model_dump_json())
        latency_ms = round((time.monotonic() - t0) * 1000)

        u = msg.usage
        meter.llm_calls += 1
        meter.cached_calls += cached
        if not cached:
            meter.input_tokens += u.input_tokens
            meter.output_tokens += u.output_tokens
            meter.cache_read_tokens += u.cache_read_input_tokens or 0
            meter.cache_write_tokens += u.cache_creation_input_tokens or 0
        self.tracer.log(
            kind="llm", agent=agent, question_id=question_id, model=self.model,
            stop_reason=msg.stop_reason, cached=cached,
            latency_ms=latency_ms, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
            cache_read=u.cache_read_input_tokens, cache_write=u.cache_creation_input_tokens,
        )
        return msg
