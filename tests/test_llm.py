from types import SimpleNamespace

import pytest
from anthropic.types import Message

from sqlagent.llm import LLM, Meter
from sqlagent.systems.base import extract_sql
from sqlagent.trace import Tracer


def make_message(text="```sql\nSELECT 1\n```", inp=1000, out=100, read=0, write=0):
    return Message.model_validate({
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": inp, "output_tokens": out,
                  "cache_creation_input_tokens": write, "cache_read_input_tokens": read},
    })


class FakeClient:
    def __init__(self, msg):
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)
        self._msg = msg

    async def _create(self, **params):
        self.calls += 1
        self.last_params = params
        return self._msg


@pytest.fixture
def llm_factory(tmp_path):
    def make(client):
        tracer = Tracer(tmp_path / "trace.jsonl", "r1", "test")
        return LLM("claude-haiku-4-5", "r1", tracer, client=client, cache_dir=tmp_path / "cache")
    return make


async def test_identical_calls_replay_from_cache(llm_factory):
    client = FakeClient(make_message(inp=1000, out=100))
    llm = llm_factory(client)
    m1, m2 = Meter(), Meter()
    params = dict(max_tokens=10, messages=[{"role": "user", "content": "hi"}])
    await llm.create(meter=m1, agent="a", **params)
    await llm.create(meter=m2, agent="a", **params)
    assert client.calls == 1
    assert m1.cached_calls == 0 and m2.cached_calls == 1


async def test_temperature_defaults_to_zero(llm_factory):
    client = FakeClient(make_message())
    llm = llm_factory(client)
    await llm.create(meter=Meter(), agent="a", max_tokens=1, messages=[{"role": "user", "content": "x"}])
    assert client.last_params["extra_body"] == {"temperature": 0}


@pytest.mark.parametrize("text,sql", [
    ("```sql\nSELECT a FROM t;\n```", "SELECT a FROM t"),
    ("first\n```sql\nSELECT 1\n```\nfixed:\n```sql\nSELECT 2\n```", "SELECT 2"),
    ("SELECT x FROM y", "SELECT x FROM y"),
    ("I cannot answer this.", None),
])
def test_extract_sql(text, sql):
    assert extract_sql(text) == sql
