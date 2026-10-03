import copy
from types import SimpleNamespace

import pytest
from anthropic.types import Message

from sqlagent.llm import LLM, Meter
from sqlagent.mcp_server.server import build_server
from sqlagent.paths import db_path
from sqlagent.systems.single_agent import SingleAgent
from sqlagent.trace import Tracer

QUESTION = {"question_id": 1, "db_id": "shop", "evidence": "", "question": "How many orders?"}
GOOD = "SELECT COUNT(*) FROM orders"


def reply(*blocks):
    content = [{"type": "text", "text": b} if isinstance(b, str) else
               {"type": "tool_use", "id": b[0], "name": b[1], "input": b[2]} for b in blocks]
    return Message.model_validate({
        "id": "msg", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
        "content": content, "stop_sequence": None,
        "stop_reason": "tool_use" if any(c["type"] == "tool_use" for c in content) else "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 10,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    })


class ScriptedClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **params):
        self.calls.append(copy.deepcopy(params))  # the agent keeps appending to messages
        return self.replies.pop(0)


@pytest.fixture
def run_agent(fixture_db, tmp_path):
    async def run(replies, **kwargs):
        client = ScriptedClient(replies)
        llm = LLM("claude-haiku-4-5", "r", Tracer(tmp_path / "trace.jsonl", "r", "single_agent"),
                  client=client, use_cache=False)
        agent = SingleAgent(llm, db_path_for=lambda _: fixture_db,
                            server_for=lambda _: build_server(fixture_db), **kwargs)
        try:
            pred = await agent.answer(QUESTION, meter := Meter())
        finally:
            await agent.aclose()
        return pred, client.calls, meter
    return run


async def test_parallel_tool_calls_return_in_one_message(run_agent):
    pred, calls, meter = await run_agent([
        reply("Checking.", ("t1", "run_query", {"sql": GOOD}),
              ("t2", "describe_table", {"table": "orders"})),
        reply(("t3", "submit_answer", {"sql": GOOD})),
    ])
    results = calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert "4" in results[0]["content"] and "Table orders" in results[1]["content"]
    assert results[-1]["cache_control"] == {"type": "ephemeral"}
    assert pred.sql == GOOD and pred.notes == "submitted" and pred.steps == 2
    assert meter.tool_calls == 3 and meter.llm_calls == 2


async def test_self_check_accepts_a_clean_result(run_agent):
    pred, calls, _ = await run_agent([reply(("t1", "submit_answer", {"sql": GOOD}))])
    assert pred.sql == GOOD and pred.notes == "submitted" and len(calls) == 1


async def test_self_check_rejects_broken_sql_then_accepts_fix(run_agent):
    pred, calls, _ = await run_agent([
        reply(("t1", "submit_answer", {"sql": "SELECT nope FROM orders"})),
        reply(("t2", "submit_answer", {"sql": GOOD})),
    ])
    rejected = calls[1]["messages"][-1]["content"][0]
    assert rejected["is_error"] and "Not accepted" in rejected["content"]
    assert pred.sql == GOOD and pred.notes == "submitted"


async def test_self_check_warns_on_nulls_and_accepts_confirmation(run_agent):
    sql = "SELECT note FROM orders"
    pred, calls, _ = await run_agent([
        reply(("t1", "submit_answer", {"sql": sql})),
        reply(("t2", "submit_answer", {"sql": sql})),
    ])
    warning = calls[1]["messages"][-1]["content"][0]["content"]
    assert "NULL" in warning and "in that order" in warning
    assert pred.sql == sql and pred.notes == "submitted" and pred.steps == 2


async def test_broken_query_is_rejected_even_after_a_warning(run_agent):
    sql = "SELECT note FROM orders"
    pred, calls, _ = await run_agent([
        reply(("t1", "submit_answer", {"sql": sql})),
        reply(("t2", "submit_answer", {"sql": "SELECT nope FROM orders"})),
        reply(("t3", "submit_answer", {"sql": sql})),
    ])
    assert calls[2]["messages"][-1]["content"][0]["is_error"] is True
    assert pred.sql == sql and pred.steps == 3


async def test_max_turns_falls_back_to_last_good_query(run_agent):
    pred, _, _ = await run_agent([
        reply(("t1", "run_query", {"sql": GOOD})),
        reply(("t2", "run_query", {"sql": "SELECT broken"})),
    ], max_turns=2)
    assert pred.sql == GOOD and pred.notes == "max_turns" and pred.steps == 2


async def test_last_turn_is_announced(run_agent):
    _, calls, _ = await run_agent([
        reply(("t1", "run_query", {"sql": GOOD})),
        reply(("t2", "run_query", {"sql": GOOD})),
        reply(("t3", "submit_answer", {"sql": GOOD})),
    ], max_turns=3)
    def texts(call):
        return [b for b in call["messages"][-1]["content"] if b["type"] == "text"]

    assert texts(calls[1]) == []
    assert texts(calls[2]) == [{"type": "text", "text": "This is your last turn. Call "
                                "submit_answer now with your best query."}]


async def test_text_only_replies_get_one_nudge(run_agent):
    pred, calls, _ = await run_agent([reply("The answer is 4."), reply("Still 4.")])
    assert calls[1]["messages"][-1] == {"role": "user", "content": "Call submit_answer with your final SQL."}
    assert pred.sql is None and pred.notes == "no_tool_call"


async def test_value_probing_off_removes_probe_tools(run_agent):
    _, calls, _ = await run_agent([reply(("t1", "submit_answer", {"sql": GOOD}))],
                                  value_probing=False)
    names = {t["name"] for t in calls[0]["tools"]}
    assert names == {"list_tables", "describe_table", "search_schema", "run_query", "submit_answer"}
    assert "distinct_values" not in calls[0]["system"][0]["text"]


@pytest.mark.skipif(not db_path("california_schools").exists(), reason="BIRD data not downloaded")
async def test_stdio_server_starts_and_answers(tmp_path):
    client = ScriptedClient([reply(("t1", "run_query", {"sql": "SELECT COUNT(*) FROM schools"})),
                             reply(("t2", "submit_answer", {"sql": "SELECT COUNT(*) FROM schools"}))])
    llm = LLM("claude-haiku-4-5", "r", Tracer(tmp_path / "t.jsonl", "r", "s"),
              client=client, use_cache=False)
    agent = SingleAgent(llm)
    try:
        pred = await agent.answer({**QUESTION, "db_id": "california_schools"}, Meter())
    finally:
        await agent.aclose()
    assert "17686" in client.calls[1]["messages"][-1]["content"][0]["content"]
    assert pred.notes == "submitted"
