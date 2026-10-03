"""Single agent: the one-shot schema and rules, plus the MCP database tools in a
hand-written tool-use loop that ends with submit_answer."""

import asyncio
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from mcp import Client, StdioServerParameters

from sqlagent.llm import LLM, Meter
from sqlagent.paths import db_path
from sqlagent.systems.base import SQL_RULES, Prediction, System
from sqlagent.systems.oneshot import render_schema
from sqlagent.tools import DatabaseTools

PROBE_TOOLS = {"sample_values", "distinct_values"}
NUDGE = "Call submit_answer with your final SQL."
LAST_TURN = "This is your last turn. Call submit_answer now with your best query."

SUBMIT_TOOL = {
    "name": "submit_answer",
    "description": "Submit your final answer: one SQLite query. Run it with run_query and "
                   "check the result before submitting. Its columns must be exactly the items "
                   "the question asks for, in that order, as raw values.",
    "input_schema": {
        "type": "object",
        "properties": {"sql": {"type": "string", "description": "The final SQLite query."}},
        "required": ["sql"],
        "additionalProperties": False,
    },
    "strict": True,
}


def stdio_server(db_id: str) -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=["-m", "sqlagent.mcp_server", "--db-id", db_id])


def instructions(max_turns: int, value_probing: bool) -> str:
    probe = ("- Before filtering on a text value, check its exact spelling and format with "
             "distinct_values or sample_values.\n" if value_probing else "")
    return f"""You are an expert data analyst who answers questions by writing SQLite queries.
You get a database schema, evidence (domain hints written by an expert) and a question. \
Use the tools to check your work, then call submit_answer with ONE SQLite query.

How to work:
{probe}- Run your query with run_query before submitting. Check the result makes sense: \
not empty, no unexpected NULLs, a sensible number of rows.
- You have at most {max_turns} turns. Call independent tools together in one turn.

Output: return exactly what the question asks for, nothing more. Do not decide on your own \
what would be useful to show.
- One column per requested item, in the order the question asks for them.
- Return raw column values. Do not combine columns: a "full name" of forename and surname is \
two columns, not forename || ' ' || surname.
- Do not add context columns (IDs, names, dates, counts, the value you sorted by) unless the \
question asks for them.
- Do not round, format or relabel values unless the question or evidence says so.
- Before submitting, compare the result's header with the question, item by item.

Rules:
{SQL_RULES}"""


@dataclass
class Conn:
    client: Client
    tools: list[dict]
    stop: asyncio.Event
    task: asyncio.Task


@dataclass
class QuestionState:
    last_good_sql: str | None = None
    warned: bool = False
    accepted_sql: str | None = None


def _text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")


def _warning(output: str) -> str | None:
    if output.startswith("0 rows"):
        return "The query returned no rows."
    if any(cell.strip() == "NULL" for line in output.splitlines()[2:] for cell in line.split("|")):
        return "The result contains NULL values."
    return None


class SingleAgent(System):
    name = "single_agent"

    def __init__(
        self,
        llm: LLM,
        max_turns: int = 8,
        self_check: bool = True,
        value_probing: bool = True,
        max_tokens: int = 2048,
        db_path_for: Callable = db_path,
        server_for: Callable = stdio_server,
    ):
        self.llm, self.max_turns, self.max_tokens = llm, max_turns, max_tokens
        self.self_check, self.value_probing = self_check, value_probing
        self.db_path_for, self.server_for = db_path_for, server_for
        self._conns: dict[str, Conn] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._schemas: dict[str, str] = {}

    # Each server lives in its own task: MCP's context managers must be entered and
    # exited by the same task, and questions run in many tasks.
    async def _serve(self, db_id: str, ready: asyncio.Future, stop: asyncio.Event) -> None:
        try:
            async with Client(self.server_for(db_id)) as client:
                tools = (await client.list_tools()).tools
                ready.set_result((client, tools))
                await stop.wait()
        except Exception as e:
            if not ready.done():
                ready.set_exception(e)
            else:
                raise

    async def _connect(self, db_id: str) -> Conn:
        lock = self._locks.setdefault(db_id, asyncio.Lock())
        async with lock:
            if db_id not in self._conns:
                ready, stop = asyncio.get_running_loop().create_future(), asyncio.Event()
                task = asyncio.create_task(self._serve(db_id, ready, stop))
                client, mcp_tools = await ready
                tools = [{"name": t.name, "description": t.description or "",
                          "input_schema": t.input_schema}
                         for t in mcp_tools if self.value_probing or t.name not in PROBE_TOOLS]
                self._conns[db_id] = Conn(client, tools + [SUBMIT_TOOL], stop, task)
        return self._conns[db_id]

    async def aclose(self) -> None:
        for conn in self._conns.values():
            conn.stop.set()
        await asyncio.gather(*(c.task for c in self._conns.values()), return_exceptions=True)
        self._conns.clear()

    def schema_for(self, db_id: str) -> str:
        if db_id not in self._schemas:
            self._schemas[db_id] = render_schema(DatabaseTools(self.db_path_for(db_id)))
        return self._schemas[db_id]

    def system_prompt(self, question: dict) -> list[dict]:
        """Subclasses add context here (see experienced_agent.py)."""
        return [
            {"type": "text", "text": instructions(self.max_turns, self.value_probing)},
            {"type": "text", "text": "Database schema:\n\n" + self.schema_for(question["db_id"]),
             "cache_control": {"type": "ephemeral"}},
        ]

    async def _call(self, conn: Conn, name: str, args: dict) -> tuple[str, bool]:
        try:
            result = await conn.client.call_tool(name, args)
            return _text(result), bool(result.is_error)
        except Exception as e:
            return f"Tool call failed: {type(e).__name__}: {e}", True

    async def _submit(self, conn: Conn, sql: str, state: QuestionState) -> tuple[str, bool]:
        if not self.self_check:
            state.accepted_sql = sql
            return "Accepted.", False
        output, is_error = await self._call(conn, "run_query", {"sql": sql})
        if is_error:
            return f"Not accepted, the query failed: {output}\nFix it and submit again.", True
        state.last_good_sql = sql
        warning = _warning(output)
        # One warning per question; after that a query that runs is accepted.
        if warning is None or state.warned:
            state.accepted_sql = sql
            return "Accepted.", False
        state.warned = True
        return (f"Not accepted yet. {warning}\nResult:\n{output}\n\n"
                "Are these columns exactly the items the question asks for, in that order? "
                "Revise the query, or submit the same SQL again to confirm."), False

    async def _run_tool(self, conn: Conn, block, state: QuestionState, meter: Meter,
                        question_id) -> dict:
        t0 = time.monotonic()
        if block.name == "submit_answer":
            output, is_error = await self._submit(conn, block.input.get("sql", ""), state)
        else:
            output, is_error = await self._call(conn, block.name, block.input)
            if block.name == "run_query" and not is_error:
                state.last_good_sql = block.input.get("sql")
        meter.tool_calls += 1
        self.llm.tracer.log(kind="tool", agent=self.name, question_id=question_id,
                            name=block.name, latency_ms=round((time.monotonic() - t0) * 1000),
                            is_error=is_error, output=output[:300])
        result = {"type": "tool_result", "tool_use_id": block.id, "content": output}
        if is_error:
            result["is_error"] = True
        return result

    async def answer(self, question: dict, meter: Meter) -> Prediction:
        db_id, qid = question["db_id"], question["question_id"]
        conn = await self._connect(db_id)
        system = self.system_prompt(question)
        messages = [{"role": "user", "content":
                     f"Evidence: {question['evidence'] or '(none)'}\n\nQuestion: {question['question']}"}]
        state, nudged, steps = QuestionState(), False, 0

        for turn in range(self.max_turns):
            msg = await self.llm.create(
                meter=meter, agent=self.name, question_id=qid, max_tokens=self.max_tokens,
                system=system, tools=conn.tools, messages=messages,
            )
            steps += 1
            messages.append({"role": "assistant", "content": [_block(b) for b in msg.content]})
            uses = [b for b in msg.content if b.type == "tool_use"]
            if not uses:
                if nudged:
                    return Prediction(state.last_good_sql, steps, "no_tool_call")
                nudged = True
                messages.append({"role": "user", "content": NUDGE})
                continue
            results = await asyncio.gather(
                *(self._run_tool(conn, b, state, meter, qid) for b in uses))
            if state.accepted_sql is not None:
                return Prediction(state.accepted_sql, steps, "submitted")
            # Only the newest message carries a cache breakpoint, so the history is
            # cached turn to turn without exceeding the 4-breakpoint limit.
            for m in messages:
                if m["role"] == "user" and isinstance(m["content"], list):
                    for r in m["content"]:
                        r.pop("cache_control", None)
            results[-1]["cache_control"] = {"type": "ephemeral"}
            if turn == self.max_turns - 2:
                results.append({"type": "text", "text": LAST_TURN})
            messages.append({"role": "user", "content": list(results)})

        return Prediction(state.last_good_sql, steps, "max_turns")


def _block(b) -> dict:
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    if b.type == "text":
        return {"type": "text", "text": b.text}
    return b.model_dump(exclude_none=True)
