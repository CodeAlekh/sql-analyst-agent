"""Baseline: the full schema (with BIRD's column descriptions and example rows) in one
prompt, one model call, no tools, no execution feedback. THis is the first system we implemented."""

from sqlagent.db import connect_readonly
from sqlagent.llm import LLM, Meter
from sqlagent.paths import db_path
from sqlagent.systems.base import SQL_RULES, Prediction, System, extract_sql
from sqlagent.tools import DatabaseTools, render_rows

SYSTEM_PROMPT = f"""You are an expert data analyst who writes SQLite queries.
Given a database schema, evidence (domain hints written by an expert) and a question, \
write ONE SQLite query that answers the question.

Rules:
{SQL_RULES}
- Reply with only the SQL inside one ```sql code block."""


def _ident(name: str) -> str:
    return name if name.isidentifier() else f"`{name}`"


def render_schema(tools: DatabaseTools, example_rows: int = 3) -> str:
    parts = []
    conn = connect_readonly(tools.db_path)
    try:
        for t in tools.schema.values():
            lines = [f"CREATE TABLE {_ident(t.name)} ("]
            for i, c in enumerate(t.columns):
                col = f"  {_ident(c.name)} {c.type}{' PRIMARY KEY' if c.pk else ''}"
                if c.fk:
                    col += f" REFERENCES {c.fk}"
                col += "," if i < len(t.columns) - 1 else ""
                note = c.description if c.description.lower() != c.name.lower() else ""
                if c.value_description:
                    note += f" [values: {c.value_description[:150]}]"
                lines.append(col + (f"  -- {note.strip()}" if note.strip() else ""))
            lines.append(f");  -- {t.row_count:,} rows")
            cur = conn.execute(f'SELECT * FROM "{t.name}" LIMIT {example_rows}')
            rows = render_rows([d[0] for d in cur.description], cur.fetchall(), 40)
            lines.append(f"/* {example_rows} example rows:\n{rows}\n*/")
            parts.append("\n".join(lines))
    finally:
        conn.close()
    return "\n\n".join(parts)


class OneShot(System):
    name = "oneshot"

    def __init__(self, llm: LLM, max_tokens: int = 1024):
        self.llm, self.max_tokens = llm, max_tokens
        self._schemas: dict[str, str] = {}

    def schema_for(self, db_id: str) -> str:
        if db_id not in self._schemas:
            self._schemas[db_id] = render_schema(DatabaseTools(db_path(db_id)))
        return self._schemas[db_id]

    async def answer(self, question: dict, meter: Meter) -> Prediction:
        msg = await self.llm.create(
            meter=meter, agent="oneshot", question_id=question["question_id"],
            max_tokens=self.max_tokens,
            # Same schema for every question on a DB, so cache it (Haiku needs >= 4,096 tokens)
            system=[
                {"type": "text", "text": SYSTEM_PROMPT},
                {"type": "text", "text": "Database schema:\n\n" + self.schema_for(question["db_id"]),
                 "cache_control": {"type": "ephemeral"}},
            ],
            messages=[{"role": "user", "content":
                       f"Evidence: {question['evidence'] or '(none)'}\n\nQuestion: {question['question']}"}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        sql = extract_sql(text)
        return Prediction(sql=sql, steps=1, notes="" if sql else f"no SQL in reply: {text[:200]}")
