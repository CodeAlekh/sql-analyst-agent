"""Build the experience file: the single agent's genuine mistakes on the iteration split,
each with the gold SQL and a lesson the model writes about its own mistake.

    uv run python -m sqlagent.experience
"""

import asyncio
import json
import re

from dotenv import load_dotenv

from sqlagent.db import run_agent_query
from sqlagent.llm import LLM, Meter
from sqlagent.paths import DATA, QUESTIONS, ROOT, SPLITS, db_path
from sqlagent.tools import render_rows
from sqlagent.trace import Tracer

SOURCE_RUN = "single_agent-v3-iteration"
# Failures triaged as the model's fault, not a gold or question issue (docs/failure_analysis.md).
GENUINE_MISTAKES = (49, 120, 159, 187, 264, 319, 921, 1529)
EXPERIENCE_PATH = DATA / "experience" / f"{SOURCE_RUN}.json"
# Gold or question issues (docs/failure_analysis.md): kept out of the solved examples so the
# agent doesn't learn annotation errors.
FLAGGED = (2, 19, 54, 102, 121, 147, 162, 197, 296, 648, 683, 908, 973, 1021, 1107, 1233, 1245,
           1249, 1421, 1458)

LESSON_PROMPT = """You answered a text-to-SQL question wrong. Review your mistake.

Question: {question}
Evidence: {evidence}

Your SQL:
{wrong_sql}
Your result:
{wrong_result}

Correct SQL:
{gold_sql}
Correct result:
{gold_result}

First compare the two results, step by step:
1. Columns: are they the same, in the same order? Which are extra or missing?
2. Rows: is the number of rows the same?
3. Values: where do the values differ?

Then, based on the differences you found, write the lesson you should remember for future \
questions on any database, in one or two sentences: what to do differently. Do not mention this \
question's tables, columns or values. Put it on the last line, starting with "Lesson:"."""


def preview(db_id: str, sql: str, rows: int = 5) -> str:
    try:
        cols, result, _ = run_agent_query(db_path(db_id), sql)
    except Exception as e:
        return f"error: {e}"
    return f"{len(result)} rows\n" + render_rows(cols, result[:rows])


async def build() -> list[dict]:
    questions = {q["question_id"]: q for q in json.loads(QUESTIONS.read_text())}
    preds = {p["question_id"]: p for p in map(json.loads, (ROOT / "runs" / SOURCE_RUN /
                                                         "predictions.jsonl").open())}
    run_dir = ROOT / "runs" / "experience-build"
    run_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM("claude-haiku-4-5", "experience-build",
              Tracer(run_dir / "trace.jsonl", "experience-build", "experience"))

    async def one(qid: int) -> dict:
        q, wrong = questions[qid], preds[qid]["sql"]
        msg = await llm.create(
            meter=Meter(), agent="reflect", question_id=qid, max_tokens=600,
            messages=[{"role": "user", "content": LESSON_PROMPT.format(
                question=q["question"], evidence=q["evidence"] or "(none)",
                wrong_sql=wrong.strip(), wrong_result=preview(q["db_id"], wrong),
                gold_sql=q["SQL"].strip(), gold_result=preview(q["db_id"], q["SQL"]))}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        lesson = re.split(r"\**Lesson\**:\**", text)[-1].strip()  # tolerate markdown bold
        return {"question_id": qid, "db_id": q["db_id"], "question": q["question"],
                "evidence": q["evidence"], "wrong_sql": wrong.strip(),
                "gold_sql": q["SQL"].strip(), "lesson": lesson}

    return await asyncio.gather(*(one(qid) for qid in GENUINE_MISTAKES))


def load(path=EXPERIENCE_PATH) -> list[dict]:
    return json.loads(path.read_text())


def conventions() -> list[dict]:
    """Solved iteration questions (question, evidence, gold SQL), minus flagged ones and the
    mistakes, which are shown as wrong vs correct pairs instead."""
    ids = set(json.loads((SPLITS / "iteration.json").read_text())["question_ids"])
    ids -= set(FLAGGED) | set(GENUINE_MISTAKES)
    return [{"question_id": q["question_id"], "db_id": q["db_id"], "question": q["question"],
             "evidence": q["evidence"], "gold_sql": q["SQL"].strip()}
            for q in json.loads(QUESTIONS.read_text()) if q["question_id"] in ids]


def main() -> None:
    load_dotenv(ROOT / ".env")
    print(f"{len(GENUINE_MISTAKES)} mistakes from {SOURCE_RUN} -> {EXPERIENCE_PATH.relative_to(ROOT)}")
    entries = asyncio.run(build())
    EXPERIENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPERIENCE_PATH.write_text(json.dumps(entries, indent=1, ensure_ascii=False) + "\n")
    for e in entries:
        print(f"q{e['question_id']}: {e['lesson']}")


if __name__ == "__main__":
    main()
