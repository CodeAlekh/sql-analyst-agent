"""Execution accuracy (EX), same as the official BIRD scorer (mini_dev evaluation_ex.py):
set(pred_rows) == set(gold_rows), and a 30 s timeout or any error scores 0. So row order
and duplicates are ignored, column order is not. Only difference: predictions run
read-only here, so a write query errors instead of changing the database."""

from dataclasses import dataclass
from pathlib import Path

from sqlagent.db import execute

OFFICIAL_TIMEOUT_S = 30.0


def ex_match(pred_rows: list[tuple], gold_rows: list[tuple]) -> bool:
    return set(pred_rows) == set(gold_rows)


@dataclass
class ExResult:
    correct: bool
    error: str | None = None


def score(
    pred_sql: str | None,
    gold_rows: list[tuple],
    db_path: Path,
    timeout_s: float = OFFICIAL_TIMEOUT_S,
) -> ExResult:
    if not pred_sql or not pred_sql.strip():
        return ExResult(False, "no SQL produced")
    try:
        pred_rows = execute(db_path, pred_sql, timeout_s)
    except Exception as e:
        return ExResult(False, f"{type(e).__name__}: {e}")
    return ExResult(ex_match(pred_rows, gold_rows))
