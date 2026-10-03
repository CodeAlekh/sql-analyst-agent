"""Run a system on a split, score it with EX and write runs/<run_id>/.

    uv run python -m sqlagent.eval.run --system oneshot --split dev30

The test split also needs --final. Re-running the same command resumes.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from sqlagent.db import execute
from sqlagent.eval.ex import score
from sqlagent.llm import LLM, MODELS, Meter
from sqlagent.paths import QUESTIONS, ROOT, SPLITS, db_path
from sqlagent.systems.base import System
from sqlagent.trace import Tracer

RUNS = ROOT / "runs"
SYSTEMS = ("oneshot", "single_agent", "experienced_agent")
# Errors that mean the whole run is misconfigured, not that one question failed.
FATAL = (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
         anthropic.NotFoundError, anthropic.BadRequestError)


def load_split(name: str) -> list[dict]:
    ids = json.loads((SPLITS / f"{name}.json").read_text())["question_ids"]
    by_id = {q["question_id"]: q for q in json.loads(QUESTIONS.read_text())}
    return [by_id[i] for i in ids]


def build_system(name: str, llm: LLM, self_check: bool = True, value_probing: bool = True,
                 use_conventions: bool = True) -> System:
    if name == "oneshot":
        from sqlagent.systems.oneshot import OneShot
        return OneShot(llm)
    if name == "single_agent":
        from sqlagent.systems.single_agent import SingleAgent
        return SingleAgent(llm, self_check=self_check, value_probing=value_probing)
    if name == "experienced_agent":
        from sqlagent.systems.experienced_agent import ExperiencedAgent
        return ExperiencedAgent(llm, use_conventions=use_conventions, self_check=self_check,
                                value_probing=value_probing)
    raise SystemExit(f"unknown system {name!r}")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def summarize(preds: list[dict]) -> dict:
    def ex(rows):
        return round(100 * sum(p["correct"] for p in rows) / len(rows), 1) if rows else None

    by_diff = defaultdict(list)
    for p in preds:
        by_diff[p["difficulty"]].append(p)
    lat = sorted(p["latency_s"] for p in preds)

    def mean(values):
        values = list(values)
        return round(statistics.mean(values), 2) if values else None

    return {
        "n": len(preds),
        "ex": ex(preds),
        "ex_by_difficulty": {d: {"n": len(v), "ex": ex(v)} for d, v in sorted(by_diff.items())},
        "latency_mean_s": mean(lat),
        "latency_p90_s": round(lat[int(0.9 * (len(lat) - 1))], 2) if lat else None,
        "steps_mean": mean(p["steps"] for p in preds),
        "llm_calls_mean": mean(p["meter"]["llm_calls"] for p in preds),
        "tool_calls_mean": mean(p["meter"]["tool_calls"] for p in preds),
        "no_sql": sum(p["sql"] is None for p in preds),
    }


async def run(args) -> None:
    questions = load_split(args.split)[: args.limit]
    run_dir = RUNS / args.run_id
    pred_path = run_dir / "predictions.jsonl"
    done = {p["question_id"] for p in read_jsonl(pred_path)}
    todo = [q for q in questions if q["question_id"] not in done]

    print(f"run {args.run_id}: {args.system} on {args.split} ({len(todo)} to do, {len(done)} done), "
          f"model {args.model}")
    if not todo:
        print("nothing to do")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            meta_path.write_text(json.dumps({
                "run_id": args.run_id, "system": args.system, "split": args.split,
                "model": args.model, "limit": args.limit, "started": datetime.now().isoformat(),
                "argv": sys.argv[1:]}, indent=1))

        tracer = Tracer(run_dir / "trace.jsonl", args.run_id, args.system)
        llm = LLM(args.model, args.run_id, tracer, use_cache=not args.no_cache)
        system = build_system(args.system, llm, not args.no_self_check, not args.no_probing,
                              not args.no_conventions)
        sem = asyncio.Semaphore(args.concurrency)

        async def one(q: dict) -> None:
            async with sem:
                meter, t0 = Meter(), time.monotonic()
                try:
                    pred = await system.answer(q, meter)
                    sql, steps, notes = pred.sql, pred.steps, pred.notes
                except FATAL:
                    raise
                except Exception as e:  # one bad question shouldn't kill the run
                    sql, steps, notes = None, 0, f"system error: {type(e).__name__}: {e}"
                latency = time.monotonic() - t0
                path = db_path(q["db_id"])
                gold = await asyncio.to_thread(execute, path, q["SQL"])
                result = await asyncio.to_thread(score, sql, gold, path)
                record = {
                    "question_id": q["question_id"], "db_id": q["db_id"],
                    "difficulty": q["difficulty"], "sql": sql, "correct": result.correct,
                    "exec_error": result.error, "steps": steps, "notes": notes,
                    "latency_s": round(latency, 2), "meter": asdict(meter),
                }
                with pred_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                mark = "ok  " if result.correct else "FAIL"
                print(f"  {mark} q{q['question_id']:<5} {q['difficulty']:<11} "
                      f"{latency:5.1f}s")

        tasks = [asyncio.create_task(one(q)) for q in todo]
        try:
            await asyncio.gather(*tasks)
        except FATAL as e:
            for t in tasks:
                t.cancel()
            print(f"run stopped: {type(e).__name__}: {e}. Completed questions are saved; "
                  "re-run the same command to resume.")
        finally:
            await system.aclose()

    preds = read_jsonl(pred_path)
    summary = summarize([p for p in preds if p["question_id"] in {q["question_id"] for q in questions}])
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--split", required=True, choices=["dev30", "iteration", "test", "dev30_clean",
                                                       "iteration_clean", "test_clean"])
    ap.add_argument("--model", default="claude-haiku-4-5", choices=MODELS)
    ap.add_argument("--run-id")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-self-check", action="store_true", help="ablation: accept any submit")
    ap.add_argument("--no-probing", action="store_true", help="ablation: hide value-probing tools")
    ap.add_argument("--no-conventions", action="store_true",
                    help="experienced_agent: leave out the solved examples")
    args = ap.parse_args()
    if args.split.startswith("test") and not args.final:
        raise SystemExit("the test split is frozen: pass --final only for final numbers")
    args.run_id = args.run_id or f"{args.system}-{args.split}-{args.model.removeprefix('claude-')}"
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
