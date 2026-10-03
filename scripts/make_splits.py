"""Make the splits from BIRD New Dev (seed 42): test (150, final numbers only),
iteration (100) and dev30 (30, a subset of iteration). test and iteration are stratified
by difficulty x database. A question whose gold SQL fails to run is replaced by the next
candidate from the same group."""

import json
import random
from collections import Counter, defaultdict

from sqlagent.db import execute
from sqlagent.paths import QUESTIONS, SPLITS, db_path

SEED = 42
SIZES = {"test": 150, "iteration": 100}
DEV30 = 30
TIMEOUT_S = 30.0


def gold_sql_runs(q: dict) -> bool:
    try:
        execute(db_path(q["db_id"]), q["SQL"], TIMEOUT_S)
        return True
    except Exception as e:
        print(f"  skip q{q['question_id']} [{q['db_id']}]: {e}")
        return False


def by_diff_db(q: dict) -> tuple:
    return q["difficulty"], q["db_id"]


def allocate(strata: dict, n: int) -> dict:
    # Proportional, with leftover slots going to the largest remainders so counts sum to n.
    total = sum(len(v) for v in strata.values())
    quotas = {k: n * len(v) / total for k, v in strata.items()}
    counts = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - counts[k]), k))[: n - sum(counts.values())]:
        counts[k] += 1
    return counts


def stratified_sample(rows, n, key, rng, check=None):
    """Returns (picked, skipped ids)."""
    strata = defaultdict(list)
    for r in rows:
        strata[key(r)].append(r)
    picked, skipped = [], []
    for k, count in sorted(allocate(strata, n).items()):
        candidates = sorted(strata[k], key=lambda r: r["question_id"])
        rng.shuffle(candidates)
        taken = 0
        for q in candidates:
            if taken == count:
                break
            if check and not check(q):
                skipped.append(q["question_id"])
                continue
            picked.append(q)
            taken += 1
        assert taken == count, f"not enough valid questions in group {k}"
    return sorted(picked, key=lambda r: r["question_id"]), skipped


def main() -> None:
    rng = random.Random(SEED)
    pool = json.loads(QUESTIONS.read_text())

    test, skipped = stratified_sample(pool, SIZES["test"], by_diff_db, rng, gold_sql_runs)
    used = {q["question_id"] for q in test} | set(skipped)
    rest = [q for q in pool if q["question_id"] not in used]
    iteration, skipped2 = stratified_sample(rest, SIZES["iteration"], by_diff_db, rng, gold_sql_runs)
    skipped += skipped2
    dev30, _ = stratified_sample(iteration, DEV30, lambda q: q["difficulty"], rng)

    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, rows in [("test", test), ("iteration", iteration), ("dev30", dev30)]:
        ids = [q["question_id"] for q in rows]
        (SPLITS / f"{name}.json").write_text(json.dumps(
            {"name": name, "seed": SEED, "source": "birdsql/bird_sql_dev_20251106",
             "skipped_gold_failures": sorted(skipped), "question_ids": ids}, indent=1))
        print(f"{name:9s} n={len(ids):3d}  {dict(sorted(Counter(q['difficulty'] for q in rows).items()))}"
              f"  dbs={len({q['db_id'] for q in rows})}")
    assert not ({q["question_id"] for q in test} & {q["question_id"] for q in iteration})
    assert {q["question_id"] for q in dev30} <= {q["question_id"] for q in iteration}


if __name__ == "__main__":
    main()
