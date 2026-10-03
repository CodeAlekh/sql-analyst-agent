import random

import pytest

from sqlagent.db import execute
from sqlagent.eval.ex import ex_match, score


def official_calculate_ex(predicted_res, ground_truth_res):
    # Verbatim from bird-bench/mini_dev evaluation/evaluation_ex.py
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def test_row_order_and_duplicates_ignored():
    assert ex_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a")])
    assert ex_match([(1,), (1,), (2,)], [(2,), (1,)])


def test_column_order_matters():
    assert not ex_match([(1, "a")], [("a", 1)])


def test_empty_results_match():
    assert ex_match([], [])


def test_type_differences_follow_python_equality():
    assert ex_match([(1,)], [(1.0,)])  # 1 == 1.0 in Python, as in the official scorer
    assert not ex_match([("1",)], [(1,)])


def test_parity_with_official_logic_on_random_results():
    rng = random.Random(0)
    values = [None, 0, 1, 1.0, 2.5, "a", "b", ""]
    for _ in range(2000):
        width = rng.randint(1, 3)
        a = [tuple(rng.choice(values) for _ in range(width)) for _ in range(rng.randint(0, 4))]
        b = rng.choice([a[::-1], a + a[:1], [tuple(rng.choice(values) for _ in range(width))
                                             for _ in range(rng.randint(0, 4))]])
        assert ex_match(a, b) == bool(official_calculate_ex(a, b))


@pytest.fixture
def gold(fixture_db):
    return execute(fixture_db, "SELECT name FROM customer WHERE id <= 2")


def test_score_correct_despite_different_sql(fixture_db, gold):
    assert score("SELECT name FROM customer WHERE name IN ('Bob', 'Ann')", gold, fixture_db).correct


@pytest.mark.parametrize("pred", [None, "", "SELECT nope FROM customer",
                                  "DELETE FROM customer"])
def test_score_failures_are_zero_with_reason(fixture_db, gold, pred):
    r = score(pred, gold, fixture_db)
    assert not r.correct and r.error


def test_score_timeout_is_zero(fixture_db, gold):
    runaway = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT MAX(i) FROM n"
    r = score(runaway, gold, fixture_db, timeout_s=0.5)
    assert not r.correct and "QueryTimeout" in r.error
