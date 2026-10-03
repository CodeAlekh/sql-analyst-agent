import sqlite3

import pytest

from sqlagent.db import run_agent_query
from sqlagent.tools import DatabaseTools, ToolInputError


@pytest.fixture
def tools(fixture_db):
    return DatabaseTools(fixture_db, query_timeout_s=1.0, max_rows=2, max_cell_chars=20)


def test_schema_loads_keys_and_descriptions_in_both_encodings(tools):
    cust, orders = tools.schema["customer"], tools.schema["orders"]
    assert cust.row_count == 3 and cust.column("id").pk
    assert orders.column("customer_id").fk == "customer.id"
    assert cust.column("home country").description == "country of residence"
    assert "order total" in orders.column("amount").description


def test_list_and_describe(tools):
    assert "customer (3 rows)" in tools.list_tables()
    out = tools.describe_table("ORDERS")  # case-insensitive
    assert "FK customer.id" in out and "Example rows:" in out


def test_unknown_names_suggest_alternatives(tools):
    with pytest.raises(ToolInputError, match="Did you mean: customer"):
        tools.describe_table("customers")
    with pytest.raises(ToolInputError, match="Home Country"):
        tools.distinct_values("customer", "HomeCountry")


def test_search_schema_finds_description_matches(tools):
    assert "orders.amount" in tools.search_schema("total")


def test_distinct_values_like_is_substring_by_default(tools):
    out = tools.distinct_values("customer", "Home Country", like="us")
    assert "USA" in out and "US |" in out and "Nepal" not in out


def test_run_query_truncates_rows_and_cells(tools):
    out = tools.run_query("SELECT order_id, note FROM orders ORDER BY order_id")
    assert out.startswith("4 rows (showing first 2)")
    assert "13" not in out  # only 2 rows shown
    long = tools.run_query("SELECT note FROM orders WHERE order_id = 13")
    assert "…" in long and "on and on and on and on" not in long


def test_run_query_empty_result_gives_a_hint(tools):
    assert "distinct_values" in tools.run_query("SELECT * FROM customer WHERE name = 'Zed'")


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "PRAGMA table_info(orders)",
    "  -- sneaky\n UPDATE orders SET amount = 0",
])
def test_run_query_rejects_non_select(tools, sql):
    with pytest.raises(ToolInputError, match="Only read-only SELECT"):
        tools.run_query(sql)


def test_run_query_rejects_multiple_statements(tools):
    with pytest.raises(ToolInputError, match="exactly one"):
        tools.run_query("SELECT 1; DELETE FROM orders")


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "ATTACH DATABASE ':memory:' AS other",
    "PRAGMA writable_schema = ON",
])
def test_executor_blocks_writes_attach_and_pragma_even_without_prefix_check(fixture_db, sql):
    with pytest.raises(sqlite3.DatabaseError):
        run_agent_query(fixture_db, sql)
    assert sqlite3.connect(fixture_db).execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 4


def test_run_query_times_out_with_actionable_message(tools):
    runaway = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT MAX(i) FROM n"
    with pytest.raises(ToolInputError, match="timed out"):
        tools.run_query(runaway)


def test_sql_errors_suggest_columns(tools):
    with pytest.raises(ToolInputError, match="Similar columns: .*Home Country"):
        tools.run_query("SELECT HomeCountry FROM customer")


def test_oneshot_schema_is_valid_ddl_with_descriptions(tools, fixture_db):
    from sqlagent.systems.oneshot import render_schema

    ddl = render_schema(tools)
    assert "REFERENCES customer.id" in ddl and "-- country of residence" in ddl
    # Strip comments/examples and check the CREATE statements actually parse.
    import re
    creates = re.sub(r"/\*.*?\*/", "", ddl, flags=re.S)
    creates = re.sub(r"--[^\n]*", "", creates).replace("REFERENCES customer.id", "")
    sqlite3.connect(":memory:").executescript(creates)
