"""Tools for exploring and querying one SQLite database. Each returns short text for
the model to read. mcp_server exposes them over MCP."""

import difflib
import re
import sqlite3
from pathlib import Path

from sqlagent.db import QueryTimeout, connect_readonly, run_agent_query
from sqlagent.schema import Table, load_schema

_LEADING_COMMENTS = re.compile(r"^\s*(--[^\n]*\n|/\*.*?\*/)*\s*", re.S)


class ToolInputError(Exception):
    """Bad input from the model; the message tells it how to fix the call."""


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _cell(v, max_chars: int) -> str:
    s = "NULL" if v is None else str(v)
    s = s.replace("\n", " ").replace("|", "/")
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def render_rows(cols: list[str], rows: list[tuple], max_chars: int = 100) -> str:
    lines = [" | ".join(cols)]
    lines += [" | ".join(_cell(v, max_chars) for v in r) for r in rows]
    return "\n".join(lines)


def _suggest(name: str, options: list[str], n: int = 3) -> str:
    lower = {o.lower(): o for o in options}
    hits = difflib.get_close_matches(name.lower(), list(lower), n=n, cutoff=0.5)
    return ", ".join(lower[h] for h in hits) if hits else ", ".join(options[:8])


class DatabaseTools:
    def __init__(
        self,
        db_path: Path,
        query_timeout_s: float = 15.0,
        max_rows: int = 50,
        max_cell_chars: int = 100,
    ):
        self.db_path = Path(db_path)
        self.query_timeout_s = query_timeout_s
        self.max_rows = max_rows
        self.max_cell_chars = max_cell_chars
        self.schema: dict[str, Table] = load_schema(self.db_path)

    def _table(self, name: str) -> Table:
        for t in self.schema.values():
            if t.name.lower() == name.strip().strip('"`[]').lower():
                return t
        raise ToolInputError(
            f"Unknown table '{name}'. Did you mean: {_suggest(name, list(self.schema))}? "
            "Use list_tables to see all tables."
        )

    def _column(self, table: Table, name: str):
        col = table.column(name.strip().strip('"`[]'))
        if col is None:
            raise ToolInputError(
                f"Unknown column '{name}' in table '{table.name}'. Did you mean: "
                f"{_suggest(name, [c.name for c in table.columns])}? "
                f"Use describe_table('{table.name}') to see its columns."
            )
        return col

    def _conn(self) -> sqlite3.Connection:
        return connect_readonly(self.db_path)

    def list_tables(self) -> str:
        lines = [f"{len(self.schema)} tables:"]
        for t in self.schema.values():
            cols = ", ".join(c.name for c in t.columns)
            lines.append(f"- {t.name} ({t.row_count:,} rows): {cols}")
        return "\n".join(lines)

    def describe_table(self, table: str, example_rows: int = 3) -> str:
        t = self._table(table)
        lines = [f"Table {t.name} ({t.row_count:,} rows)", "column | type | key | description"]
        for c in t.columns:
            key = "PK" if c.pk else (f"FK {c.fk}" if c.fk else "")
            desc = c.description
            if c.value_description:
                desc += f" [values: {c.value_description[:200]}]"
            lines.append(f"{c.name} | {c.type} | {key} | {desc}")
        if example_rows:
            conn = self._conn()
            try:
                cur = conn.execute(f"SELECT * FROM {_q(t.name)} LIMIT ?", (example_rows,))
                cols = [d[0] for d in cur.description]
                lines += ["", "Example rows:", render_rows(cols, cur.fetchall(), 40)]
            finally:
                conn.close()
        return "\n".join(lines)

    def search_schema(self, keyword: str, limit: int = 10) -> str:
        # Score: word hits in column name (3), table name (2), description (1),
        # plus fuzzy similarity of the whole keyword to the column name.
        words = [w for w in re.findall(r"[a-z0-9]+", keyword.lower()) if len(w) > 1]
        if not words:
            raise ToolInputError("keyword must contain at least one word, e.g. 'county'.")
        scored = []
        for t in self.schema.values():
            for c in t.columns:
                name, tname = c.name.lower(), t.name.lower()
                desc = f"{c.description} {c.value_description}".lower()
                lexical = sum(3 * (w in name) + 2 * (w in tname) + (w in desc) for w in words)
                fuzzy = difflib.SequenceMatcher(None, keyword.lower(), name).ratio()
                if lexical > 0 or fuzzy >= 0.75:
                    scored.append((lexical + 2 * fuzzy, t.name, c))
        if not scored:
            return f"No columns match '{keyword}'. Try a synonym or list_tables."
        scored.sort(key=lambda x: (-x[0], x[1], x[2].name))
        lines = [f"Top matches for '{keyword}' (table.column | type | description):"]
        for _, tname, c in scored[:limit]:
            lines.append(f"{tname}.{c.name} | {c.type} | {c.description[:120]}")
        return "\n".join(lines)

    def sample_values(self, table: str, column: str, limit: int = 10) -> str:
        t = self._table(table)
        c = self._column(t, column)
        limit = max(1, min(limit, 50))
        conn = self._conn()
        try:
            vals = [r[0] for r in conn.execute(
                f"SELECT DISTINCT {_q(c.name)} FROM {_q(t.name)} "
                f"WHERE {_q(c.name)} IS NOT NULL LIMIT ?", (limit,))]
        finally:
            conn.close()
        shown = ", ".join(repr(v) if isinstance(v, str) else str(v) for v in vals)
        return f"{t.name}.{c.name} ({c.type}) sample values: {shown or '(all NULL)'}"

    def distinct_values(
        self, table: str, column: str, like: str | None = None, limit: int = 20
    ) -> str:
        t = self._table(table)
        c = self._column(t, column)
        limit = max(1, min(limit, 100))
        col, tab = _q(c.name), _q(t.name)
        where, params = "", []
        if like:
            if "%" not in like and "_" not in like:
                like = f"%{like}%"
            where, params = f"WHERE {col} LIKE ?", [like]
        conn = self._conn()
        try:
            n_distinct, n_null = conn.execute(
                f"SELECT COUNT(DISTINCT {col}), SUM({col} IS NULL) FROM {tab}").fetchone()
            rows = conn.execute(
                f"SELECT {col}, COUNT(*) AS n FROM {tab} {where} GROUP BY {col} "
                f"ORDER BY n DESC LIMIT ?", [*params, limit]).fetchall()
        finally:
            conn.close()
        head = (f"{t.name}.{c.name}: {n_distinct:,} distinct values, {n_null or 0:,} NULLs"
                + (f"; matching {like!r}:" if like else "; most frequent:"))
        if not rows:
            return head + f" none. Try a broader pattern or sample_values('{t.name}', '{c.name}')."
        return head + "\n" + render_rows([c.name, "count"], rows, self.max_cell_chars)

    def run_query(self, sql: str, max_rows: int | None = None) -> str:
        max_rows = max(1, min(max_rows or self.max_rows, self.max_rows))
        body = _LEADING_COMMENTS.sub("", sql).strip().rstrip(";").strip()
        if not re.match(r"(?i)(select|with)\b", body):
            raise ToolInputError("Only read-only SELECT (or WITH ... SELECT) queries are allowed.")
        try:
            cols, rows, capped = run_agent_query(self.db_path, body, self.query_timeout_s)
        except QueryTimeout as e:
            raise ToolInputError(
                f"Query timed out ({e}). Simplify it: filter earlier, avoid cross joins, "
                "or check join conditions."
            ) from e
        except sqlite3.DatabaseError as e:
            raise ToolInputError(self._explain_sql_error(str(e))) from e
        total = f"{len(rows):,}+" if capped else f"{len(rows):,}"
        if not rows:
            return (f"0 rows (columns: {', '.join(cols)}). If you expected results, check "
                    "filter values with distinct_values and join keys.")
        shown = rows[:max_rows]
        head = f"{total} rows" + (f" (showing first {len(shown)})" if len(shown) < len(rows) else "")
        return head + "\n" + render_rows(cols, shown, self.max_cell_chars)

    def _explain_sql_error(self, msg: str) -> str:
        hint = ""
        if m := re.search(r"no such column: ([\w.\"`]+)", msg):
            name = m.group(1).split(".")[-1].strip('"`')
            all_cols = sorted({c.name for t in self.schema.values() for c in t.columns})
            hint = f" Similar columns: {_suggest(name, all_cols)}."
            hint += " Quote names containing spaces with double quotes or backticks."
        elif m := re.search(r"no such table: ([\w.\"`]+)", msg):
            name = m.group(1).strip('"`')
            hint = f" Tables: {_suggest(name, list(self.schema))}."
        elif "not authorized" in msg:
            hint = " Only reading tables is allowed (no PRAGMA, ATTACH or writes)."
        elif "one statement" in msg:
            hint = " Send exactly one SELECT statement."
        elif "ambiguous column" in msg:
            hint = " Qualify the column with its table alias, e.g. T1.id."
        return f"SQL error: {msg.rstrip('.')}.{hint}"
