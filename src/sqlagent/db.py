"""Read-only SQLite access shared by the tools, the splits script and the EX scorer."""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

# Model-written SQL may only read. mode=ro already blocks writes; this also blocks
# ATTACH, PRAGMA and the rest.
ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


class QueryTimeout(Exception):
    pass


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    # Sort/join spill space goes to RAM; the system /tmp can be full.
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def authorize(action, *_):
    return sqlite3.SQLITE_OK if action in ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


@contextmanager
def timed_connection(db_path: Path, timeout_s: float, untrusted: bool = False):
    conn = connect_readonly(db_path)
    if untrusted:
        conn.set_authorizer(authorize)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
    try:
        yield conn
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e) and time.monotonic() > deadline:
            raise QueryTimeout(f"query exceeded {timeout_s:.0f}s") from e
        raise
    finally:
        conn.close()


def execute(db_path: Path, sql: str, timeout_s: float = 30.0) -> list[tuple]:
    with timed_connection(db_path, timeout_s) as conn:
        return conn.execute(sql).fetchall()


def run_agent_query(
    db_path: Path, sql: str, timeout_s: float = 15.0, fetch_cap: int = 10_000
) -> tuple[list[str], list[tuple], bool]:
    """Run model-written SQL. Returns (columns, rows, hit_fetch_cap)."""
    with timed_connection(db_path, timeout_s, untrusted=True) as conn:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(fetch_cap + 1)
        return cols, rows[:fetch_cap], len(rows) > fetch_cap
