"""Tables, columns, keys and row counts of a BIRD database, merged with the column
descriptions from database_description/*.csv."""

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

from sqlagent.db import connect_readonly


@dataclass
class Column:
    name: str
    type: str
    pk: bool = False
    fk: str | None = None  # "other_table.other_column"
    description: str = ""
    value_description: str = ""


@dataclass
class Table:
    name: str
    row_count: int
    columns: list[Column] = field(default_factory=list)

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name.lower() == name.lower()), None)


def _clean(s: str | None) -> str:
    return " ".join((s or "").split())


def _read_csv(path: Path) -> list[dict]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return list(csv.DictReader(io.StringIO(text)))


def _descriptions(desc_dir: Path) -> dict[str, dict[str, tuple[str, str]]]:
    """{table_lower: {column_lower: (description, value_description)}}"""
    out: dict[str, dict[str, tuple[str, str]]] = {}
    if not desc_dir.is_dir():
        return out
    for path in desc_dir.glob("*.csv"):
        cols = {}
        for row in _read_csv(path):
            name = _clean(row.get("original_column_name"))
            if not name:
                continue
            desc = _clean(row.get("column_description"))
            # column_name sometimes holds a longer, readable name
            expanded = _clean(row.get("column_name"))
            if expanded and expanded.lower() not in desc.lower():
                desc = f"{expanded}: {desc}" if desc else expanded
            cols[name.lower()] = (desc, _clean(row.get("value_description")))
        out[path.stem.strip().lower()] = cols
    return out


def load_schema(db_path: Path) -> dict[str, Table]:
    """Return {table_name: Table} with exact-case table names."""
    descs = _descriptions(db_path.parent / "database_description")
    conn = connect_readonly(db_path)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name")]
        tables = {}
        for t in names:
            q = t.replace('"', '""')
            n_rows = conn.execute(f'SELECT COUNT(*) FROM "{q}"').fetchone()[0]
            fks = {r[3]: f"{r[2]}.{r[4]}" for r in conn.execute(f'PRAGMA foreign_key_list("{q}")')}
            tdesc = descs.get(t.lower(), {})
            cols = []
            for _, name, ctype, _, _, pk in conn.execute(f'PRAGMA table_info("{q}")'):
                d, vd = tdesc.get(name.lower(), ("", ""))
                cols.append(Column(name, ctype or "", bool(pk), fks.get(name), d, vd))
            tables[t] = Table(t, n_rows, cols)
        return tables
    finally:
        conn.close()
