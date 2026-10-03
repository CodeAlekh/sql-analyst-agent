import sqlite3

import pytest


@pytest.fixture
def fixture_db(tmp_path):
    """A tiny BIRD-shaped database: two tables, an FK, a column with spaces, and
    description CSVs in the two encodings BIRD actually ships (UTF-8 BOM, Latin-1)."""
    d = tmp_path / "shop"
    d.mkdir()
    path = d / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT, "Home Country" TEXT);
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customer(id),
            amount REAL,
            note TEXT
        );
        INSERT INTO customer VALUES (1, 'Ann', 'USA'), (2, 'Bob', 'US'), (3, 'Cé', 'Nepal');
        INSERT INTO orders VALUES
            (10, 1, 5.0, 'first'), (11, 1, 7.5, NULL), (12, 2, 2.0, 'x'),
            (13, 3, 1.0, 'a very long note that goes on and on and on and on');
    """)
    conn.commit()
    conn.close()
    desc = d / "database_description"
    desc.mkdir()
    (desc / "customer.csv").write_bytes(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "Home Country,,country of residence,text,ISO-ish names\n".encode("utf-8-sig"))
    (desc / "orders.csv").write_bytes(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "amount,,order total in euros (€ not included),real,\n".encode("latin-1", "replace"))
    return path
