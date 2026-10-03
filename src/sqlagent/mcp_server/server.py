"""MCP server for one BIRD database: python -m sqlagent.mcp_server --db-id <db_id>"""

import functools
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from sqlagent.tools import DatabaseTools, ToolInputError


# MCP hides the message of any exception except ToolError, so convert ours.
def _actionable(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolInputError as e:
            raise ToolError(str(e)) from e

    return wrapper


def build_server(db_path: Path, **tool_kwargs) -> MCPServer:
    tools = DatabaseTools(db_path, **tool_kwargs)
    server = MCPServer(f"bird-{Path(db_path).stem}")

    @server.tool()
    @_actionable
    def list_tables() -> str:
        """List every table with its row count and column names. Start here."""
        return tools.list_tables()

    @server.tool()
    @_actionable
    def describe_table(table: str) -> str:
        """Show a table's columns (type, primary/foreign keys, human-written descriptions
        and value meanings) plus 3 example rows."""
        return tools.describe_table(table)

    @server.tool()
    @_actionable
    def search_schema(keyword: str, limit: int = 10) -> str:
        """Find columns whose name or description matches a keyword (e.g. 'free meal',
        'county'). Use it to locate where a concept from the question is stored."""
        return tools.search_schema(keyword, limit)

    @server.tool()
    @_actionable
    def sample_values(table: str, column: str, limit: int = 10) -> str:
        """Show a few distinct non-NULL values of a column, to learn its format."""
        return tools.sample_values(table, column, limit)

    @server.tool()
    @_actionable
    def distinct_values(table: str, column: str, like: str | None = None, limit: int = 20) -> str:
        """Most frequent values of a column with counts, optionally filtered by a SQL LIKE
        pattern (e.g. like='%alameda%'). Use it to find the exact spelling of a value
        before filtering on it."""
        return tools.distinct_values(table, column, like, limit)

    @server.tool()
    @_actionable
    def run_query(sql: str, max_rows: int = 50) -> str:
        """Execute one read-only SQLite SELECT and return the row count and the first rows.
        Quote identifiers containing spaces with double quotes or backticks."""
        return tools.run_query(sql, max_rows)

    return server
