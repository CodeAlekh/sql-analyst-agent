import argparse

from sqlagent.mcp_server.server import build_server
from sqlagent.paths import db_path

parser = argparse.ArgumentParser(description="BIRD database tools over MCP (stdio)")
parser.add_argument("--db-id", required=True, help="BIRD db_id, e.g. california_schools")
args = parser.parse_args()
build_server(db_path(args.db_id)).run()
