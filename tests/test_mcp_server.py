from mcp import Client

from sqlagent.mcp_server.server import build_server


async def test_tools_are_exposed_with_schemas(fixture_db):
    async with Client(build_server(fixture_db)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"list_tables", "describe_table", "search_schema",
                          "sample_values", "distinct_values", "run_query"}
    assert tools["run_query"].input_schema["required"] == ["sql"]
    assert tools["describe_table"].description


async def test_call_returns_text_and_errors_stay_actionable(fixture_db):
    async with Client(build_server(fixture_db)) as client:
        ok = await client.call_tool("run_query", {"sql": "SELECT COUNT(*) AS n FROM orders"})
        bad = await client.call_tool("describe_table", {"table": "customers"})
    assert not ok.is_error and "n\n4" in ok.content[0].text
    assert bad.is_error and "Did you mean: customer" in bad.content[0].text
