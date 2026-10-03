"""Base class for the systems (one-shot, single agent, multi-agent)."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sqlagent.llm import Meter


@dataclass
class Prediction:
    sql: str | None
    steps: int = 0  # agent turns; 1 for one-shot
    notes: str = ""  # e.g. why the agent stopped


# SQL rules shared by every system, so the comparison stays fair.
SQL_RULES = """\
- Select exactly the columns the question asks for, in the order it asks for them. \
Do not add extra columns (such as IDs or names) that were not requested.
- Follow the evidence: it defines terms, formulas and the exact values to filter on.
- Quote identifiers that contain spaces or special characters with backticks.
- SQLite has no YEAR(), MONTH() or DATE_FORMAT(). Use strftime, e.g. \
strftime('%Y', col) = '2014' (it returns text)."""


class System(ABC):
    name: str

    @abstractmethod
    async def answer(self, question: dict, meter: Meter) -> Prediction: pass

    async def aclose(self) -> None:
        """Release held resources, such as MCP servers. Most systems hold none."""
        return None


_SQL_REGEX = re.compile(r"```(?:sql|sqlite)?\s*\n(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str | None:
    blocks = _SQL_REGEX.findall(text)
    if blocks:
        return blocks[-1].strip().rstrip(";").strip() or None
    stripped = text.strip().rstrip(";").strip()
    return stripped if re.match(r"(?i)(select|with)\b", stripped) else None
