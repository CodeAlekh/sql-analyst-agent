"""Append-only JSONL log of every LLM and tool call."""

import json
import time
from pathlib import Path


class Tracer:
    def __init__(self, path: Path, run_id: str, system: str):
        self.path, self.run_id, self.system = path, run_id, system
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **event) -> None:
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "system": self.system, **event}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
