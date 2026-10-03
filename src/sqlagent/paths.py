from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
BIRD = DATA / "bird"
DB_DIR = BIRD / "dev_databases"
QUESTIONS = BIRD / "dev_20251106.json"
SPLITS = DATA / "splits"


def db_path(db_id: str) -> Path:
    return DB_DIR / db_id / f"{db_id}.sqlite"
