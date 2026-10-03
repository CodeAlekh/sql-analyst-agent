"""Download BIRD New Dev (20251106) questions from Hugging Face and the dev databases
from the official dev.zip (New Dev reuses them). Skips files already present."""

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from sqlagent.paths import BIRD, DB_DIR, QUESTIONS, RAW

QUESTIONS_URL = (
    "https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106/resolve/main/"
    "data/dev_20251106-00000-of-00001.json"
)
DEV_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
DEV_ZIP_SHA256 = "cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630"
EXPECTED_DBS = 11
EXPECTED_QUESTIONS = 1534


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


def fetch_questions() -> None:
    if not QUESTIONS.exists():
        download(QUESTIONS_URL, QUESTIONS)
    rows = json.loads(QUESTIONS.read_text())
    assert len(rows) == EXPECTED_QUESTIONS, f"expected {EXPECTED_QUESTIONS} questions, got {len(rows)}"
    print(f"questions ok: {len(rows)}")


def fetch_databases() -> None:
    zip_path = RAW / "dev.zip"
    if not zip_path.exists():
        download(DEV_ZIP_URL, zip_path)
    digest = sha256(zip_path)
    if digest != DEV_ZIP_SHA256:
        raise SystemExit(f"dev.zip checksum mismatch: {digest}")

    if not DB_DIR.exists() or len(list(DB_DIR.glob("*/*.sqlite"))) < EXPECTED_DBS:
        with zipfile.ZipFile(zip_path) as outer:
            inner_name = next(n for n in outer.namelist() if n.endswith("dev_databases.zip"))
            tables_name = next(n for n in outer.namelist() if n.endswith("dev_tables.json"))
            BIRD.mkdir(parents=True, exist_ok=True)
            (BIRD / "dev_tables.json").write_bytes(outer.read(tables_name))
            print(f"extracting {inner_name}")
            with outer.open(inner_name) as inner_file, zipfile.ZipFile(inner_file) as inner:
                for member in inner.infolist():
                    name = member.filename
                    # macOS junk files inside the zip
                    if "__MACOSX" in name or Path(name).name.startswith("._"):
                        continue
                    inner.extract(member, BIRD)
    dbs = sorted(DB_DIR.glob("*/*.sqlite"))
    assert len(dbs) == EXPECTED_DBS, f"expected {EXPECTED_DBS} databases, found {len(dbs)}"
    print(f"databases ok: {len(dbs)} ({', '.join(p.stem for p in dbs)})")


if __name__ == "__main__":
    fetch_questions()
    fetch_databases()
