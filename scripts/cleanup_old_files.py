from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import re

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")

def _parse_yyyymmdd_from_name(name: str) -> str | None:
    m = re.search(r"(\d{8})", name)
    return m.group(1) if m else None

def _is_older_than(file_path: Path, cutoff: datetime) -> bool:
    yyyymmdd = _parse_yyyymmdd_from_name(file_path.name)
    if not yyyymmdd:
        return False
    try:
        dt = datetime.strptime(yyyymmdd, "%Y%m%d")
        return dt < cutoff
    except ValueError:
        return False

def cleanup(days: int = 365) -> dict:
    cutoff = datetime.utcnow() - timedelta(days=days)
    deleted = []

    # market_data_*.json delete >365d
    for p in DATA_DIR.glob("market_data_*.json"):
        if _is_older_than(p, cutoff):
            p.unlink(missing_ok=True)
            deleted.append(str(p))

    # reports/*.md delete >365d
    if REPORTS_DIR.exists():
        for p in REPORTS_DIR.glob("*.md"):
            if _is_older_than(p, cutoff):
                p.unlink(missing_ok=True)
                deleted.append(str(p))

    return {"cutoff_utc": cutoff.isoformat(), "deleted": deleted}

if __name__ == "__main__":
    print(cleanup(days=365))
