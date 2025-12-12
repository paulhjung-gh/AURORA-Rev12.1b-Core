# scripts/cleanup_artifacts.py
import re
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta, date
import calendar

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

RE_REPORT = re.compile(r"^aurora_daily_report_(\d{8})\.md$")
RE_MARKET = re.compile(r"^market_data_(\d{8})\.json$")


def _parse_yyyymmdd(s: str) -> date | None:
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except Exception:
        return None


def _is_month_end(d: date) -> bool:
    last_day = calendar.monthrange(d.year, d.month)[1]
    return d.day == last_day


def _is_month_start(d: date) -> bool:
    return d.day == 1


def cleanup_reports(retain_days: int = 60) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = date.today() - timedelta(days=retain_days)

    deleted = []
    kept = []
    scanned = 0

    for p in sorted(REPORTS_DIR.glob("aurora_daily_report_*.md")):
        scanned += 1
        m = RE_REPORT.match(p.name)
        if not m:
            kept.append(p.name)
            continue

        d = _parse_yyyymmdd(m.group(1))
        if d is None:
            kept.append(p.name)
            continue

        # keep if within window OR month boundary (start/end)
        if d >= cutoff or _is_month_start(d) or _is_month_end(d):
            kept.append(p.name)
            continue

        p.unlink(missing_ok=True)
        deleted.append(p.name)

    return {
        "reports": {
            "retain_days": retain_days,
            "cutoff_yyyymmdd": cutoff.strftime("%Y%m%d"),
            "scanned": scanned,
            "deleted": deleted,
            "kept_count": len(kept),
        }
    }


def cleanup_market_data(retain_days: int = 14) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = date.today() - timedelta(days=retain_days)

    deleted = []
    kept = []
    scanned = 0

    for p in sorted(DATA_DIR.glob("market_data_*.json")):
        scanned += 1
        m = RE_MARKET.match(p.name)
        if not m:
            kept.append(p.name)
            continue

        d = _parse_yyyymmdd(m.group(1))
        if d is None:
            kept.append(p.name)
            continue

        if d >= cutoff:
            kept.append(p.name)
            continue

        p.unlink(missing_ok=True)
        deleted.append(p.name)

    return {
        "market_data": {
            "retain_days": retain_days,
            "cutoff_yyyymmdd": cutoff.strftime("%Y%m%d"),
            "scanned": scanned,
            "deleted": deleted,
            "kept_count": len(kept),
        }
    }


def main() -> int:
    rep = cleanup_reports(retain_days=60)
    mkt = cleanup_market_data(retain_days=14)

    summary = {"date_yyyymmdd": date.today().strftime("%Y%m%d"), **rep, **mkt}
    print("[CLEANUP] Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
