# scripts/cleanup_artifacts.py
import re
import json
from pathlib import Path
from datetime import datetime, timedelta, date
import calendar

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

RE_REPORT = re.compile(r"^aurora_daily_report_(\d{8})\.md$")

RE_MARKET = re.compile(r"^market_data_(\d{8})\.json$")
RE_TARGET_WEIGHTS = re.compile(r"^aurora_target_weights_(\d{8})\.json$")
RE_CMA_STATE = re.compile(r"^cma_state_(\d{8})\.json$")


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


def _cleanup_by_regex(
    directory: Path,
    regex: re.Pattern,
    retain_days: int,
    keep_month_boundary: bool = False,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    cutoff = date.today() - timedelta(days=retain_days)

    deleted = []
    kept = []
    scanned = 0

    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        scanned += 1

        m = regex.match(p.name)
        if not m:
            kept.append(p.name)
            continue

        d = _parse_yyyymmdd(m.group(1))
        if d is None:
            kept.append(p.name)
            continue

        # keep if within window OR (optional) month boundary
        if d >= cutoff:
            kept.append(p.name)
            continue
        if keep_month_boundary and (_is_month_start(d) or _is_month_end(d)):
            kept.append(p.name)
            continue

        p.unlink(missing_ok=True)
        deleted.append(p.name)

    return {
        "retain_days": retain_days,
        "cutoff_yyyymmdd": cutoff.strftime("%Y%m%d"),
        "scanned": scanned,
        "deleted": deleted,
        "kept_count": len(kept),
    }


def main() -> int:
    # Reports: 60 days + keep month start/end
    reports_summary = _cleanup_by_regex(
        directory=REPORTS_DIR,
        regex=RE_REPORT,
        retain_days=60,
        keep_month_boundary=True,
    )

    # Data artifacts: 14 days only (market/target_weights/cma_state)
    market_summary = _cleanup_by_regex(
        directory=DATA_DIR,
        regex=RE_MARKET,
        retain_days=14,
        keep_month_boundary=False,
    )
    target_weights_summary = _cleanup_by_regex(
        directory=DATA_DIR,
        regex=RE_TARGET_WEIGHTS,
        retain_days=14,
        keep_month_boundary=False,
    )
    cma_state_summary = _cleanup_by_regex(
        directory=DATA_DIR,
        regex=RE_CMA_STATE,
        retain_days=14,
        keep_month_boundary=False,
    )

    summary = {
        "date_yyyymmdd": date.today().strftime("%Y%m%d"),
        "reports": reports_summary,
        "market_data": market_summary,
        "target_weights": target_weights_summary,
        "cma_state": cma_state_summary,
    }

    print("[CLEANUP] Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
