import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Inputs (no raw_today)
FX_PATH = DATA_DIR / "fx_history.json"
FRED_PATH = DATA_DIR / "market_data_fred.json"

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def fetch_yahoo_history_csv(ticker: str, days: int = 60) -> pd.DataFrame:
    """
    Yahoo Finance deterministic historical CSV endpoint.
    We only use daily historical data (prev close).
    """
    end = int(time.time())
    start = end - days * 86400
    ticker_enc = requests.utils.quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker_enc}"
        f"?period1={start}&period2={end}&interval=1d&events=history&includeAdjustedClose=true"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(pd.io.common.StringIO(r.text))
    return df

def yahoo_prev_close(ticker: str, days: int = 90) -> float:
    df = fetch_yahoo_history_csv(ticker, days=days)
    df = df.dropna(subset=["Close"])
    if df.empty:
        raise RuntimeError(f"Yahoo history empty for {ticker}")
    return float(df["Close"].iloc[-1])

def yahoo_close_series(ticker: str, days: int) -> list[float]:
    df = fetch_yahoo_history_csv(ticker, days=max(days + 30, 200))
    df = df.dropna(subset=["Close"])
    closes = [float(x) for x in df["Close"].tolist()]
    if len(closes) < days:
        return closes
    return closes[-days:]

def fx_vol_21d_sigma(fx_hist: list[float]) -> float:
    """
    21D sigma of log returns, non-annualized. Clip [0, 0.05].
    Requires last 22 prices.
    """
    if len(fx_hist) < 22:
        return 0.0
    arr = np.array(fx_hist[-22:], dtype=float)
    arr = arr[arr > 0]
    if len(arr) < 22:
        return 0.0
    logret = np.diff(np.log(arr))
    sigma = float(np.std(logret))
    return float(np.clip(sigma, 0.0, 0.05))

def merge_fred_into_market(market: dict) -> dict:
    """
    market_data_fred.json 의 latest 블록을
    최종 market_data_YYYYMMDD.json 의 risk / rates / macro 섹션에 직접 주입한다.
    (기존 스키마 유지)
    """
    if not FRED_PATH.exists():
        print("[WARN] market_data_fred.json 가 없어 FRED merge 생략")
        return market

    fred = load_json(FRED_PATH)
    latest = fred.get("latest", {}) if isinstance(fred, dict) else {}

    # risk
    market["risk"]["hy_oas"] = latest.get("hy_oas_bps")

    # rates
    market["rates"]["dgs2"] = latest.get("dgs2")
    market["rates"]["dgs10"] = latest.get("dgs10")
    market["rates"]["ffr_upper"] = latest.get("ffr_upper")

    # macro
    market["macro"]["cpi_yoy"] = latest.get("cpi_yoy")
    market["macro"]["unemployment"] = latest.get("unemployment")
    market["macro"]["pmi_markit"] = latest.get("pmi_markit")

    return market

def main():
    print(f"[DEBUG] CWD={Path.cwd()}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not FX_PATH.exists():
        raise RuntimeError("fx_history.json missing. Update FX History step must run first.")

    fx_hist = load_json(FX_PATH)
    if not isinstance(fx_hist, list) or len(fx_hist) < 130:
        raise RuntimeError(f"fx_history.json insufficient. need>=130 got={len(fx_hist) if isinstance(fx_hist, list) else 'non-list'}")

    today = datetime.now().strftime("%Y%m%d")
    out_path = DATA_DIR / f"market_data_{today}.json"

    # Yahoo sources (replace raw_today)
    # SPX: use ^GSPC (prev close), and keep 3Y closes for drawdown calc if needed elsewhere
    spx_last = yahoo_prev_close("^GSPC", days=120)
    spx_3y_1095 = yahoo_close_series("^GSPC", days=1095)

    # VIX: ^VIX
    vix_last = yahoo_prev_close("^VIX", days=120)

    # ETFs (same bucket name as 기존 raw["etf"])
    etf = {
        "VOO": yahoo_prev_close("VOO", days=120),
        "QQQ": yahoo_prev_close("QQQ", days=120),
        "SCHD": yahoo_prev_close("SCHD", days=120),
        "SGOV": yahoo_prev_close("SGOV", days=120),
        "VWO": yahoo_prev_close("VWO", days=120),
        "XLE": yahoo_prev_close("XLE", days=120),
        "GLD": yahoo_prev_close("GLD", days=120),
    }

    # build (기존 스키마 유지)
    market = {
        "date": today,
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "fx": {
            "usdkrw": float(fx_hist[-1]),
            "usdkrw_history_130d": fx_hist[-130:],
            "usdkrw_history_21d": fx_hist[-21:],
            "fx_vol_21d_sigma": fx_vol_21d_sigma(fx_hist),
        },
        # 기존 코드의 "spx": raw["spx"] 자리에 단순 수치만 넣으면
        # 엔진이 dict를 기대할 수도 있으니, 안전하게 dict 형태로 제공
        "spx": {
            "last": float(spx_last),
            "closes_3y_1095": spx_3y_1095,
        },
        "risk": {"vix": float(vix_last), "hy_oas": None},
        "rates": {"dgs2": None, "dgs10": None, "ffr_upper": None},
        "macro": {"cpi_yoy": None, "unemployment": None, "pmi_markit": None},
        "etf": etf,
    }

    # FRED+PMI 값 삽입 (기존 방식 유지)
    market = merge_fred_into_market(market)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(market, f, indent=2, ensure_ascii=False)

    st = out_path.stat()
    print("[OK] market_data JSON created:", out_path)
    print(f"[DEBUG] size={st.st_size} usdkrw={market['fx']['usdkrw']} vix={market['risk']['vix']} spx={market['spx']['last']}")

if __name__ == "__main__":
    main()
