import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FX_TICKER = "KRW=X"

ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "data" / "fx_history.json"


def _download_fx_csv_last_1y(ticker: str) -> pd.DataFrame:
    """
    Yahoo Finance deterministic historical CSV endpoint.
    Spec/Guide에서 권장하는 방식. (1y 범위에서 trading days 충분히 확보 후 tail(130))
    """
    end = int(time.time())
    start = end - 365 * 86400
    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        f"?period1={start}&period2={end}&interval=1d&events=history&includeAdjustedClose=true"
    )
    df = pd.read_csv(url)
    return df


def _download_fx_yfinance(ticker: str) -> pd.DataFrame:
    """Fallback only: yfinance (환경에 따라 empty/blocked 가능)"""
    df = yf.download(ticker, period="1y", progress=False)
    return df


def main():
    print("=== AURORA FX UPDATE SCRIPT v4 (CSV primary, yfinance fallback) ===")
    print("[INFO] target:", FX_TICKER)
    print("[INFO] output:", str(HISTORY_PATH))

    df = None
    source = "yahoo_csv"

    # 1) Primary: Yahoo CSV endpoint (deterministic)
    try:
        df = _download_fx_csv_last_1y(FX_TICKER)
        # CSV는 컬럼이 보통: Date, Open, High, Low, Close, Adj Close, Volume
        if df is None or df.empty or "Close" not in df.columns:
            raise RuntimeError("CSV download returned empty or missing Close")
        close = df["Close"]
    except Exception as e:
        print("[WARN] yahoo_csv failed -> fallback to yfinance. reason:", repr(e))
        source = "yfinance"

        # 2) Fallback: yfinance
        df = _download_fx_yfinance(FX_TICKER)
        # yfinance는 멀티인덱스 컬럼이거나 "Close"가 없을 수 있음
        if df is None or df.empty or "Close" not in df:
            raise RuntimeError("KRW=X download failed or empty (yfinance)")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

    # 3) Clean series
    close = pd.Series(close).dropna().astype(float)

    # 4) Last 130 trading days (KDE anchor input requirement)
    hist = close.tail(130).tolist()
    if len(hist) < 130:
        raise RuntimeError(f"FX history insufficient: got {len(hist)} (<130). source={source}")

    # 5) Write fx_history.json
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)

    st = HISTORY_PATH.stat()
    print(f"[OK] wrote {HISTORY_PATH} size={st.st_size} bytes source={source}")
    print("[OK] history_len:", len(hist))
    print("[OK] last_fx:", hist[-1])

    # 6) 21D sigma (non-annualized) - debug print only
    sigma = None
    if len(hist) >= 22:
        arr = np.array(hist[-22:], dtype=float)
        logret = np.diff(np.log(arr))
        sigma = float(np.std(logret))
    print("[OK] fx_vol_21d_sigma:", sigma)


if __name__ == "__main__":
    main()
