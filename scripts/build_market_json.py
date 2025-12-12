import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FX_PATH = DATA_DIR / "fx_history.json"
FRED_PATH = DATA_DIR / "market_data_fred.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Yahoo CSV (primary) with retry/backoff ----------
def fetch_yahoo_history_csv(ticker: str, days: int = 200, retries: int = 4) -> pd.DataFrame:
    """
    Yahoo Finance deterministic historical CSV endpoint.
    BUT: GitHub Actions 환경에서 429가 자주 떠서, retry+backoff + user-agent 사용.
    """
    end = int(time.time())
    start = end - days * 86400

    ticker_enc = requests.utils.quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker_enc}"
        f"?period1={start}&period2={end}&interval=1d&events=history&includeAdjustedClose=true"
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AURORA/1.0; +https://github.com/paulhjung-gh/AURORA-Rev12.1b-Core)"
    }

    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=25)
            r.raise_for_status()
            df = pd.read_csv(pd.io.common.StringIO(r.text))
            return df
        except Exception as e:
            last_err = e
            # exponential backoff: 2,4,8,16 sec
            sleep_s = 2 ** (i + 1)
            print(f"[WARN] yahoo_csv failed ({ticker}) attempt={i+1}/{retries} -> sleep {sleep_s}s. reason={repr(e)}")
            time.sleep(sleep_s)

    raise RuntimeError(f"Yahoo CSV failed after retries for {ticker}. last_err={repr(last_err)}")


# ---------- yfinance fallback ----------
def yf_prev_close(ticker: str, period: str = "6mo") -> float:
    df = yf.download(ticker, period=period, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance empty for {ticker}")
    # Close may be Series or DataFrame
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.Series(close).dropna()
    if close.empty:
        raise RuntimeError(f"yfinance Close empty for {ticker}")
    return float(close.iloc[-1])


def yf_close_series(ticker: str, period: str = "6y", n: int = 1095) -> list[float]:
    df = yf.download(ticker, period=period, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance empty for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.Series(close).dropna().astype(float)
    closes = close.tolist()
    if len(closes) < n:
        return closes
    return closes[-n:]


# ---------- Unified getters: try Yahoo CSV then fallback to yfinance ----------
def prev_close(ticker: str, days: int = 200) -> float:
    try:
        df = fetch_yahoo_history_csv(ticker, days=days)
        df = df.dropna(subset=["Close"])
        if df.empty:
            raise RuntimeError("CSV Close empty")
        return float(df["Close"].iloc[-1])
    except Exception as e:
        print(f"[WARN] prev_close yahoo_csv failed -> fallback yfinance ({ticker}). reason={repr(e)}")
        return yf_prev_close(ticker, period="1y")


def close_series(ticker: str, n: int = 1095, days: int = 2200) -> list[float]:
    """
    Need 1095 closes (~3y trading days). Yahoo CSV may 429 -> fallback yfinance.
    """
    try:
        df = fetch_yahoo_history_csv(ticker, days=days)
        df = df.dropna(subset=["Close"])
        closes = [float(x) for x in df["Close"].tolist()]
        if len(closes) < n:
            return closes
        return closes[-n:]
    except Exception as e:
        print(f"[WARN] close_series yahoo_csv failed -> fallback yfinance ({ticker}). reason={repr(e)}")
        return yf_close_series(ticker, period="6y", n=n)


# ---------- FXVol (21D log-return sigma, non-annualized) ----------
def fx_vol_21d_sigma(fx_hist: list[float]) -> float:
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

    market["risk"]["hy_oas"] = latest.get("hy_oas_bps")

    market["rates"]["dgs2"] = latest.get("dgs2")
    market["rates"]["dgs10"] = latest.get("dgs10")
    market["rates"]["ffr_upper"] = latest.get("ffr_upper")

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

    # --- Yahoo primary + yfinance fallback (429-safe) ---
    spx_last = prev_close("^GSPC", days=2200)
    spx_3y_1095 = close_series("^GSPC", n=1095, days=2600)

    vix_last = prev_close("^VIX", days=400)

    etf = {
        "VOO": prev_close("VOO", days=400),
        "QQQ": prev_close("QQQ", days=400),
        "SCHD": prev_close("SCHD", days=400),
        "SGOV": prev_close("SGOV", days=400),
        "VWO": prev_close("VWO", days=400),
        "XLE": prev_close("XLE", days=400),
        "GLD": prev_close("GLD", days=400),
    }

    market = {
        "date": today,
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "fx": {
            "usdkrw": float(fx_hist[-1]),
            "usdkrw_history_130d": fx_hist[-130:],
            "usdkrw_history_21d": fx_hist[-21:],
            "fx_vol_21d_sigma": fx_vol_21d_sigma(fx_hist),
        },
        "spx": {
            "last": float(spx_last),
            "closes_3y_1095": spx_3y_1095,
        },
        "risk": {"vix": float(vix_last), "hy_oas": None},
        "rates": {"dgs2": None, "dgs10": None, "ffr_upper": None},
        "macro": {"cpi_yoy": None, "unemployment": None, "pmi_markit": None},
        "etf": etf,
    }

    market = merge_fred_into_market(market)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(market, f, indent=2, ensure_ascii=False)

    st = out_path.stat()
    print("[OK] market_data JSON created:", out_path)
    print(f"[DEBUG] size={st.st_size} usdkrw={market['fx']['usdkrw']} vix={market['risk']['vix']} spx={market['spx']['last']}")


if __name__ == "__main__":
    main()
