import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FX_PATH = DATA_DIR / "fx_history.json"
FRED_PATH = DATA_DIR / "market_data_fred.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _normalize_yf_download(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance download() 결과를 멀티티커 기준으로 표준화:
    - columns: (Field, Ticker) or (Ticker, Field) 케이스 모두 대응
    """
    if df is None or df.empty:
        raise RuntimeError("yfinance download returned empty dataframe")

    # yfinance는 보통 columns가 MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # 케이스1: (PriceField, Ticker)
        if df.columns.names and df.columns.names[0] in ("Price", None):
            # 강제 변환은 어려우니 아래 접근 로직에서 처리
            return df
        return df

    # 단일 티커면 columns가 단일 인덱스일 수 있음
    return df


def yf_last_close_from_bulk(df: pd.DataFrame, ticker: str) -> float:
    """
    bulk df에서 ticker의 마지막 Close를 뽑는다.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # 일반적으로 ('Close', 'VOO') 형태가 많음
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)].dropna()
            if s.empty:
                raise RuntimeError(f"Close empty for {ticker}")
            return float(s.iloc[-1])

        # 혹시 (ticker, 'Close') 형태
        if (ticker, "Close") in df.columns:
            s = df[(ticker, "Close")].dropna()
            if s.empty:
                raise RuntimeError(f"Close empty for {ticker}")
            return float(s.iloc[-1])

        # 다른 형태면 마지막 수단으로 컬럼 탐색
        closes = [c for c in df.columns if (isinstance(c, tuple) and "Close" in c and ticker in c)]
        if closes:
            s = df[closes[0]].dropna()
            if s.empty:
                raise RuntimeError(f"Close empty for {ticker}")
            return float(s.iloc[-1])

        raise RuntimeError(f"Cannot locate Close for {ticker} in yfinance bulk frame")

    # 단일 티커 dataframe
    if "Close" not in df.columns:
        raise RuntimeError("Close column missing (single ticker)")
    s = df["Close"].dropna()
    if s.empty:
        raise RuntimeError("Close empty (single ticker)")
    return float(s.iloc[-1])


def yf_close_series_single(ticker: str, period: str, n: int) -> list[float]:
    """
    3Y drawdown용 close series. (index는 bulk에서 잘 안 나오는 경우 있어 단독 요청)
    """
    df = yf.download(ticker, period=period, progress=False)
    if df is None or df.empty or "Close" not in df:
        raise RuntimeError(f"yfinance empty for {ticker} series")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.Series(close).dropna().astype(float)
    closes = close.tolist()
    if len(closes) < n:
        return closes
    return closes[-n:]


def main():
    print(f"[DEBUG] CWD={Path.cwd()}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not FX_PATH.exists():
        raise RuntimeError("fx_history.json missing. Update FX History step must run first.")

    fx_hist = load_json(FX_PATH)
    if not isinstance(fx_hist, list) or len(fx_hist) < 130:
        raise RuntimeError(f"fx_history.json insufficient. need>=130 got={len(fx_hist) if isinstance(fx_hist, list) else 'non-list'}")

    if not FRED_PATH.exists():
        raise RuntimeError("market_data_fred.json missing. Fetch Market Data step must succeed before build.")

    today = datetime.now().strftime("%Y%m%d")
    out_path = DATA_DIR / f"market_data_{today}.json"

    # ===== yfinance only (Yahoo CSV endpoint 제거) =====
    # 1) bulk download for ETF + VIX (reduce requests)
    tickers_bulk = ["^VIX", "VOO", "QQQ", "SCHD", "SGOV", "VWO", "XLE", "GLD"]
    df_bulk = yf.download(tickers_bulk, period="1y", progress=False, group_by="column")
    df_bulk = _normalize_yf_download(df_bulk)

    vix_last = yf_last_close_from_bulk(df_bulk, "^VIX")

    etf = {
        "VOO": yf_last_close_from_bulk(df_bulk, "VOO"),
        "QQQ": yf_last_close_from_bulk(df_bulk, "QQQ"),
        "SCHD": yf_last_close_from_bulk(df_bulk, "SCHD"),
        "SGOV": yf_last_close_from_bulk(df_bulk, "SGOV"),
        "VWO": yf_last_close_from_bulk(df_bulk, "VWO"),
        "XLE": yf_last_close_from_bulk(df_bulk, "XLE"),
        "GLD": yf_last_close_from_bulk(df_bulk, "GLD"),
    }

    # 2) SPX last + 3y series (index는 별도 요청이 안정적)
    spx_series_1095 = yf_close_series_single("^GSPC", period="6y", n=1095)
    if len(spx_series_1095) == 0:
        raise RuntimeError("SPX series empty (^GSPC)")
    spx_last = float(spx_series_1095[-1])

    # ===== build output (기존 스키마 유지) =====
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
            "closes_3y_1095": spx_series_1095,
        },
        "risk": {"vix": float(vix_last), "hy_oas": None},
        "rates": {"dgs2": None, "dgs10": None, "ffr_upper": None},
        "macro": {"cpi_yoy": None, "unemployment": None, "pmi_markit": None},
        "etf": etf,
    }

    # FRED+PMI merge (필수)
    market = merge_fred_into_market(market)

    # 필수값 검증 (fail-fast)
    if market["risk"]["vix"] is None:
        raise RuntimeError("VIX missing")
    if market["risk"]["hy_oas"] is None:
        raise RuntimeError("HY OAS missing (from FRED latest)")
    if market["rates"]["dgs2"] is None or market["rates"]["dgs10"] is None or market["rates"]["ffr_upper"] is None:
        raise RuntimeError("Rates missing (DGS2/DGS10/FFR)")
    if market["macro"]["cpi_yoy"] is None or market["macro"]["unemployment"] is None or market["macro"]["pmi_markit"] is None:
        raise RuntimeError("Macro missing (CPI/UNEMP/PMI)")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(market, f, indent=2, ensure_ascii=False)

    st = out_path.stat()
    print("[OK] market_data JSON created:", out_path)
    print(f"[DEBUG] size={st.st_size} usdkrw={market['fx']['usdkrw']} vix={market['risk']['vix']} spx={market['spx']['last']}")


if __name__ == "__main__":
    main()
