import json
from pathlib import Path

import pandas as pd
import yfinance as yf

OUTPUT_PATH = Path("data/raw_today.json")


def get_close_series(ticker: str, period: str = "1d") -> pd.Series:
    """
    yfinance 결과에서 'Close'만 안전하게 Series로 뽑아서 반환.
    단일 티커 기준 (SPX, VIX, ETF 등 공통 사용).
    """
    df = yf.download(ticker, period=period, progress=False)

    # MultiIndex / DataFrame 모두 커버
    if isinstance(df, pd.DataFrame):
        if "Close" in df.columns:
            close = df["Close"]
        else:
            close = df.iloc[:, 0]
    else:
        close = pd.Series(df)

    # Close가 DataFrame이면 첫 컬럼만 사용
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.Series(close).dropna().astype(float)
    return close


def fetch_all() -> dict:
    data: dict = {}

    print("=== AURORA FETCH MARKET v2 ===")

    # ========== SPX ==========
    spx = get_close_series("^GSPC", period="5y")
    print("[SPX] len:", len(spx))
    data["spx"] = {
        "index_level": float(spx.iloc[-1]),
        "history_3y": spx.tail(1095).tolist(),  # 이제 spx 는 Series라 안전
    }

    # ========== Risk ==========
    vix = get_close_series("^VIX", period="1y")
    data["risk"] = {
        "vix": float(vix.iloc[-1]),
        "hy_oas": None,  # FRED API 연동 전까지는 placeholder
    }

    # ========== Rates (placeholder) ==========
    data["rates"] = {
        "dgs2": None,
        "dgs10": None,
        "ffr_upper": None,
    }

    # ========== Macro (placeholder) ==========
    data["macro"] = {
        "ism_pmi": None,
        "pmi_markit": None,
        "cpi_yoy": None,
        "unemployment": None,
    }

    # ========== ETF Prices ==========
