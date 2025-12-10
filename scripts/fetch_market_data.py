"""
AURORA Rev12.1b – Level-3 Market Data Fetch Script

역할:
- Yahoo Finance: FX / SPX / VIX / ETF 가격 & 히스토리
- FRED API: HY OAS / 2Y / 10Y / FFR / CPI / 실업률 / PMI 등
- 최종적으로 data/raw_today.json 생성
    → 이후 build_market_json.py 가 이 파일을 읽어서
      MarketDataSpec 규격의 market_data_YYYYMMDD.json 을 만든다.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
import yfinance as yf

OUTPUT_PATH = Path("data/raw_today.json")

# -------------------------------
# 1. Yahoo Finance helpers
# -------------------------------

def get_close_series(ticker: str, period: str = "1d") -> pd.Series:
    """
    Yahoo Finance에서 단일 티커의 종가(Previous Close) 시리즈만 안전하게 추출.
    - 항상 pd.Series(float) 로 반환.
    - MultiIndex / DataFrame 구조도 방어적으로 처리.
    """
    df = yf.download(ticker, period=period, progress=False)

    # DataFrame 형태로 들어오는 대부분의 경우
    if isinstance(df, pd.DataFrame):
        if "Close" in df.columns:
            close = df["Close"]
        else:
            # 혹시 Close 컬럼 이름이 다를 때 첫 컬럼 사용
            close = df.iloc[:, 0]
    else:
        close = pd.Series(df)

    # 혹시 Series가 아니라 DataFrame이면 다시 첫 컬럼만 사용
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.Series(close).dropna().astype(float)
    return close


# -------------------------------
# 2. FRED helpers
# -------------------------------

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def _fred_observations(
    series_id: str,
    api_key: str,
    observation_start: str = "2018-01-01",
) -> List[float]:
    """
    FRED series observations 전체를 float 리스트로 반환.
    누락 값(value='.') 은 제외.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
        "sort_order": "asc",
    }
    resp = requests.get(FRED_BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    obs = data.get("observations", [])
    values: List[float] = []

    for o in obs:
        v = o.get("value", ".")
        if v != ".":
            try:
                values.append(float(v))
            except ValueError:
                continue

    return values


def fred_latest(
    series_id: str,
    api_key: str,
    observation_start: str = "2018-01-01",
) -> Optional[float]:
    """
    FRED 시리즈의 가장 최근 실제 값 하나만 반환.
    값이 없으면 None.
    """
    try:
        values = _fred_observations(series_id, api_key, observation_start)
        if not values:
            return None
        return float(values[-1])
    except Exception as e:
        print(f"[WARN] FRED latest fetch failed for {series_id}: {e}")
        return None


def fred_series(
    series_id: str,
    api_key: str,
    observation_start: str = "2010-01-01",
) -> List[float]:
    """
    CPI YoY 계산 등에 사용할 전체 시리즈.
    """
    try:
        return _fred_observations(series_id, api_key, observation_start)
    except Exception as e:
        print(f"[WARN] FRED series fetch failed for {series_id}: {e}")
        return []


def fred_cpi_yoy(api_key: str) -> Optional[float]:
    """
    CPIAUCSL 지수로부터 YoY % 계산.
    - 월별 데이터 기준으로 최근 값 vs 12개월 전 값 사용.
    """
    series = fred_series("CPIAUCSL", api_key, observation_start="2010-01-01")
    if len(series) < 13:
        print("[WARN] Not enough CPI observations for YoY calculation.")
        return None

    latest = series[-1]
    prev_12m = series[-13]
    if prev_12m == 0:
        return None

    yoy = (latest / prev_12m - 1.0) * 100.0
    return float(yoy)


# -------------------------------
# 3. Main fetch function
# -------------------------------

def fetch_all() -> dict:
    data: dict = {}

    print("=== AURORA Rev12.1b – MARKET FETCH Level-3 ===")

    # ---------------- SPX (index + 3Y history) ----------------
    spx = get_close_series("^GSPC", period="5y")
    print("[SPX] history length:", len(spx))
    data["spx"] = {
        "index_level": float(spx.iloc[-1]),
        # 3년 ≒ 1095 거래일
        "history_3y": spx.tail(1095).tolist(),
    }

    # ---------------- Risk: VIX + HY OAS ----------------
    vix = get_close_series("^VIX", period="1y")

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("[WARN] FRED_API_KEY not found in environment. "
              "HY OAS / Rates / Macro will be set to None.")
        hy_oas = None
        dgs2 = dgs10 = ffr_upper = None
        ism_pmi = pmi_markit = cpi_yoy = unemployment = None
    else:
        # HY OAS (BAMLH0A0HYM2)
        hy_oas = fred_latest("BAMLH0A0HYM2", api_key, observation_start="2018-01-01")

        # ---------------- Rates ----------------
        dgs2 = fred_latest("DGS2", api_key, observation_start="2018-01-01")
        dgs10 = fred_latest("DGS10", api_key, observation_start="2018-01-01")
        ffr_upper = fred_latest("DFEDTARU", api_key, observation_start="2018-01-01")

        # ---------------- Macro ----------------
        # ISM Manufacturing PMI – FRED proxy: NAPM
        ism_pmi = fred_latest("NAPM", api_key, observation_start="2018-01-01")
        # Markit / S&P Global Manufacturing PMI – FRED proxy: PMIMAN (일반적으로 사용)
        pmi_markit = fred_latest("PMIMAN", api_key, observation_start="2018-01-01")
        # CPI YoY
        cpi_yoy = fred_cpi_yoy(api_key)
        # Unemployment rate
        unemployment = fred_latest("UNRATE", api_key, observation_start="2018-01-01")

    data["risk"] = {
        "vix": float(vix.iloc[-1]),
        "hy_oas": hy_oas,
    }

    data["rates"] = {
        "dgs2": dgs2,
        "dgs10": dgs10,
        "ffr_upper": ffr_upper,
    }

    data["macro"] = {
        "ism_pmi": ism_pmi,
        "pmi_markit": pmi_markit,
        "cpi_yoy": cpi_yoy,
        "unemployment": unemployment,
    }

    # ---------------- ETF prices ----------------
    etfs = ["VOO", "QQQ", "SCHD", "SGOV", "VWO", "XLE", "GLD", "GLDM"]
    etf_data = {}
    for t in etfs:
        s = get_close_series(t, period="1y")
        etf_data[t] = float(s.iloc[-1])
    data["etf"] = etf_data

    return data


def main():
    data = fetch_all()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("[OK] raw_today.json created at", OUTPUT_PATH)


if __name__ == "__main__":
    main()
