import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FX_TICKER = "KRW=X"
HISTORY_PATH = Path("data/fx_history.json")


def load_history():
    """기존 fx_history.json 읽기 (없으면 빈 리스트)."""
    if HISTORY_PATH.exists():
        with HISTORY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(hist):
    """fx_history.json 저장."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)


def _get_close_series(ticker: str, period: str) -> pd.Series:
    """
    yfinance 결과에서 'Close' 시리즈만 안전하게 뽑아오는 헬퍼.
    단일 티커 기준.
    """
    df = yf.download(ticker, period=period, progress=False)

    # df 가 MultiIndex 이거나 DataFrame 이더라도 Close 컬럼만 Series 로 뽑기
    if isinstance(df, pd.DataFrame):
        if "Close" in df.columns:
            close = df["Close"]
        else:
            # 혹시라도 첫 번째 컬럼만 사용
            close = df.iloc[:, 0]
    else:
        # 이 경우는 거의 없지만 방어적으로 처리
        close = pd.Series(df)

    close = close.dropna().astype(float)
    return close


def bootstrap_history():
    """최초 1회: 130일 환율 기록 확보."""
    close = _get_close_series(FX_TICKER, period="1y")
    # 최근 130개만 사용
    return close.tail(130).tolist()


def fetch_latest_fx():
    """가장 최근 1일 종가만 가져오기."""
    close = _get_close_series(FX_TICKER, period="5d")
    return float(close.iloc[-1])


def update_fx():
    """fx_history.json 을 130일 rolling 으로 업데이트."""
    hist = load_history()

    # 초기 세팅: 기록이 130개 미만이면 bootstrap
    if len(hist) < 130:
        hist = bootstrap_history()

    # 새로 1일치 추가
    new_val = fetch_latest_fx()
    hist.append(new_val)

    # 항상 130개만 유지
    hist = hist[-130:]

    save_history(hist)

    # 21일 σ 계산
    sigma = None
    if len(hist) >= 22:
        prices = np.array(hist[-22:], dtype=float)
        logret = np.diff(np.log(prices))
        sigma = float(np.std(logret))

    return hist, new_val, sigma


if __name__ == "__main__":
    hist, new_fx, sigma = update_fx()
    print("[FX Update OK]")
    print("Latest FX:", new_fx)
    print("FX Vol 21d:", sigma)
    print("History length:", len(hist))
