import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FX_TICKER = "KRW=X"
HISTORY_PATH = Path("data/fx_history.json")


def main():
    print("=== AURORA FX UPDATE SCRIPT v3 ===")

    # 1) KRW=X 1년치 다운로드
    df = yf.download(FX_TICKER, period="1y", progress=False)
    print("Downloaded type :", type(df))
    print("Downloaded cols :", df.columns)

    # 2) Close 시리즈만 안전하게 추출
    close = df["Close"]
    # 만약 여전히 DataFrame이면 첫 컬럼만 사용
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # Series 형태로 정리
    close = pd.Series(close).dropna().astype(float)

    # 3) 최근 130개만 사용
    hist = close.tail(130).tolist()

    # 4) fx_history.json 저장
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    # 5) 21일 sigma 계산
    sigma = None
    if len(hist) >= 22:
        arr = np.array(hist[-22:], dtype=float)
        logret = np.diff(np.log(arr))
        sigma = float(np.std(logret))

    print("[FX Update OK]")
    print("History length:", len(hist))
    print("Last FX       :", hist[-1] if hist else None)
    print("FX Vol 21d    :", sigma)


if __name__ == "__main__":
    main()
