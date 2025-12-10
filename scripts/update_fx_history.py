import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FX_TICKER = "KRW=X"
HISTORY_PATH = Path("data/fx_history.json")


def load_history():
    if HISTORY_PATH.exists():
        with HISTORY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(hist):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)


def _get_close_series(ticker: str, period: str) -> pd.Series:
    df = yf.download(ticker, period=period, progress=False)

    if isinstance(df, pd.DataFrame):
        if "Close" in df.columns:
            close = df["Close"]
        else:
            close = df.iloc[:, 0]
    else:
        close = pd.Series(df)

    close = close.dropna().astype(float)
    return close


def bootstrap_history():
    close = _get_close_series(FX_TICKER, period="1y")
    return close.tail(130).tolist()


def fetch_latest_fx():
    close = _get_close_series(FX_TICKER, period="5d")
    return float(close.iloc[-1])


def update_fx():
    hist = load_history()

    if len(hist) < 130:
        hist = bootstrap_history()

    new_val = fetch_latest_fx()
    hist.append(new_val)
    hist = hist[-130:]

    save_history(hist)

    sigma = None
    if len(hist) >= 22:
        arr = np.array(hist[-22:], dtype=float)
        logret = np.diff(np.log(arr))
        sigma = float(np.std(logret))

    return hist, new_val, sigma


if __name__ == "__main__":
    hist, new_fx, sigma = update_fx()
    print("[FX Update OK]")
    print("Latest FX:", new_fx)
    print("FX Vol 21d:", sigma)
    print("History length:", len(hist))
