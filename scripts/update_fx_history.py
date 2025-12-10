import json
from pathlib import Path
import yfinance as yf
import numpy as np

FX_TICKER = "KRW=X"
HISTORY_PATH = Path("data/fx_history.json")

def load_history():
    if HISTORY_PATH.exists():
        with HISTORY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(hist):
    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

def fetch_latest_fx():
    data = yf.download(FX_TICKER, period="1d", progress=False)["Close"]
    return float(data.iloc[-1])

def bootstrap_history():
    data = yf.download(FX_TICKER, period="1y", progress=False)["Close"]
    return data.tail(130).tolist()

def update_fx():
    hist = load_history()

    # 초기 세팅
    if len(hist) < 130:
        hist = bootstrap_history()

    # 새로 1일치 추가
    new_val = fetch_latest_fx()
    hist.append(new_val)

    # 130일 유지
    hist = hist[-130:]

    save_history(hist)

    # 21일 σ
    if len(hist) >= 22:
        prices = np.array(hist[-22:])
        logret = np.diff(np.log(prices))
        sigma = float(np.std(logret))
    else:
        sigma = None

    return hist, new_val, sigma

if __name__ == "__main__":
    hist, new_fx, sigma = update_fx()
    print("[FX Update OK]")
    print("Latest FX:", new_fx)
    print("FX Vol 21d:", sigma)
