import json
from pathlib import Path
import yfinance as yf

DATA_PATH = Path("data/market_data_20251210.json")  # 최신 파일로 변경 권장
FX_TICKER = "KRW=X"

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def bootstrap_fx_history():
    """최초 1회: 130일 환율 기록 확보."""
    fx = yf.download(FX_TICKER, period="1y", progress=False)["Close"]
    hist = fx.tail(130).tolist()
    return hist

def update_fx_history(hist):
    """매일 1일치 append 후 130일 유지."""
    new_fx = yf.download(FX_TICKER, period="1d", progress=False)["Close"].iloc[-1]
    hist.append(float(new_fx))
    if len(hist) > 130:
        hist = hist[-130:]
    return hist

def compute_fx_vol(hist):
    """21일 log-return sigma."""
    import numpy as np
    if len(hist) < 22:
        return None
    prices = np.array(hist[-22:])
    logret = np.diff(np.log(prices))
    return float(np.std(logret))

def main():
    data = load_json(DATA_PATH)
    hist = data["fx"].get("usdkrw_history_130d", [])

    if len(hist) < 130:
        hist = bootstrap_fx_history()

    hist = update_fx_history(hist)
    fx_vol = compute_fx_vol(hist)

    data["fx"]["usdkrw_history_130d"] = hist
    data["fx"]["usdkrw_history_21d"] = hist[-21:]
    data["fx"]["fx_vol_21d"] = fx_vol

    save_json(DATA_PATH, data)
    print("[OK] FX 130-day history & 21-day vol updated.")

if __name__ == "__main__":
    main()
