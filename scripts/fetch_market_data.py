import json
from pathlib import Path
import yfinance as yf
import pandas as pd
import requests

OUTPUT_PATH = Path("data/market_data_auto.json")

def fred(series):
    url = f"https://api.stlouisfed.org/fred/series/observations"
    # (주의: FRED는 key 필요하지만 Grok은 대체가능, GitHub Actions에서는 key 필요)
    return None  # placeholder

def fetch_yahoo(ticker, period="1d"):
    return yf.download(ticker, period=period, progress=False)

def fetch_all():
    data = {}

    # ================ FX ===================
    fx_full = fetch_yahoo("KRW=X", period="1y")["Close"]
    data["fx"] = {
        "usdkrw": float(fx_full.iloc[-1]),
        "usdkrw_history_130d": fx_full.tail(130).tolist(),
        "usdkrw_history_21d": fx_full.tail(21).tolist(),
    }

    # ================ SPX ===================
    spx_full = fetch_yahoo("^GSPC", period="5y")["Close"]
    data["spx"] = {
        "index_level": float(spx_full.iloc[-1]),
        "history_3y": spx_full.tail(1095).tolist(),
    }

    # ================ VIX ===================
    data["risk"] = {
        "vix": float(fetch_yahoo("^VIX")["Close"].iloc[-1]),
        # HY OAS는 API key 필요. Manual or Grok 방식 권장.
        "hy_oas": None
    }

    # ================ Rates ===================
    data["rates"] = {
        "dgs2": None,
        "dgs10": None,
        "ffr_upper": None
    }

    # ================ Macro ===================
    data["macro"] = {
        "ism_pmi": None,
        "pmi_markit": None,
        "cpi_yoy": None,
        "unemployment": None,
    }

    # ================ ETF Prices ===================
    etfs = ["VOO", "QQQ", "SCHD", "SGOV", "VWO", "XLE", "GLD", "GLDM"]
    etf_prices = {}
    for t in etfs:
        px = fetch_yahoo(t)["Close"].iloc[-1]
        etf_prices[t] = float(px)

    data["etf"] = etf_prices

    return data

def main():
    data = fetch_all()
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("[OK] market_data_auto.json created.")

if __name__ == "__main__":
    main()
