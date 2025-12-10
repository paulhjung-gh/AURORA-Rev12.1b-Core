import json
from pathlib import Path
import yfinance as yf

OUTPUT_PATH = Path("data/raw_today.json")

def fetch_yahoo_price(ticker, period="1d"):
    df = yf.download(ticker, period=period, progress=False)
    return float(df["Close"].iloc[-1])

def fetch_all():
    data = {}

    # SPX
    spx = yf.download("^GSPC", period="5y", progress=False)["Close"]
    data["spx"] = {
        "index_level": float(spx.iloc[-1]),
        "history_3y": spx.tail(1095).tolist()
    }

    # Risk
    data["risk"] = {
        "vix": fetch_yahoo_price("^VIX"),
        "hy_oas": None  # API Key 필요, Actions에서 placeholder 유지
    }

    # Rates (FRED API Key 필요)
    data["rates"] = {
        "dgs2": None,
        "dgs10": None,
        "ffr_upper": None
    }

    # Macro
    data["macro"] = {
        "ism_pmi": None,
        "pmi_markit": None,
        "cpi_yoy": None,
        "unemployment": None
    }

    # ETFs
    etfs = ["VOO", "QQQ", "SCHD", "SGOV", "VWO", "XLE", "GLD", "GLDM"]
    etf_data = {}
    for t in etfs:
        etf_data[t] = fetch_yahoo_price(t)
    data["etf"] = etf_data

    return data

def main():
    data = fetch_all()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("[OK] raw_today.json created")

if __name__ == "__main__":
    main()
