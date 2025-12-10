import json
from pathlib import Path
from datetime import datetime

RAW_PATH = Path("data/raw_today.json")
FX_PATH = Path("data/fx_history.json")

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    raw = load_json(RAW_PATH)
    fx_hist = load_json(FX_PATH)

    today = datetime.now().strftime("%Y%m%d")
    out_path = Path(f"data/market_data_{today}.json")

    final = {
        "date": today,
        "fx": {
            "usdkrw": fx_hist[-1],
            "usdkrw_history_130d": fx_hist,
            "usdkrw_history_21d": fx_hist[-21:]
        },
        "spx": raw["spx"],
        "risk": raw["risk"],
        "rates": raw["rates"],
        "macro": raw["macro"],
        "etf": raw["etf"]
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print("[OK] market_data JSON created:", out_path)

if __name__ == "__main__":
    main()
