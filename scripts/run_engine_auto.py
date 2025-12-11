# scripts/run_engine_auto.py

import json
from pathlib import Path
from datetime import datetime


DATA_DIR = Path("data")


def find_latest_market_json() -> Path:
    """
    data/ 폴더에서 가장 최근 market_data_*.json 파일을 찾는다.
    """
    candidates = sorted(DATA_DIR.glob("market_data_*.json"))
    if not candidates:
        raise FileNotFoundError("[ERROR] data/ 폴더에 market_data_*.json 파일이 없습니다.")
    return candidates[-1]


def main():
    latest_path = find_latest_market_json()

    with latest_path.open("r", encoding="utf-8") as f:
        market = json.load(f)

    print(f"[INFO] Loaded market data JSON: {latest_path}")
    print(f"[INFO] Top-level keys: {list(market.keys())}")

    risk = market.get("risk", {})
    rates = market.get("rates", {})
    macro = market.get("macro", {})

    print("[INFO] Risk block:", risk)
    print("[INFO] Rates block:", rates)
    print("[INFO] Macro block:", macro)

    pmi_latest = macro.get("pmi_markit")
    print(f"[INFO] S&P Global Manufacturing PMI (Latest): {pmi_latest}")

    print("[OK] AURORA 엔진 입력용 시장 데이터 검증 완료.")


if __name__ == "__main__":
    main()
