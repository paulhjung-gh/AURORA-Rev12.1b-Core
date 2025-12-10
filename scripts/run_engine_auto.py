import json
from pathlib import Path
from datetime import datetime
import sys

# repo root 를 path 에 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.AuroraX_Rev12_1b_Engine import AuroraX121


def main():
    today = datetime.now().strftime("%Y%m%d")
    data_path = Path(f"data/market_data_{today}.json")

    if not data_path.exists():
        print("[ERROR] market_data file not found:", data_path)
        return

    with data_path.open("r", encoding="utf-8") as f:
        market = json.load(f)

    engine = AuroraX121()
    result = engine.run(market)  # 엔진에 run() 메서드 있다고 가정

    out_path = Path(f"data/results_{today}.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("[OK] Engine result saved:", out_path)

if __name__ == "__main__":
    main()
