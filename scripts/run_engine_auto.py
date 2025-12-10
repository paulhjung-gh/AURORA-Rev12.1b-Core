import json
from pathlib import Path
from datetime import datetime
import sys

# repo root 경로 설정
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# AuroraX121 클래스 임포트
from engine.AuroraX_Rev12_1b_Engine import AuroraX121

def main():
    # 오늘 날짜를 기준으로 market_data 파일명 설정
    today = datetime.now().strftime("%Y%m%d")
    data_path = Path(f"data/market_data_{today}.json")

    # 데이터 파일이 없으면 종료
    if not data_path.exists():
        print(f"[ERROR] market_data file not found: {data_path}")
        return

    # JSON 파일 읽기
    with data_path.open("r", encoding="utf-8") as f:
        market = json.load(f)

    # AuroraX121 엔진 실행
    engine = AuroraX121()
    result = engine.run(market)  # 엔진 실행

    # 결과 파일로 저장
    out_path = Path(f"data/results_{today}.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[INFO] 엔진 실행 결과 저장됨: {out_path}")

if __name__ == "__main__":
    main()
