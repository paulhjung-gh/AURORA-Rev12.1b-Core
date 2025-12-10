import json
from pathlib import Path

# 엔진 코드 import (이미 engine/에 올려둔 파일 기준)
from engine.AuroraX_Rev12_1b_Engine import AuroraX121

DATA_PATH = Path("data/market_data_20251210.json")

def load_market_data(path: Path) -> dict:
    """market_data_YYYYMMDD.json을 읽어서 dict로 반환."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    if not DATA_PATH.exists():
        print(f"[ERROR] {DATA_PATH} 파일이 없습니다.")
        return

    data = load_market_data(DATA_PATH)

    # 1) 기본 확인용 출력
    print("=== AURORA Rev12.1b – Market Data Snapshot ===")
    print(f"Date        : {data.get('date')}")
    print(f"USD/KRW     : {data['fx']['usdkrw']}")
    print(f"SPX Level   : {data['spx']['index_level']}")
    print(f"VIX         : {data['risk']['vix']}")
    print(f"HY OAS (bps): {data['risk']['hy_oas']}")
    print(f"DGS2 / DGS10: {data['rates']['dgs2']} / {data['rates']['dgs10']}")
    print(f"CPI YoY     : {data['macro']['cpi_yoy']}")
    print(f"Unemp Rate  : {data['macro']['unemployment']}")
    print("ETF Prices  :", data["etf"])

    # 2) 엔진 인스턴스 생성 (AuroraX121)
    engine = AuroraX121()

    # 3) 여기서부터는 나중에:
    #    - data dict에서 FD 신호(FXW, MacroScore, Drawdown, FXVol 등) 계산
    #    - ML / Systemic / State / SGOV / Satellite / Duration / CMA
    #    - 최종 Target Weights 산출
    #
    # 지금은 "연결 자리"만 만들어 두고,
    # 실제 수식/계산은 엔진 내부에 이미 있으니,
    # ChatGPT/Grok이 이 파일과 JSON/엔진코드를 같이 읽고
    # 그때그때 계산을 도와주게 하는 구조로 쓸 수 있다.

    print("\n[INFO] 여기까지가 data -> 엔진 연결의 스켈레톤입니다.")
    print("       이후 단계에서 AuroraX121 메서드들을 호출해서")
    print("       실제 Target Weights(SPX/NDX/DIV/EM/ENERGY/Dur/SGOV/Gold)를 계산하게 만들 수 있습니다.")

if __name__ == "__main__":
    main()
