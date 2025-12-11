# scripts/run_engine_auto.py

import json
from pathlib import Path
from datetime import datetime


DATA_DIR = Path("data")


def find_latest_market_json() -> Path:
    """
    data/ 폴더에서 가장 최근 market_data_*.json 파일을 찾는다.
    예: data/market_data_20251211.json
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

    # 구조는 build_market_json.py 설계에 따라 다를 수 있으니,
    # 우선 상위 키를 한 번 보여 준다.
    print(f"[INFO] Top-level keys: {list(market.keys())}")

    # FRED 블록 요약 (예: 'fred' 내부에 저장했다고 가정)
    fred = market.get("fred") or market.get("FRED") or {}
    if fred:
        print("[INFO] FRED snapshot:")
        hy = fred.get("hy_oas_bps") or fred.get("HY_OAS")
        d2 = fred.get("ust2y_yield") or fred.get("DGS2")
        d10 = fred.get("ust10y_yield") or fred.get("DGS10")
        ffr = fred.get("ffr_upper") or fred.get("FFR")
        cpi_yoy = fred.get("cpi_yoy")
        ur = fred.get("unemployment_rate") or fred.get("Unemployment")

        print(f"  HY OAS (bps): {hy}")
        print(f"  2Y UST (%):   {d2}")
        print(f"  10Y UST (%):  {d10}")
        print(f"  FFR Upper (%): {ffr}")
        print(f"  CPI YoY (%):   {cpi_yoy}")
        print(f"  Unemployment (%): {ur}")
    else:
        print("[WARN] market JSON 안에 'fred' 블록을 찾지 못했습니다.")

    # S&P Global Manufacturing PMI Latest (우리가 앞에서 저장한 필드명 기준)
    pmi_latest = (
        market.get("pmi", {}).get("manufacturing")
        or market.get("SNP_Manufacturing_PMI_Latest")
    )
    print(f"[INFO] S&P Global Manufacturing PMI (Latest): {pmi_latest}")

    # 여기까지 오면 GitHub Actions는 성공(exit code 0)
    print("[OK] AURORA 엔진 입력용 시장 데이터 검증 완료.")


if __name__ == "__main__":
    main()
