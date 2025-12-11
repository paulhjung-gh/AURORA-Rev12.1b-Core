# scripts/build_market_json.py

import json
from pathlib import Path
from datetime import datetime

# 입력 파일들
RAW_PATH = Path("data/raw_today.json")
FX_PATH = Path("data/fx_history.json")
FRED_PATH = Path("market_data_fred.json")  # FRED+PMI 결과 파일


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_fred_into_market(market: dict) -> dict:
    """
    market_data_fred.json 의 latest 블록을
    최종 market_data_YYYYMMDD.json 의 risk / rates / macro 섹션에 주입한다.
    """
    if not FRED_PATH.exists():
        print("[WARN] market_data_fred.json 이 없어 FRED merge 를 건너뜁니다.")
        return market

    with FRED_PATH.open("r", encoding="utf-8") as f:
        fred = json.load(f)

    latest = fred.get("latest", {})

    # --- risk 블록 ---
    risk = market.get("risk", {})
    if latest.get("hy_oas_bps") is not None:
        risk["hy_oas"] = latest["hy_oas_bps"]
    market["risk"] = risk

    # --- rates 블록 ---
    rates = market.get("rates", {})
    if latest.get("dgs2") is not None:
        rates["dgs2"] = latest["dgs2"]
    if latest.get("dgs10") is not None:
        rates["dgs10"] = latest["dgs10"]
    if latest.get("ffr_upper") is not None:
        rates["ffr_upper"] = latest["ffr_upper"]
    market["rates"] = rates

    # --- macro 블록 ---
    macro = market.get("macro", {})
    if latest.get("cpi_yoy") is not None:
        macro["cpi_yoy"] = latest["cpi_yoy"]
    if latest.get("unemployment") is not None:
        macro["unemployment"] = latest["unemployment"]
    if latest.get("pmi_markit") is not None:
        # ISM → S&P Global US Manufacturing PMI 대체
        macro["pmi_markit"] = latest["pmi_markit"]
    market["macro"] = macro

    return market


def main():
    # 1) raw_today / fx_history 로 기본 market 구조 생성
    raw = load_json(RAW_PATH)
    fx_hist = load_json(FX_PATH)

    today = datetime.now().strftime("%Y%m%d")
    out_path = Path(f"data/market_data_{today}.json")

    market = {
        "date": today,
        "fx": {
            "usdkrw": fx_hist[-1],
            "usdkrw_history_130d": fx_hist,
            "usdkrw_history_21d": fx_hist[-21:],
        },
        "spx": raw["spx"],
        "risk": raw["risk"],
        "rates": raw["rates"],
        "macro": raw["macro"],
        "etf": raw["etf"],
    }

    # 2) 여기서 FRED + PMI latest 를 risk/rates/macro 에 주입
    market = merge_fred_into_market(market)

    # 3) 최종 JSON 저장
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(market, f, indent=2, ensure_ascii=False)

    print("[OK] market_data JSON created:", out_path)


if __name__ == "__main__":
    main()
