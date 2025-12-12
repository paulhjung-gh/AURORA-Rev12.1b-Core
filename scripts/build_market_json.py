import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw_today.json"
FX_PATH = DATA_DIR / "fx_history.json"
FRED_PATH = DATA_DIR / "market_data_fred.json"

def main():
    print(f"[DEBUG] CWD={Path.cwd()}")
    print(f"[DEBUG] RAW_PATH={RAW_PATH} exists={RAW_PATH.exists()}")
    if RAW_PATH.exists():
        st = RAW_PATH.stat()
        print(f"[DEBUG] raw_today mtime={datetime.fromtimestamp(st.st_mtime)} size={st.st_size}")

    raw = load_json(RAW_PATH)
    fx_hist = load_json(FX_PATH)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_fred_into_market(market: dict) -> dict:
    """
    market_data_fred.json 의 latest 블록을
    최종 market_data_YYYYMMDD.json 의 risk / rates / macro 섹션에 직접 주입한다.
    """
    if not FRED_PATH.exists():
        print("[WARN] market_data_fred.json 가 없어 FRED merge 생략")
        return market

    fred = load_json(FRED_PATH)
    latest = fred.get("latest", {})

    # risk
    market["risk"]["hy_oas"] = latest.get("hy_oas_bps")

    # rates
    market["rates"]["dgs2"] = latest.get("dgs2")
    market["rates"]["dgs10"] = latest.get("dgs10")
    market["rates"]["ffr_upper"] = latest.get("ffr_upper")

    # macro
    market["macro"]["cpi_yoy"] = latest.get("cpi_yoy")
    market["macro"]["unemployment"] = latest.get("unemployment")
    market["macro"]["pmi_markit"] = latest.get("pmi_markit")

    return market


def main():
    raw = load_json(RAW_PATH)
    fx_hist = load_json(FX_PATH)

    today = datetime.now().strftime("%Y%m%d")
    out_path = DATA_DIR / f"market_data_{today}.json"

    # 기본 뼈대 생성
    market = {
        "date": today,
        "fx": {
            "usdkrw": fx_hist[-1],
            "usdkrw_history_130d": fx_hist,
            "usdkrw_history_21d": fx_hist[-21:]
        },
        "spx": raw["spx"],
        "risk": {"vix": raw["risk"]["vix"], "hy_oas": None},
        "rates": {"dgs2": None, "dgs10": None, "ffr_upper": None},
        "macro": {"cpi_yoy": None, "unemployment": None, "pmi_markit": None},
        "etf": raw["etf"]
    }

    # FRED+PMI 값 삽입
    market = merge_fred_into_market(market)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(market, f, indent=2, ensure_ascii=False)

    print("[OK] market_data JSON created:", out_path)


if __name__ == "__main__":
    main()
