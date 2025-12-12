# scripts/fetch_market_data.py

import os
import re
import json
from typing import List, Tuple, Dict, Any
from datetime import datetime

import requests
from bs4 import BeautifulSoup


FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


class MarketDataError(Exception):
    pass


def fetch_fred_data(series_id: str) -> List[float]:
    """
    FRED API에서 특정 시리즈의 전체 시계열을 가져온다.
    value 가 "." 인 값은 제외하고 float 리스트로 반환.
    """
    if not FRED_API_KEY:
        raise MarketDataError("FRED_API_KEY 환경변수가 설정되어 있지 않습니다.")

    url = f"{FRED_BASE_URL}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        observations = data.get("observations", [])
        values: List[float] = []

        for o in observations:
            v = o.get("value", ".")
            if v == ".":
                continue
            try:
                values.append(float(v))
            except ValueError:
                continue

        return values

    except requests.RequestException as e:
        raise MarketDataError(f"FRED API 호출 실패 (series_id={series_id}): {e}") from e


def compute_cpi_yoy(cpi_series: List[float]) -> float:
    """
    CPIAUCSL 시리즈로부터 YoY 변화를 계산.
    최근 값 / 12개월 전 값 - 1 * 100
    """
    if len(cpi_series) < 13:
        raise MarketDataError("CPI 시계열 길이가 13 미만이라 YoY 계산 불가")

    latest = cpi_series[-1]
    year_ago = cpi_series[-13]
    if year_ago == 0:
        raise MarketDataError("12개월 전 CPI 값이 0이라 YoY 계산 불가")

    return (latest / year_ago - 1.0) * 100.0


def fetch_sp_global_pmi() -> Tuple[float, List[float]]:
    """
    S&P Global US Manufacturing PMI 값을 Trading Economics 페이지에서 스크래핑.
    - URL: https://tradingeconomics.com/united-states/manufacturing-pmi
    - 반환: (latest_value, historical_list)
    latest 값은 반드시 있어야 하며, 없으면 MarketDataError 발생.
    historical 은 실패해도 워크플로우를 죽이지 않고 빈 리스트 허용.
    """
    url = "https://tradingeconomics.com/united-states/manufacturing-pmi"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AURORA-Rev12.1b Bot/1.0)"}

    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        # 1) 최신 값: 본문 문장 전체에서 패턴 검색
        text_all = " ".join(soup.stripped_strings)

        m = re.search(
            r"Manufacturing PMI in the United States[^0-9\-+]*([-+]?\d+(?:\.\d+)?)\s*points",
            text_all,
        )
        if not m:
            m = re.search(
                r"United States Manufacturing PMI[^0-9\-+]*([-+]?\d+(?:\.\d+)?)",
                text_all,
            )

        if not m:
            raise MarketDataError(
                f"PMI 본문에서 숫자를 찾지 못했습니다: {text_all[:120]}..."
            )

        latest_value = float(m.group(1))

        # 2) Historical 12개월: 안 되면 경고만 찍고 빈 리스트 허용
        historical: List[float] = []
        try:
            table = soup.find("table", {"id": "calendar"}) or soup.find(
                "table", {"id": "tb_calendar"}
            )
            if table:
                rows = table.find_all("tr")
                for row in rows[1:13]:  # 헤더 제외 후 12개 행
                    cells = row.find_all("td")
                    if not cells:
                        continue

                    value_str = None
                    for cell in cells:
                        txt = cell.get_text(strip=True)
                        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", txt):
                            value_str = txt
                            break
                    if value_str is None:
                        continue
                    try:
                        historical.append(float(value_str))
                    except ValueError:
                        continue
            else:
                print("[WARN] PMI historical 테이블(calendar/tb_calendar)을 찾지 못했습니다.")
        except Exception as e:
            print(f"[WARN] PMI historical 파싱 실패, latest만 사용합니다: {e}")

        return latest_value, historical

    except requests.RequestException as e:
        raise MarketDataError(f"PMI 페이지 요청 실패: {e}") from e


def fetch_all() -> Dict[str, Any]:
    """
    FRED + PMI 모든 시계열을 수집.
    - latest: 엔진 FD 입력에 바로 쓰는 값
    - history: 필요한 최소 구간만 보관 (옵션)
    어떤 시리즈라도 비어 있으면 예외를 던진다.
    """
    hy_oas_data = fetch_fred_data("BAMLH0A0HYM2")   # HY OAS (bps)
    dgs2_data = fetch_fred_data("DGS2")             # 2Y Treasury Yield
    dgs10_data = fetch_fred_data("DGS10")           # 10Y Treasury Yield
    ffr_data = fetch_fred_data("DFEDTARU")          # FFR Upper
    cpi_index_data = fetch_fred_data("CPIAUCSL")    # CPI Index
    unemployment_data = fetch_fred_data("UNRATE")   # Unemployment Rate

    for name, series in [
        ("HY_OAS", hy_oas_data),
        ("DGS2", dgs2_data),
        ("DGS10", dgs10_data),
        ("FFR", ffr_data),
        ("CPI_Index", cpi_index_data),
        ("Unemployment", unemployment_data),
    ]:
        if not series:
            raise MarketDataError(f"{name} 시계열이 비어 있습니다.")

    cpi_yoy = compute_cpi_yoy(cpi_index_data)
    pmi_latest, pmi_hist = fetch_sp_global_pmi()

    latest = {
        "hy_oas_bps": hy_oas_data[-1],
        "dgs2": dgs2_data[-1],
        "dgs10": dgs10_data[-1],
        "ffr_upper": ffr_data[-1],
        "cpi_yoy": cpi_yoy,
        "unemployment": unemployment_data[-1],
        "pmi_markit": pmi_latest,
    }

    history = {
        "pmi_12m": pmi_hist[-12:],           # 최대 12개
        "cpi_13m": cpi_index_data[-13:],     # YoY 계산용 13개월
        "dgs2_30d": dgs2_data[-30:],         # 최근 1개월 금리
        "dgs10_30d": dgs10_data[-30:],       # 최근 1개월 금리
    }

    return {
        "latest": latest,
        "history": history,
    }


def save_to_json(data: Dict[str, Any], filename: str) -> None:
    # 날짜(YYYYMMDD)만 박아서 업데이트 여부를 검증 (시간 저장은 하지 않음)
    today = datetime.utcnow().strftime("%Y%m%d")
    data.setdefault("meta", {})
    if not isinstance(data["meta"], dict):
        data["meta"] = {}
    data["meta"]["generated_yyyymmdd"] = today

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] {filename} 저장 완료")
    print(f"[INFO] generated_yyyymmdd={today}")


if __name__ == "__main__":
    md = fetch_all()
    save_to_json(md, "data/market_data_fred.json")
