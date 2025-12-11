import os
import re
import json
from typing import List, Tuple, Dict, Any

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
    CPIAUCSL 시리즈로부터 YoY 변화를 계산한다.
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
    실패 시 예외 발생 → 워크플로우 전체 실패 (PMI는 필수 FD 입력).
    """
    url = "https://tradingeconomics.com/united-states/manufacturing-pmi"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AURORA-Rev12.1b Bot/1.0)"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 1) 상단 설명문에서 첫 번째 숫자 = 최신 PMI
        # 예: "Manufacturing PMI in the United States decreased to 52.2 points in November ..."
        headline = soup.find("h1")
        if not headline or not headline.text:
            raise MarketDataError("PMI 헤드라인(h1)을 찾지 못했습니다.")

        head_text = headline.get_text(strip=True)
        m = re.search(r"([-+]?\d+(?:\.\d+)?)", head_text)
        if not m:
            raise MarketDataError(f"PMI 헤드라인에서 숫자를 찾지 못했습니다: {head_text}")
        latest_value = float(m.group(1))

        # 2) Historical 테이블에서 최근 12개 값 추출
        table = soup.find("table", {"id": "calendar"}) or soup.find(
            "table", {"id": "tb_calendar"}
        )
        if not table:
            raise MarketDataError("PMI historical 테이블(calendar/tb_calendar)을 찾지 못했습니다.")

        historical: List[float] = []
        rows = table.find_all("tr")

        # 첫 행은 헤더, 이후 12개 행만 사용
        for row in rows[1:13]:
            cells = row.find_all("td")
            if not cells:
                continue

            value_str = None
            for cell in cells:
                txt = cell.get_text(strip=True)
                # "52.2" 같은 값만 잡는다
                if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", txt):
                    value_str = txt
                    break

            if value_str is None:
                continue

            try:
                historical.append(float(value_str))
            except ValueError:
                continue

        if not historical:
            raise MarketDataError("PMI historical 값을 하나도 추출하지 못했습니다.")

        return latest_value, historical

    except requests.RequestException as e:
        raise MarketDat
