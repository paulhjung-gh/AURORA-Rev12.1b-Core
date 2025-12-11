import requests
from bs4 import BeautifulSoup
import json
import os

# FRED API Key (GitHub Secrets에서 가져오는 것을 추천)
# GitHub Actions에서는 FRED_API_KEY 를 env 로 넣어두고 사용:
#   env:
#     FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
FRED_API_KEY = os.getenv("FRED_API_KEY", "cfbd6d50f04185cacd4a46310bc8448e")

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_data(series_id):
    """
    FRED API를 호출하여 데이터를 가져오는 함수.
    :param series_id: FRED 시리즈 ID (예: 'BAMLH0A0HYM2' 등)
    :return: 날짜와 값이 포함된 리스트 (value가 null인 경우는 제외)
    """
    if not FRED_API_KEY:
        print("[ERROR] FRED_API_KEY 가 설정되어 있지 않습니다.")
        return []

    url = f"{FRED_BASE_URL}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()  # API 호출 실패 시 예외 발생
        data = resp.json()
        observations = data.get("observations", [])
        values = []

        # 관측값을 추출하고 값이 존재하는 경우만 추가
        for o in observations:
            value = o.get("value", ".")
            if value != ".":
                try:
                    values.append(float(value))
                except ValueError:
                    # 값 파싱 실패 시 스킵
                    continue
        return values
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] FRED API 호출 중 오류 발생 (series_id={series_id}): {e}")
        return []


def fetch_sp_global_pmi():
    """
    S&P Global US Manufacturing PMI 값을 Trading Economics 페이지에서 스크래핑한다.
    - URL: https://tradingeconomics.com/united-states/manufacturing-pmi
    - 반환: (latest_value: float, historical: list[float])
      historical 은 최근값 포함 총 12개 정도의 시계열 (필요시 엔진에서 사용).
    - 실패 시에는 Exception 을 던져 GitHub Actions 를 바로 실패시키도록 한다.
    """

    url = "https://tradingeconomics.com/united-states/manufacturing-pmi"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AURORA-Rev12.1b Bot/1.0)"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 1) 상단 헤드라인(최근 값)에서 숫자 추출
        # 예: "Manufacturing PMI in the United States decreased to 52.2 points in November"
        headline = soup.find("h1")
        if not headline or not headline.text:
            raise RuntimeError("헤드라인 텍스트를 찾지 못했습니다.")

        head_text = headline.text.strip()
        # 첫 번째 숫자만 추출
        import re
        m = re.search(r"([-+]?\d+(?:\.\d+)?)", head_text)
        if not m:
            raise RuntimeError(f"헤드라인에서 숫자를 찾지 못했습니다: {head_text}")
        latest_value = float(m.group(1))

        # 2) 아래 Historical 테이블에서 최근 12개 값 추출 (Actual 컬럼)
        # Trading Economics 기준: id="calendar" 혹은 "tb_calendar" 인 테이블 사용
        table = soup.find("table", {"id": "calendar"}) or soup.find(
            "table", {"id": "tb_calendar"}
        )
        if not table:
            raise RuntimeError("Historical 테이블(calendar/tb_calendar)을 찾지 못했습니다.")

        historical = []
        rows = table.find_all("tr")

        # 첫 번째 행은 헤더이므로 제외하고, 이후 12행 정도만 본다.
        for row in rows[1:13]:
            cells = row.find_all("td")
            if not cells:
                continue

            # 일반적으로 Actual 값이 있는 컬럼(보통 1번째 또는 2번째)
            value_str = None
            for cell in cells:
                txt = cell.get_text(strip=True)
                # "52.2", "51.9"처럼 숫자로 시작하는 값만 필터링
                if re.match(r"^[-+]?\d+(?:\.\d+)?$", txt):
                    value_str = txt
                    break

            if value_str is None:
                continue

            try:
                historical.append(float(value_str))
            except ValueError:
                continue

        if not historical:
            raise RuntimeError("Historical PMI 시계열을 추출하지 못했습니다.")

        # latest_value 를 historical 맨 앞에 정렬상 맞게 넣어둘 수도 있지만,
        # 보통 headline 이 최신, 테이블 첫 행도 비슷한 값이라
        # latest_value 를 별도 필드로 두고 historical 은 순수 테이블 값으로 둔다.
        return latest_value, historical

    except Exception as e:
        # 이 함수에서 실패하면 그대로 엔진/Actions 전체를 실패시켜야 함
        raise RuntimeError(f"S&P Global PMI 스크래핑 실패: {e}")



def fetch_all():
    """
    필요한 모든 데이터를 FRED API + S&P Global PMI(Trading Economics)에서 가져온다.
    - HY OAS / 2Y / 10Y / FFR / CPI Index / Unemployment : FRED
    - Manufacturing PMI : Trading Economics (S&P Global)
    """
    # --- FRED series ---
    hy_oas_data = fetch_fred_data("BAMLH0A0HYM2")  # HY OAS (bps)
    dgs2_data = fetch_fred_data("DGS2")            # 2Y Treasury Yield
    dgs10_data = fetch_fred_data("DGS10")          # 10Y Treasury Yield
    ffr_data = fetch_fred_data("DFEDTARU")         # FFR Upper
    cpi_index_data = fetch_fred_data("CPIAUCSL")   # CPI Index (YoY는 엔진에서 계산)
    unemployment_data = fetch_fred_data("UNRATE")  # 실업률

    # 값이 하나도 없으면 바로 에러로 처리 (데이터 소스 문제)
    if not hy_oas_data:
        raise RuntimeError("HY OAS (BAMLH0A0HYM2) 데이터가 비어 있습니다.")
    if not dgs2_data:
        raise RuntimeError("DGS2 데이터가 비어 있습니다.")
    if not dgs10_data:
        raise RuntimeError("DGS10 데이터가 비어 있습니다.")
    if not ffr_data:
        raise RuntimeError("DFEDTARU 데이터가 비어 있습니다.")
    if not cpi_index_data:
        raise RuntimeError("CPIAUCSL 데이터가 비어 있습니다.")
    if not unemployment_data:
        raise RuntimeError("UNRATE 데이터가 비어 있습니다.")

    # --- S&P Global US Manufacturing PMI (무조건 값 있어야 함) ---
    sp_pmi_latest, sp_pmi_hist = fetch_sp_global_pmi()

    return {
        # 원시 시계열
        "HY_OAS": hy_oas_data,
        "DGS2": dgs2_data,
        "DGS10": dgs10_data,
        "FFR": ffr_data,
        "CPI_Index": cpi_index_data,
        "Unemployment": unemployment_data,
        "SP_PMI_History": sp_pmi_hist,

        # 엔진이 바로 쓸 수 있게 latest 값도 저장
        "SP_PMI_Latest": sp_pmi_latest,
    }

    }


def save_to_json(data, filename):
    """
    데이터를 JSON 파일로 저장하는 함수.
    :param data: 저장할 데이터
    :param filename: 저장할 파일명
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] {filename}에 데이터가 저장되었습니다.")


if __name__ == "__main__":
    market_data = fetch_all()
    save_to_json(market_data, "market_data_fred.json")

