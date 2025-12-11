import os
import re
import requests
import json

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


def fetch_snp_pmi():
    """
    S&P Global US Manufacturing PMI를 TradingEconomics 웹페이지에서 스크래핑해서 가져오는 함수.
    - 소스: https://tradingeconomics.com/country-list/manufacturing-pmi
    - United States 행의 첫 번째 숫자(최신 값)를 파싱
    :return: 최신 PMI 값 (float) 또는 None
    """
    url = "https://tradingeconomics.com/country-list/manufacturing-pmi"
    headers = {
        # 간단한 UA를 넣어 403/봇 차단을 피하는 용도
        "User-Agent": "Mozilla/5.0 (compatible; AURORA-Rev12.1b Bot/1.0)"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        html = resp.text

        # "United States, 52.2, 52.5" 같은 패턴에서 첫 번째 숫자(52.2)를 추출
        # 테이블 구조가 변해도 "United States" 바로 뒤의 첫 번째 실수값을 잡는 방식
        m = re.search(r"United States[^0-9]+([0-9]+\.[0-9]+)", html)
        if not m:
            raise ValueError("United States Manufacturing PMI 값을 HTML에서 찾지 못했습니다.")
        latest_value = float(m.group(1))
        return latest_value

    except Exception as e:
        print(f"[ERROR] S&P Global PMI 스크래핑 중 오류 발생: {e}")
        return None


def fetch_all():
    """
    필요한 모든 데이터를 FRED API + 웹 스크래핑으로부터 가져오는 함수.
    :return: 각 지표별 데이터를 딕셔너리로 반환
    """
    # --- FRED 기반 시리즈들 ---
    hy_oas_data = fetch_fred_data("BAMLH0A0HYM2")   # HY OAS 데이터
    dgs2_data = fetch_fred_data("DGS2")             # 2Y Treasury Yield
    dgs10_data = fetch_fred_data("DGS10")           # 10Y Treasury Yield
    ffr_data = fetch_fred_data("DFEDTARU")          # FFR Upper (Federal Funds Rate)
    cpi_index_data = fetch_fred_data("CPIAUCSL")    # CPI Index (YoY는 추후 엔진에서 계산)
    unemployment_data = fetch_fred_data("UNRATE")   # Unemployment Rate

    # --- S&P Global US Manufacturing PMI (TradingEconomics 웹에서 스크래핑) ---
    snp_pmi_latest = fetch_snp_pmi()                # float or None

    return {
        "HY_OAS": hy_oas_data,
        "DGS2": dgs2_data,
        "DGS10": dgs10_data,
        "FFR": ffr_data,
        "CPI_Index": cpi_index_data,
        "Unemployment": unemployment_data,
        # PMI는 타임시리즈가 아니라 "최신 값 한 개"만 있으면 되므로 float로 저장
        "SNP_Manufacturing_PMI_Latest": snp_pmi_latest,
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
    # FRED + S&P PMI를 통해 모든 데이터를 가져옴
    market_data = fetch_all()

    # 결과를 JSON 파일로 저장
    save_to_json(market_data, "market_data_fred.json")
