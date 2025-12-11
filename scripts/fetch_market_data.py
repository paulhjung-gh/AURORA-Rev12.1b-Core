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
    S&P Global US Manufacturing PMI (TradingEconomics 기준)를 가져오는 함수.
    소스: https://tradingeconomics.com/united-states/manufacturing-pmi

    페이지 안의 "Latest Value" 숫자를 정규식으로 파싱한다.
    :return: 최신 PMI 값 (float) 또는 None
    """
    url = "https://tradingeconomics.com/united-states/manufacturing-pmi"
    headers = {
        # 간단한 User-Agent 로봇 차단 회피용
        "User-Agent": "Mozilla/5.0 (compatible; AURORA-Rev12.1b Bot/1.0)"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        html = resp.text

        # TradingEconomics는 보통 이런 형태로 latest value를 노출:
        # id="ctl00_ContentPlaceHolder1_LatestValue">52.2</span>
        m = re.search(
            r'id="ctl00_ContentPlaceHolder1_LatestValue">\s*([-+]?\d+(?:\.\d+)?)',
            html
        )
        if m:
            return float(m.group(1))

        # 혹시 위 패턴이 바뀐 경우를 대비한 fallback:
        # "Latest Value 52.2" 같은 패턴 잡기
        m2 = re.search(
            r"Latest Value[^0-9\-+]*([-+]?\d+(?:\.\d+)?)",
            html
        )
        if m2:
            return float(m2.group(1))

        # 그래도 못 찾으면 None 리턴 (엔진에서 graceful degrade)
        print("[WARN] S&P Global PMI: HTML에서 Latest Value 패턴을 찾지 못했습니다.")
        return None

    except Exception as e:
        print(f"[ERROR] S&P Global PMI 스크래핑 중 오류 발생: {e}")
        return None


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
