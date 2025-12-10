import requests
import json

# FRED API Key (GitHub Secrets에서 가져옴)
FRED_API_KEY = cfbd6d50f04185cacd4a46310bc8448e
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

def fetch_fred_data(series_id):
    """
    FRED API를 호출하여 데이터를 가져오는 함수.
    :param series_id: FRED 시리즈 ID (예: 'BAMLH0A0HYM2' 등)
    :return: 날짜와 값이 포함된 리스트 (value가 null인 경우는 제외)
    """
    url = f"{FRED_BASE_URL}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    try:
        resp = requests.get(url)
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
                    continue  # 값이 잘못된 경우에는 무시하고 계속 진행
        return values
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] FRED API 호출 중 오류 발생: {e}")
        return []

# 호출 예시
series_id = "BAMLH0A0HYM2"  # 시리즈 ID 예시
data = fetch_fred_data(series_id)
print(data)

def fetch_all():
    """
    필요한 모든 데이터를 FRED API로부터 가져오는 함수.
    :return: 각 지표별 데이터를 딕셔너리로 반환
    """
    hy_oas_data = fetch_fred_data("BAMLH0A0HYM2")  # HY OAS 데이터
    dgs2_data = fetch_fred_data("DGS2")  # 2Y Treasury Yield
    dgs10_data = fetch_fred_data("DGS10")  # 10Y Treasury Yield
    ffr_data = fetch_fred_data("DFEDTARU")  # FFR Upper (Federal Funds Rate)
    ism_pmi_data = fetch_fred_data("ISM")  # ISM PMI
    cpi_yoy_data = fetch_fred_data("CPIAUCSL")  # CPI YoY
    unemployment_data = fetch_fred_data("UNRATE")  # Unemployment Rate

    return {
        "HY_OAS": hy_oas_data,
        "DGS2": dgs2_data,
        "DGS10": dgs10_data,
        "FFR": ffr_data,
        "ISM_PMI": ism_pmi_data,
        "CPI_YoY": cpi_yoy_data,
        "Unemployment": unemployment_data
    }

def save_to_json(data, filename):
    """
    데이터를 JSON 파일로 저장하는 함수.
    :param data: 저장할 데이터
    :param filename: 저장할 파일명
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
        print(f"[INFO] {filename}에 데이터가 저장되었습니다.")

if __name__ == "__main__":
    # FRED API를 통해 모든 데이터를 가져옴
    market_data = fetch_all()

    # 결과를 JSON 파일로 저장
    save_to_json(market_data, "market_data_fred.json")
