# Core-MarketDataSpec-Rev12.1b  
AURORA Rev12.1b 엔진이 요구하는 **시장 데이터 규격(Full Deterministic Edition)**  
(한글 base + 영문 혼용)

---

## 1. Purpose  
본 문서는 Rev12.1b 엔진이 매일 구동되기 위해 **반드시 충족해야 하는 시장 데이터 입력값의 규격(specification)** 을 정의한다.  
이 스펙에 따라 생성된 `market_data_YYYYMMDD.json` 파일은 엔진의 **단일 입력(single source of truth)** 역할을 한다.

---

## 2. Official Data Sources (공식 데이터 소스)

Rev12.1b Governance에 따라, 모든 데이터는 다음 공식 경로에서만 가져와야 한다.

### 2.1 Yahoo Finance  
- KRW=X (USD/KRW sell rate)  
- ^GSPC (S&P500 Index)  
- ^VIX (Volatility Index)  
- ETF prices:  
  - VOO (SPX), QQQ (NDX), SCHD (DIV)  
  - SGOV (T-Bill proxy)  
  - VWO (EM), XLE (Energy)  
  - GLD / GLDM (Gold)

### 2.2 FRED (Federal Reserve Economic Data)
- HY OAS: `BAMLH0A0HYM2`  
- FFR Upper Bound: `DFEDTARU`  
- 2Y Treasury Yield: `DGS2`  
- 10Y Treasury Yield: `DGS10`

### 2.3 Other Macroeconomic Data
- **ISM Manufacturing PMI**  
- **S&P Global Manufacturing PMI**  
- **CPI YoY**  
- **Unemployment Rate**

이 네 가지는 최신 발표 수치를 사용하며, 발표 주기가 길기 때문에 “전날 값 = 최신 공식값”으로 간주한다.

---

## 3. Required Time Series (필수 시계열 길이)

Rev12.1b 엔진은 다음 기간의 시계열 데이터를 반드시 필요로 한다.

### 3.1 KRW=X (USD/KRW)
- **최근 130 거래일**: KDE Anchor 계산  
- **최근 21일**: FX Volatility 계산  
- Latest close (당일/전일 종가)

### 3.2 S&P500 (^GSPC)
- **최근 1095 거래일 (약 3년)**  
  → Long-Horizon Drawdown 계산에 필수

### 3.3 ETF Prices
- Latest close (하루치로 충분)

---

## 4. JSON Structure Specificat
data/
└── market_data_YYYYMMDD.json

### 4.1 JSON Schema (정식 스키마)

```json
{
  "date": "2025-12-10",
  "fx": {
    "usdkrw": 1470.15,
    "usdkrw_history_130d": [ ... 130 floats ... ],
    "usdkrw_history_21d": [ ... 21 floats ... ]
  },
  "spx": {
    "index_level": 6049.88,
    "history_3y": [ ... 1095 floats ... ]
  },
  "rates": {
    "dgs2": 3.56,
    "dgs10": 4.14,
    "ffr_upper": 4.00
  },
  "risk": {
    "vix": 16.78,
    "hy_oas": 285.0
  },
  "macro": {
    "ism_pmi": 48.2,
    "pmi_markit": 52.2,
    "cpi_yoy": 3.0,
    "unemployment": 4.4
  },
  "etf": {
    "VOO": 554.23,
    "QQQ": 487.00,
    "SCHD": 75.30,
    "SGOV": 100.00,
    "VWO": 43.12,
    "XLE": 89.20,
    "GLD": 191.40,
    "GLDM": 19.50
  }
}
5. Field-by-Field Explanation (필드 세부 설명)
5.1 fx (환율)

usdkrw: 최신 USD/KRW 종가

usdkrw_history_130d: KDE Anchor 계산용

usdkrw_history_21d: FX Vol 계산용 σ(21d)

5.2 spx

index_level: S&P500 최신 종가

history_3y: Drawdown 계산용 피크 탐색 + 현재 대비 하락률

5.3 rates

dgs2: 미국 2년물 금리

dgs10: 미국 10년물 금리

ffr_upper: 연준 FFR Upper Bound

Yield Curve Spread = DGS10 – DGS2 는 엔진 내부에서 자동 계산된다.

5.4 risk

vix: 변동성 지수

hy_oas: 하이일드 OAS (크레딧 리스크)

5.5 macro

필수 FD 매크로 시그널 4개:

ISM PMI

S&P Global PMI

CPI YoY

Unemployment Rate

MacroScore = 공식 수식으로 구성된 FD 점수
(엔진에서 자동 계산)

5.6 etf

엔진의 실제 비중/가격 매핑에 사용:

Core Equity: VOO, QQQ, SCHD

Satellite: VWO, XLE

Duration: TLT or EDV (택 1; 가격 필드는 필요 시 추가)

SGOV

Gold: GLD / GLDM

6. Data Refresh Rules (데이터 갱신 규칙)
6.1 Daily Refresh

KRW=X

SPX

VIX

All ETFs

DGS2 / DGS10 / FFR Upper (FRED 전날값을 최신값으로 간주)

6.2 Monthly / Periodic Refresh

ISM PMI

PMI Markit

CPI YoY

Unemployment Rate

→ 발표 주기가 다르므로 “가장 최근 발표값”으로 저장

7. Deterministic Constraints (결정론적 제약)

JSON 파일에 포함된 값만 엔진 입력으로 사용한다.

엔진 구동 시 외부 API 호출 금지

동일 JSON 입력 → 동일 엔진 출력

Missing values 허용 불가

KRW=X, SPX 시계열 길이는 반드시 스펙대로 충족해야 한다.

8. Example File Path
data/market_data_20251210.json
data/market_data_20251211.json
...


GitHub Actions 또는 수동 fetch 스크립트가 이 파일을 생성한다.

9. Summary

이 문서는 Rev12.1b 엔진이 요구하는 시장 데이터의 형태, 기간, 소스, 스키마를 완전히 정의한다.
이 스펙은 scripts/fetch_market_data.py 와 엔진의 I/O 규격의 기반이며,
Rev12.1b deterministic 원칙의 핵심 구성 요소이다.
