# AURORA Rev12.1b 운영 매뉴얼 (Ops-Runbook-AI-KR)

## 1. 목적 (Purpose)
이 문서는 AURORA Rev12.1b 엔진을 운영하는 데 필요한 절차와 지침을 설명합니다. 운영자는 이 매뉴얼을 통해 엔진을 실행하고 모니터링하며, 발생할 수 있는 문제를 해결할 수 있습니다.

## 2. 시스템 개요 (System Overview)
AURORA Rev12.1b 엔진은 다양한 레이어를 포함하는 결정론적 포트폴리오 관리 시스템입니다. 이 시스템은 시장 데이터를 기반으로 포트폴리오의 비중을 자동으로 계산하며, 여러 가지 투자 자산군에 대한 알파를 생성합니다.

- **Market Data Layer**: 외부 시장 데이터를 수집하고 이를 엔진에 입력합니다.
- **ML Layer**: 시장의 위험도 및 기회 지표를 계산합니다.
- **Systemic Layer**: 시스템 리스크를 계산하여 전체 투자 전략에 영향을 미칩니다.
- **SGOV, Satellite, Duration**: 각 자산군의 비중을 계산하고, 포트폴리오를 자동으로 조정합니다.

## 3. 운영 절차 (Operational Procedures)

### 3.1 데이터 수집 (Data Collection)
1. **시장 데이터**: 매일 `Yahoo Finance`에서 `USD/KRW` 매도 환율, S&P500, 나스닥 100, VIX 등 필요한 데이터를 가져옵니다. 이를 통해 엔진의 주요 입력값을 생성합니다.
   - **API 호출 예시**:
     ```python
     import yfinance as yf
     spx = yf.download("^GSPC", period="1y")
     ```

2. **데이터 파일 업데이트**: 모든 시장 데이터 파일은 GitHub 저장소에서 자동으로 커밋되며, 최신 데이터를 기반으로 매일 엔진이 구동됩니다.

### 3.2 엔진 실행 (Engine Execution)
1. **엔진 실행**: 엔진을 실행하여 최신 데이터를 바탕으로 포트폴리오 비중을 산출합니다. 이를 위해 `engine.py` 파일을 사용합니다.
   - 실행 명령어:
     ```bash
     python engine.py
     ```

2. **주요 계산**:
   - **SGOV Floor 계산**: 금리 및 리스크를 바탕으로 SGOV Floor 값을 계산합니다.
   - **Satellite 비중**: 시장 상황에 따라 Satellite 비중을 조정합니다.
   - **Duration 비중**: 매크로 경제 상황에 맞춰 Duration 비중을 설정합니다.

### 3.3 문제 해결 (Problem Resolution)
1. **데이터 불러오기 오류**: 외부 데이터 소스에서 데이터를 가져오지 못할 경우, 마지막으로 업데이트된 데이터를 사용하여 엔진을 실행합니다.
   - 예시: `Yahoo Finance`에서 데이터가 막히는 경우, 최근 130일간의 환율 데이터를 사용할 수 있습니다.
   
2. **시스템 오류**: 시스템 장애가 발생할 경우, 서버를 재시작하여 서비스를 복구합니다. 재시작 명령어:
   ```bash
   sudo systemctl restart aurora-engine
