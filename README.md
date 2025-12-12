AURORA Rev12.1b Core (Daily Auto Engine)

AURORA Rev12.1b는 자동 수집된 market data를 기반으로 **Core target weights(100%)**를 산출하고, 별도 계정(CMA)에서 운용되는 CMA TAS Dynamic Threshold Overlay의 **매수/매도 제안(suggested_exec)**까지 함께 산출합니다.
Engine decision is AI-driven; humans only execute buys/sells.

Key Principles

Deterministic run: run_engine_auto.py는 외부 다운로드로 보충하지 않음. 오직 data/market_data_YYYYMMDD.json만 신뢰.

Role separation

Core Portfolio: 엔진 내부에서 target weights 산출 (ISA 165 / Direct 235 고정 흐름 반영)

CMA Overlay: 엔진 외부 계좌. Core 엔진에 피드백 없음(❌)

Weights sum to 100%: 항상 총합 100%로 normalize.

Gold sleeve fixed: 월납입 규칙 기반 고정.

Repository Structure
data/ (engine-managed outputs)

자동 수집/생성되는 파일들.

market_data_YYYYMMDD.json : Final consolidated market JSON (engine input)

fx_history.json : USD/KRW history (130 trading days)

aurora_target_weights_YYYYMMDD.json : Engine output (signals + weights + CMA)

cma_state_YYYYMMDD.json : CMA state snapshot (sell-gate tracking 등)

raw_sources/ : 원천(raw) 데이터 보관 (optional)

insert/ (operator-managed inputs)

사람이 직접 넣는 입력 파일. GitHub Actions에서 insert → data로 inject됨.

insert/cma_balance.json : CMA 잔고/상태 입력 (월초 갱신 권장)

Policy: insert = human input, data = engine output.

Inputs
1) Market Data (auto)

Workflow에서 자동 생성:

FRED series: HY OAS, DGS2, DGS10, FFR, CPI (YoY), Unemployment

PMI: TradingEconomics scraping

FX: KRW=X (130d), volatility(21d sigma)

SPX/VIX 등 (build 단계에서 통합)

Final schema is stored in:

data/market_data_YYYYMMDD.json

2) CMA Balance (manual, via insert)

insert/cma_balance.json (example)

{
  "asof_yyyymm": "202512",
  "deployed_krw": 0,
  "cash_krw": 12000000,
  "ref_base_krw": 0
}


asof_yyyymm: 운영 기준 월 (YYYYMM)

deployed_krw: CMA에서 이미 Risk-on에 투입된 금액

cash_krw: CMA 현금

ref_base_krw: Operator fixed baseline

초기에는 0 가능 (baseline 미설정 상태 → 매수 로직 비활성처럼 동작)

첫 BUY 이후에는 원하는 기준값으로 고정 입력(권장)

Outputs
1) Target Weights (Core 100%)

Assets:

Core: SPX, NDX, DIV

Satellite: EM, ENERGY

Defensive / duration: SGOV, DURATION

Inflation hedge sleeve: GOLD

Report 및 JSON에는 항상 위 8개 축으로 100% 출력.

2) CMA Overlay (TAS Dynamic Threshold)

Outputs include:

threshold

deploy_factor

target_deploy_krw

delta_raw_krw

fx_scale (BUY only)

suggested_exec_krw (positive=BUY, negative=SELL)

Portfolio Rules (Fixed)
Monthly Flow

ISA: 1,650,000 KRW

Direct: 2,350,000 KRW

Total: 4,000,000 KRW

Gold Sleeve

ISA Gold: 100,000 KRW

Direct Gold: 100,000 KRW

Total Gold: 200,000 KRW

Gold sleeve weight = 200k / 4,000k = 5% (fixed)

Governance Caps (Clamp)

SGOV <= 80%

Satellite <= 20%

Duration <= 30%

Running Locally
pip install -r requirements.txt
export FRED_API_KEY="YOUR_KEY"

python scripts/fetch_market_data.py
python scripts/update_fx_history.py
python scripts/build_market_json.py

# optional: CMA input
cp insert/cma_balance.json data/cma_balance.json

python scripts/run_engine_auto.py


Generated files:

data/aurora_target_weights_YYYYMMDD.json

reports/aurora_daily_report_YYYYMMDD.md

GitHub Actions (Daily Automation)

Workflow runs:

Fetch Market Data (FRED + PMI)

Update FX History (KRW=X)

Build Final Market JSON

Inject CMA Balance (insert → data)

Run Engine

Commit & Push

Required Secret:

FRED_API_KEY

CMA Sell Gate: s0_count (Monthly Increment)

cma_state_YYYYMMDD.json tracks a SELL-gate counter:

의미: S0_NORMAL 상태가 “월 단위”로 연속 몇 번 유지되었는지

Daily run을 하더라도 asof_yyyymm이 바뀌지 않으면 1회만 증가하도록 설계(월 단위).

This avoids “daily-run inflation” of the sell counter.

Troubleshooting
Yahoo 401 / 429 (Unauthorized / Too Many Requests)

Occurs with Yahoo CSV endpoints or rate limiting.

Mitigation:

use fallback (yfinance)

reduce retries / backoff

avoid redundant downloads inside engine (engine should use consolidated market JSON only)

CMA JSON parse error

insert/cma_balance.json must be valid JSON (comma, quotes check).

Inject step copies it into data/cma_balance.json; engine reads from data/.

Engine determinism

If engine tries to download anything during run_engine_auto.py, that is a bug.
All external data must be resolved during build steps into data/market_data_YYYYMMDD.json.

License / Disclaimer

This repository is for research and personal portfolio automation.
Not financial advice.
