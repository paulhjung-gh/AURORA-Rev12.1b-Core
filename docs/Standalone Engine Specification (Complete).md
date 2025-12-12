0. Engine Philosophy (비가역 전제)

Engine Version: AURORA Rev12.1b

Decision Maker: AI Engine

Executor: Human (Buy/Sell only)

Human discretion: ❌

Determinism: Required

External discretionary override: ❌

엔진은 상태(state)와 비중(weight) 만 산출한다.
실행 금액, 계좌 분리는 엔진 외부 영역이다.

1. Capital Structure (고정 규칙)

월 총 납입액: 4,000,000 KRW

Gold Sleeve: 고정 5%

ISA: 100,000

Direct: 100,000

Engine-controlled capital: 95%

Gold는:

타이밍 판단 ❌

리밸런싱 ❌

CMA 대상 ❌

2. Canonical Market Input (Deterministic)
2.1 Input File

data/market_data_YYYYMMDD.json

외부 다운로드 ❌

파일 내 데이터만 사용

2.2 Required Fields
FX

usdkrw (float)

usdkrw_history_21d (list, ≥2)

usdkrw_history_130d (list, ≥130, trading days)

Risk

vix (level)

hy_oas (bps or % → bps 변환)

Rates

dgs2 (%)

dgs10 (%)

ffr_upper (%)

Macro

pmi_markit (points)

cpi_yoy (%)

unemployment (%)

Equity

spx.last

spx.closes_3y_1095 (≥200)

3. P0-4 Unit Guards (Fail-Fast)
3.1 Risk

VIX ∈ [5, 80]

HY OAS:

if 0 < value < 50 → ×100 (percent → bps)

final range ∈ [50, 2000] bps

3.2 Rates

UST2Y, UST10Y, FFR ∈ [0, 25] %

3.3 Macro

PMI ∈ [0, 100]

CPI YoY ∈ [-20, 50]

Unemployment ∈ [0, 30]

3.4 Yield Curve
yc_spread_bps = (dgs10 - dgs2) * 100
range: [-300, +300]


위반 시 엔진 즉시 중단.

4. FXW (P0-2 Core Signal)
4.1 Input

USD/KRW 최근 130 trading days

4.2 KDE Construction

Gaussian KDE

Anchor = Mode of density

4.3 FXW Calculation

fxw = engine.kde.fxw(current_fx)

4.4 Guard
if anchor < (P05 - 10):
    FAIL (input series integrity broken)

5. Derived Signals
5.1 FX Volatility
fx_vol = std(log returns, 21D)

5.2 Drawdown
drawdown = (last - peak) / peak

6. Macro Score (Official Formula)

ISM 부재 시 PMI 대체 사용.

ism_n   = norm(ISM_or_PMI, 45, 60)
pmi_n   = norm(PMI,        45, 60)
cpi_n   = 1 - norm(CPI,     2,  8)
unemp_n = 1 - norm(Unemp,   3,  7)

MacroScore = clip(0.25 * (ism_n + pmi_n + cpi_n + unemp_n), 0, 1)

7. ML Layer (Blackbox, Deterministic)
Inputs

VIX

HY OAS

FXW

Drawdown

Yield Curve Spread

Outputs

ml_risk ∈ [0,1]

ml_opp ∈ [0,1]

ml_regime = f(ml_risk, ml_opp)

8. Systemic Layer
systemic_level = f(
    hy_oas,
    yc_spread,
    macro_score,
    ml_regime,
    drawdown
)

Bucket

C0: Normal

C1: Stress

C2/C3: Crisis

9. Final State Machine
if systemic_bucket in {C2, C3} or systemic_level ≥ 0.70:
    S3_HARD
elif fx_vol ≥ 0.02:
    S2_HIGH_VOL
elif ml_risk ≥ 0.80 or vix ≥ 30 or drawdown ≤ -30%:
    S2_HIGH_VOL
elif ml_risk ≥ 0.60 or vix ≥ 22 or drawdown ≤ -10%:
    S1_MILD
else:
    S0_NORMAL

10. Portfolio Construction
10.1 Caps

SGOV ≤ 80%

Satellite ≤ 20%

Duration ≤ 30%

10.2 Allocation Logic

Engine allocates 95%

Gold fixed at 5%

10.3 Core Split
SPX = 52.5%
NDX = 24.5%
DIV = 23.0%

10.4 Satellite Split
EM = 2/3
ENERGY = 1/3

10.5 Normalization
sum(weights) == 100%

11. CMA Overlay (External but Coupled)

Monthly inflow와 완전 분리

TAS Dynamic Threshold (God Mode)

입력:

fxw, vix, hy_oas, drawdown, ml_risk, systemic_bucket

출력:

BUY / SELL 금액 (KRW)

12. Engine Outputs
JSON

signals

state

weights

cma_overlay

Human-readable

Daily Markdown Report

13. Explicit Non-Scope

월 매수 금액

ISA / 직투 분리

실행 단가

감정 판단

예측

14. Final Declaration

이 문서는:

Rev12.1b를 완전히 정의

다른 문서 참조 ❌

다른 AI에서도 동일 입력 → 동일 출력 보장
