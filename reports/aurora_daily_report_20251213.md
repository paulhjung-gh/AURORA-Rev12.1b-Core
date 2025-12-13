# AURORA Rev12.2 Daily Report

- Report Date: 2025-12-13
- Engine Version: AURORA Rev12.2
- Git Commit: 74db6f5c20f10de5c85b2f426d8352af01958f32
- Run ID: 20197221887
- Timestamp(UTC): 2025-12-13T20:09:48.608929+00:00

## 1. Market Data Summary (FD inputs)
- USD/KRW (Sell Rate): 1477.30
- VIX: 15.74
- HY OAS: 288.00 bps
- UST 2Y / 10Y: 3.52% / 4.14%
- Yield Curve Spread (10Y-2Y bps): 62.0

## 2. Target Weights (Internal Portfolio 100%)

| Asset   | Weight (%) |
|---------|-----------:|
| SPX     |      29.42 |
| NDX     |      13.73 |
| DIV     |      12.89 |
| EM      |       6.00 |
| ENERGY  |       3.00 |
| DURATION |       0.00 |
| SGOV    |      29.96 |
| GOLD    |       5.00 |
| **Total** | **    100.00** |

## 3. Signals (FD / ML / Systemic)
- FXW (KDE): 0.337
- FX Vol (21D σ): 0.0045
- SPX 3Y Drawdown: -1.07%
- MacroScore: 0.610
- ML_Risk / ML_Opp / ML_Regime: 0.187 / 0.335 / 0.408
- Systemic Level / Bucket: 0.185 / C0
- Yield Curve Spread (10Y-2Y bps): 62.0

### FXW Anchor Distribution (USD/KRW, 130D KDE)
- KDE Anchor (Mode): 1387.1 | min=1348.5, p05=1357.4, p50=1392.8, p95=1470.4, max=1477.3
- Current FX: 1477.30 → above anchor (KRW weak)

## 4. Engine State
- Final State: S0_NORMAL

## 5. CMA Overlay (External) Snapshot
- CMA Snapshot (KRW): deployed=1, cash=12000, total=12001, ref_base=1, s0_count=1
- Threshold: 18.0%
- Deploy Factor: 0.0%
- Target Deploy (KRW): 0
- Suggested Exec: HOLD 0 KRW (0.00% of total CMA)

| CMA Allocation (KRW, based on Suggested Exec) | Amount |
|----------------------------------------------|-------:|
| SPX                  | 0 |
| NDX                  | 0 |
| DIV                  | 0 |
| EM                   | 0 |
| ENERGY               | 0 |
