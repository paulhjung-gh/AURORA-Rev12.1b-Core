# AURORA Rev12.1b Daily Report

- Report Date: 2025-12-12
- Engine Version: AURORA-Rev12.1b
- Git Commit: 7e8eab74d47acd4706dae9ccfeda9a62db3bf4b4
- Run ID: 20172264163
- Timestamp(UTC): 2025-12-12T15:56:32.698200+00:00

## 1. Market Data Summary (FD inputs)
- USD/KRW (Sell Rate): 1478.15
- VIX: 15.79
- HY OAS: 288.00 bps
- UST 2Y / 10Y: 3.54% / 4.13%

## 2. Target Weights (Portfolio 100%)

| Asset   | Weight (%) |
|---------|-----------:|
| SPX     |      29.11 |
| NDX     |      13.58 |
| DIV     |      12.75 |
| EM      |       6.00 |
| ENERGY  |       3.00 |
| DURATION |       0.00 |
| SGOV    |      30.56 |
| GOLD    |       5.00 |
| **Total** | **    100.00** |

## 3. FD / ML / Systemic Signals

## FXW Anchor Distribution (USD/KRW, 130D KDE)

- KDE Anchor (Mode): **1387.1**
- Distribution: min=1348.5, P05=1357.4, P25=1383.8, P50=1392.6, P75=1430.2, P95=1469.7, max=1478.2
- Current FX: 1478.15 → **above anchor (KRW weak)**

- FXW (KDE): 0.287
- FX Vol (21D σ): 0.0058
- SPX 3Y Drawdown: 0.00%
- MacroScore: 0.610
- ML_Risk / ML_Opp / ML_Regime: 0.198 / 0.315 / 0.416
- Systemic Level / Bucket: 0.187 / C0

## 4. Engine State

- Final State: S0_NORMAL

## 5. CMA Overlay (TAS Dynamic Threshold) Snapshot

- CMA Snapshot (KRW): deployed=0, cash=12000, total=12000, ref_base=0, s0_count=1
- Threshold: 18.0%
- Deploy Factor: 0.0%
- Target Deploy (KRW): 0
- Delta Raw (KRW): 0
- FX Scale (BUY only): 0.872
- Suggested Exec: HOLD 0 KRW (0.00% of total CMA)

| CMA Allocation (KRW, based on Suggested Exec) | Amount |
|----------------------------------------------|-------:|
| SPX                  | 0 |
| NDX                  | 0 |
| DIV                  | 0 |
| EM                   | 0 |
| ENERGY               | 0 |
