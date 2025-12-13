# AURORA Rev12.1b Daily Report

- Report Date: 2025-12-13
- Engine Version: AURORA-Rev12.1b
- Git Commit: 82c35dc8337953097f8be563368f8dc18a45a037
- Run ID: 20188079759
- Timestamp(UTC): 2025-12-13T06:21:00.274964+00:00

## 1. Market Data Summary (FD inputs)
- USD/KRW (Sell Rate): 1477.30
- VIX: 15.74
- HY OAS: 288.00 bps
- UST 2Y / 10Y: 3.52% / 4.14%

## 2. Target Weights (Portfolio 100%)

| Asset   | Weight (%) |
|---------|-----------:|
| SPX     |      29.12 |
| NDX     |      13.59 |
| DIV     |      12.76 |
| EM      |       6.00 |
| ENERGY  |       3.00 |
| DURATION |       0.00 |
| SGOV    |      30.54 |
| GOLD    |       5.00 |
| **Total** | **    100.00** |

## 3. FD / ML / Systemic Signals

## FXW Anchor Distribution (USD/KRW, 130D KDE)

- KDE Anchor (Mode): **1387.1**
- Distribution: min=1348.5, P05=1357.4, P25=1384.4, P50=1392.8, P75=1430.7, P95=1470.4, max=1477.3
- Current FX: 1477.30 → **above anchor (KRW weak)**

- FXW (KDE): 0.289
- FX Vol (21D σ): 0.0045
- SPX 3Y Drawdown: -1.07%
- MacroScore: 0.610
- ML_Risk / ML_Opp / ML_Regime: 0.197 / 0.315 / 0.416
- Systemic Level / Bucket: 0.186 / C0

## 4. Engine State

- Final State: S0_NORMAL

## 5. CMA Overlay (TAS Dynamic Threshold) Snapshot

- CMA Snapshot (KRW): deployed=0, cash=12000, total=12000, ref_base=0, s0_count=1
- Threshold: 18.0%
- Deploy Factor: 0.0%
- Target Deploy (KRW): 0
- Delta Raw (KRW): 0
- FX Scale (BUY only): 0.873
- Suggested Exec: HOLD 0 KRW (0.00% of total CMA)

| CMA Allocation (KRW, based on Suggested Exec) | Amount |
|----------------------------------------------|-------:|
| SPX                  | 0 |
| NDX                  | 0 |
| DIV                  | 0 |
| EM                   | 0 |
| ENERGY               | 0 |
