# AURORA Rev12.1b Daily Report

- Report Date: 2025-12-12
- Engine Version: AURORA-Rev12.1b
- Git Commit: fb0e349746fd0a88101edd311ad870689777dcf0
- Run ID: 20169834001
- Timestamp(UTC): 2025-12-12T14:27:55.130941+00:00

## 1. Market Data Summary (FD inputs)
- USD/KRW (Sell Rate): 1476.66
- VIX: 14.92
- HY OAS: 291.00 bps
- UST 2Y / 10Y: 3.54% / 4.13%

## 2. Target Weights (Portfolio 100%)

| Asset   | Weight (%) |
|---------|-----------:|
| SPX     |      29.13 |
| NDX     |      13.59 |
| DIV     |      12.76 |
| EM      |       6.00 |
| ENERGY  |       3.00 |
| DURATION |       0.00 |
| SGOV    |      30.52 |
| GOLD    |       5.00 |
| **Total** | **    100.00** |

## 3. FD / ML / Systemic Signals

## FXW Anchor Distribution (USD/KRW, 130D KDE)

- KDE Anchor (Mode): **1387.1**
- Distribution: min=1348.5, P05=1357.4, P25=1383.8, P50=1392.6, P75=1430.2, P95=1469.7, max=1476.7
- Current FX: 1476.66 → **above anchor (KRW weak)**


- FXW (KDE): 0.290
- FX Vol (21D σ): 0.0058
- SPX 3Y Drawdown: 0.00%
- MacroScore: 0.610
- ML_Risk / ML_Opp / ML_Regime: 0.191 / 0.316 / 0.413
- Systemic Level / Bucket: 0.188 / C0

## 4. Engine State

- Final State: S0_NORMAL

## 5. CMA Overlay (TAS Dynamic Threshold) Snapshot

- CMA Snapshot (KRW): deployed=0, cash=12000, total=12000, ref_base=0, s0_count=1
- Threshold: 20.0%
- Deploy Factor: 0.0%
- Target Deploy (KRW): 0
- Delta Raw (KRW): 0
- FX Scale (BUY only): 0.874
- Suggested Exec: HOLD 0 KRW (0.00% of total CMA)

| CMA Allocation (KRW, based on Suggested Exec) | Amount |
|----------------------------------------------|-------:|
| SPX                  | 0 |
| NDX                  | 0 |
| DIV                  | 0 |
| EM                   | 0 |
| ENERGY               | 0 |
