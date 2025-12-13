# AURORA Rev12.2 Daily Report

- Report Date: 2025-12-13
- Engine Version: AURORA Rev12.2
- Git Commit: b5a5f35301534e980d9f054fa11484d5fd630240
- Run ID: 20185689689
- Timestamp(UTC): 2025-12-13T02:54:11.879498+00:00

## 1. Market Data Summary (FD inputs)
- USD/KRW (Sell Rate): 1470.85
- VIX: 15.74
- HY OAS: 288.00 bps
- UST 2Y / 10Y: 3.52% / 4.14%

## 2. Target Weights (Portfolio 100%)

| Asset   | Weight (%) |
|---------|-----------:|
| SPX     |      29.20 |
| NDX     |      13.63 |
| DIV     |      12.79 |
| EM      |       6.00 |
| ENERGY  |       3.00 |
| DURATION |       0.00 |
| SGOV    |      30.38 |
| GOLD    |       5.00 |
| **Total** | **    100.00** |

## 3. FD / ML / Systemic Signals

## FXW Anchor Distribution (USD/KRW, 130D KDE)

- KDE Anchor (Mode): **1387.0**
- Distribution: min=1348.5, P05=1357.4, P25=1383.8, P50=1392.6, P75=1430.2, P95=1469.7, max=1474.9
- Current FX: 1470.85 → **above anchor (KRW weak)**

- FXW (KDE): 0.302
- FX Vol (21D σ): 0.0057
- SPX 3Y Drawdown: -1.07%
- MacroScore: 0.610
- ML_Risk / ML_Opp / ML_Regime: 0.194 / 0.321 / 0.414
- Systemic Level / Bucket: 0.186 / C0

## 4. Engine State

- Final State: S0_NORMAL

## 5. CMA Overlay (TAS Dynamic Threshold) Snapshot

- CMA Snapshot (KRW): deployed=0, cash=12000, total=12000, ref_base=0, s0_count=1
- Threshold: 18.0%
- Deploy Factor: 0.0%
- Target Deploy (KRW): 0
- Delta Raw (KRW): 0
- FX Scale (BUY only): 0.881
- Suggested Exec: HOLD 0 KRW (0.00% of total CMA)

| CMA Allocation (KRW, based on Suggested Exec) | Amount |
|----------------------------------------------|-------:|
| SPX                  | 0 |
| NDX                  | 0 |
| DIV                  | 0 |
| EM                   | 0 |
| ENERGY               | 0 |

## 6. MDD Reference (Optional)

- Core Equity MDD (reference): N/A
- Total Portfolio MDD (investor-experienced): N/A
