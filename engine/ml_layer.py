
from typing import Any


def clip(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi]."""
    return max(lo, min(hi, x))


def norm(x: float, lo: float, hi: float) -> float:
    """
    Normalize x into [0,1] using linear scaling between [lo, hi].
    Values below lo map to 0, above hi map to 1.
    """
    if hi == lo:
        return 0.0
    return clip((x - lo) / (hi - lo), 0.0, 1.0)


# =========================================================
# ML_Risk: structural / volatility risk proxy in [0,1]
# ---------------------------------------------------------
# Higher VIX, wider HY OAS, lower FXW, deeper drawdown,
# and stronger yield curve inversion -> higher ML_Risk
# =========================================================

def compute_ml_risk(
    vix: float,
    hy_oas: float,
    fxw: float,
    drawdown: float,
    yc_spread: float,
) -> float:
    """
    Compute ML_Risk in [0,1].

    Inputs (all deterministic under Rev12.1b FD spec):
      vix       : volatility index level (previous close)
      hy_oas    : HY OAS in basis points
      fxw       : FX warning score in [0,1] (higher = safer KRW FX)
      drawdown  : long-horizon drawdown (e.g. -0.25 for -25%), from 3Y SPX
      yc_spread : 10Y - 2Y in basis points

    Interpretation:
      0.0 ~ 0.3 : benign
      0.3 ~ 0.7 : normal / noisy
      0.7 ~ 1.0 : stressed / crisis
    """
    # VIX: 12~40
    vix_n = norm(vix, 12.0, 40.0)

    # HY OAS: 250~700 bps
    hy_n = norm(hy_oas, 250.0, 700.0)

    # FXW: low FXW => high FX risk
    fx_risk_n = 1.0 - clip(fxw, 0.0, 1.0)

    # Drawdown: 5%~50% mapped (magnitude only)
    dd_mag = abs(min(drawdown, 0.0))
    dd_n = norm(dd_mag, 0.05, 0.50)

    # Yield curve inversion: only negative part
    yc_inv = -min(yc_spread, 0.0)
    yc_n = norm(yc_inv, 0.0, 120.0)

    # Weighted combination (sum weights = 1.0)
    raw = (
        0.25 * vix_n +
        0.25 * hy_n +
        0.20 * fx_risk_n +
        0.20 * dd_n +
        0.10 * yc_n
    )
    return clip(raw, 0.0, 1.0)


# =========================================================
# ML_Opp: opportunity index in [0,1]
# ---------------------------------------------------------
# Higher when drawdown is meaningful (cheapness),
# FXW is supportive, and volatility/credit stress
# are not at extreme crisis levels.
# =========================================================

def compute_ml_opp(
    vix: float,
    hy_oas: float,
    fxw: float,
    drawdown: float,
) -> float:
    """
    Compute ML_Opp in [0,1].

    Intuition:
      - Larger (reasonable) drawdowns → more opportunity
      - Supportive FX (high FXW)      → more opportunity
      - Extremely high VIX / HY OAS   → opportunity limited
    """
    # Drawdown opportunity: 5%~40% -> 0..1
    dd_mag = abs(min(drawdown, 0.0))
    dd_opp = norm(dd_mag, 0.05, 0.40)

    # FXW directly supportive
    fx_opp = clip(fxw, 0.0, 1.0)

    # Volatility and credit should not be extreme
    vix_n = norm(vix, 12.0, 50.0)    # 0~1 (higher = more volatile)
    hy_n = norm(hy_oas, 250.0, 800.0)

    # Penalize extreme stress:
    stress_penalty = 0.5 * max(0.0, vix_n - 0.6) + 0.5 * max(0.0, hy_n - 0.6)
    stress_penalty = clip(stress_penalty, 0.0, 1.0)

    base = 0.40 * dd_opp + 0.40 * fx_opp + 0.20 * (1.0 - stress_penalty)
    return clip(base, 0.0, 1.0)


# =========================================================
# ML_Regime: regime index in [0,1]
# ---------------------------------------------------------
# Higher when risk is high and opportunity is low,
# lower when risk is low and opportunity is high.
# =========================================================

def compute_ml_regime(ml_risk: float, ml_opp: float) -> float:
    """
    Compute ML_Regime in [0,1].

    Interpretation:
      0.0 ~ 0.3 : benign / calm regime
      0.3 ~ 0.7 : normal / noisy regime
      0.7 ~ 1.0 : stressed / crisis regime
    """
    ml_risk_c = clip(ml_risk, 0.0, 1.0)
    ml_opp_c = clip(ml_opp, 0.0, 1.0)
    raw = 0.5 + 0.4 * (ml_risk_c - 0.5) - 0.2 * (ml_opp_c - 0.5)
    return clip(raw, 0.0, 1.0)


# =========================================================
# Signals enrichment helper
# =========================================================

def enrich_signals_with_ml(
    signals: Any,
    hy_oas: float,
    yc_spread: float,
    fxw: float,
) -> Any:
    """
    Populate a Signals-like object with ML fields.

    Requirements on `signals`:
      - attributes: vix, drawdown
      - optional pre-existing: ml_risk, ml_opp, ml_regime

    After calling this function:
      - signals.ml_risk   is set to computed ML_Risk
      - signals.ml_opp    is set to computed ML_Opp
      - signals.ml_regime is set to computed ML_Regime
    """
    ml_risk = compute_ml_risk(
        vix=signals.vix,
        hy_oas=hy_oas,
        fxw=fxw,
        drawdown=signals.drawdown,
        yc_spread=yc_spread,
    )
    ml_opp = compute_ml_opp(
        vix=signals.vix,
        hy_oas=hy_oas,
        fxw=fxw,
        drawdown=signals.drawdown,
    )
    ml_regime = compute_ml_regime(ml_risk=ml_risk, ml_opp=ml_opp)

    signals.ml_risk = ml_risk
    signals.ml_opp = ml_opp
    signals.ml_regime = ml_regime
    return signals


__all__ = [
    "clip",
    "norm",
    "compute_ml_risk",
    "compute_ml_opp",
    "compute_ml_regime",
    "enrich_signals_with_ml",
]
