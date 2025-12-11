
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


def compute_systemic_level(
    hy_oas: float,
    yc_spread: float,
    macro_score: float,
    ml_regime: float,
    drawdown: float,
) -> float:
    """
    Compute SystemicLevel in [0,1] for Rev12.1b.

    Inputs (all deterministic under 12.1b FD spec):
      hy_oas      : HY OAS (bps)
      yc_spread   : 10Y - 2Y (bps)
      macro_score : macro environment score in [0,1] (higher = better),
                    computed from ISM/PMI/CPI YoY/Unemployment
      ml_regime   : ML_Regime in [0,1]
      drawdown    : long-horizon drawdown (e.g. -0.25 for -25%) from 3Y SPX
    """
    # 1) Credit stress
    s_hy = norm(hy_oas, 250.0, 700.0)

    # 2) Yield curve inversion (only negative part)
    yc_inv = -min(yc_spread, 0.0)
    s_yc = norm(yc_inv, 0.0, 120.0)

    # 3) Macro (low macro -> high systemic risk)
    s_macro = 1.0 - clip(macro_score, 0.0, 1.0)

    # 4) Regime
    s_regime = clip(ml_regime, 0.0, 1.0)

    # 5) Long-horizon drawdown
    d = abs(min(drawdown, 0.0))
    s_dd = norm(d, 0.15, 0.50)

    raw = (
        0.30 * s_hy +
        0.20 * s_yc +
        0.20 * s_macro +
        0.20 * s_regime +
        0.10 * s_dd
    )
    return clip(raw, 0.0, 1.0)


def determine_systemic_bucket(systemic_level: float) -> str:
    """
    Map systemic_level into C0/C1/C2/C3 according to Rev12.1b thresholds.

    Thresholds (unchanged from Rev12.1):
      <0.55 → C0
      0.55–0.69 → C1
      0.70–0.79 → C2
      ≥0.80 → C3
    """
    if systemic_level >= 0.80:
        return "C3"
    if systemic_level >= 0.70:
        return "C2"
    if systemic_level >= 0.55:
        return "C1"
    return "C0"


def enrich_signals_with_systemic(
    signals: Any,
    hy_oas: float,
    yc_spread: float,
) -> Any:
    """
    Attach systemic_level and systemic_bucket to a Signals-like object.

    Requirements on `signals`:
      - attributes: macro_score, ml_regime, drawdown

    Under Rev12.1b, macro_score and drawdown are themselves
    deterministic:
      - macro_score from MacroScore(ISM, PMI, CPI YoY, Unemployment)
      - drawdown from 3Y SPX peak-to-current

    After calling this function:
      - signals.systemic_level  is set to computed SystemicLevel
      - signals.systemic_bucket is set to corresponding C0–C3 bucket
    """
    systemic_level = compute_systemic_level(
        hy_oas=hy_oas,
        yc_spread=yc_spread,
        macro_score=getattr(signals, "macro_score", 0.5),
        ml_regime=getattr(signals, "ml_regime", 0.5),
        drawdown=getattr(signals, "drawdown", 0.0),
    )
    systemic_bucket = determine_systemic_bucket(systemic_level)

    signals.systemic_level = systemic_level
    signals.systemic_bucket = systemic_bucket
    return signals


__all__ = [
    "clip",
    "norm",
    "compute_systemic_level",
    "determine_systemic_bucket",
    "enrich_signals_with_systemic",
]
