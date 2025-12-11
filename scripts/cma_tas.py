# aurora/strategy/cma_tas.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any


def clip(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi]."""
    return max(lo, min(hi, x))


@dataclass
class CmaTasInput:
    """
    Input container for CMA TAS Dynamic Threshold overlay.

    All inputs are deterministic FD/ML/Systemic outputs from Rev12.1b engine.
    - vix             : VIX level (previous close)
    - hy_oas          : HY OAS in basis points
    - dd_3y           : 3Y SPX long-horizon drawdown (e.g. -0.25 for -25%)
    - dd_10y          : 10Y high vs current drawdown (e.g. -0.20 for -20%)
    - ml_risk         : ML_Risk in [0,1]
    - state           : engine state label, e.g. "S0_NORMAL", "S1_MILD", ...
    - systemic_bucket : systemic bucket label, e.g. "C0", "C1", "C2", "C3"
    """
    vix: float
    hy_oas: float
    dd_3y: float
    dd_10y: float
    ml_risk: float
    state: str
    systemic_bucket: str


@dataclass
class CmaTasOutput:
    """
    Output container for CMA TAS Dynamic Threshold overlay.

    - deploy_factor   : final deploy ratio in [0,1]
    - final_threshold : DD threshold used to decide deploy
    - meta            : debug / trace info for operators
    """
    deploy_factor: float
    final_threshold: float
    meta: Dict[str, Any]


def _normalize_dd(dd_3y: float) -> float:
    """
    Convert 3Y drawdown from engine convention (negative percentage) into
    a positive fraction in [0, 0.60] for TAS logic.

    Example:
      dd_3y = -0.25 (=-25%) -> 0.25
      dd_3y =  0.00         -> 0.00

    We cap at 0.60 (i.e. 60% drawdown).
    """
    dd = max(0.0, -dd_3y)  # engine: -0.25 = -25% DD
    return clip(dd, 0.0, 0.60)


def compute_cma_tas(inp: CmaTasInput) -> CmaTasOutput:
    """
    Compute CMA TAS deploy factor and final threshold.

    High-level logic:
      1) If state/systemic gating says "no CMA", return deploy=0.
      2) Build Dynamic Threshold from:
         - base 18%
         - VIX regime (16~20%)
         - long-term 10Y DD (Japan-proof)
         - HY OAS credit stress
      3) Convert 3Y DD into [0,0.60] and compute deploy_raw.
      4) Apply deep crash enhancement (DD >= 40%).
      5) Apply ML_Risk penalty (ML_Risk >= 0.75).
      6) Clip into [0,1] and return.
    """

    meta: Dict[str, Any] = {
        "input": asdict(inp),
        "gates": {},
        "threshold_components": {},
        "deploy_components": {},
    }

    # -------------------------------------------------------
    # 0. State / Systemic gating (Governance-level hard filters)
    # -------------------------------------------------------

    # Allow both full labels ("S0_NORMAL") and shorthand ("S0")
    state = inp.state
    if state in ("S0", "S1"):
        normalized_state = state
    elif state.startswith("S0_"):
        normalized_state = "S0"
    elif state.startswith("S1_"):
        normalized_state = "S1"
    else:
        normalized_state = state

    # 기본 규칙:
    # - S0, S1 에서만 CMA TAS 활성
    # - Systemic C2, C3에서는 CMA 중단
    state_ok = normalized_state in ("S0", "S1")
    systemic_ok = inp.systemic_bucket in ("C0", "C1")

    meta["gates"]["normalized_state"] = normalized_state
    meta["gates"]["systemic_bucket"] = inp.systemic_bucket
    meta["gates"]["state_ok"] = state_ok
    meta["gates"]["systemic_ok"] = systemic_ok

    if not state_ok or not systemic_ok:
        # Hard stop: 자동으로 CMA overlay를 꺼버리는 구간
        meta["gates"]["reason"] = "blocked_by_state_or_systemic"
        return CmaTasOutput(
            deploy_factor=0.0,
            final_threshold=0.0,
            meta=meta,
        )

    # -------------------------------------------------------
    # 1. Dynamic Threshold 계산
    # -------------------------------------------------------

    base_threshold = 0.18
    thr = base_threshold

    # 1-1. 변동성 조정 (VIX regime)
    if inp.vix >= 30.0:
        thr = 0.16
        vix_regime = "high"
    elif inp.vix <= 15.0:
        thr = 0.20
        vix_regime = "low"
    else:
        thr = 0.18
        vix_regime = "normal"

    meta["threshold_components"]["base_threshold"] = base_threshold
    meta["threshold_components"]["vix"] = inp.vix
    meta["threshold_components"]["vix_regime"] = vix_regime
    meta["threshold_components"]["after_vix"] = thr

    # 1-2. 장기침체(Japan-proof) 조정: 10Y high DD ≤ -15%
    if inp.dd_10y <= -0.15:
        thr += 0.04
        meta["threshold_components"]["long_term_dd_triggered"] = True
    else:
        meta["threshold_components"]["long_term_dd_triggered"] = False

    meta["threshold_components"]["dd_10y"] = inp.dd_10y
    meta["threshold_components"]["after_long_term_dd"] = thr

    # 1-3. 크레딧 스트레스 조정: HY_OAS ≥ 500bp
    if inp.hy_oas >= 500.0:
        thr += 0.02
        meta["threshold_components"]["hy_oas_triggered"] = True
    else:
        meta["threshold_components"]["hy_oas_triggered"] = False

    meta["threshold_components"]["hy_oas"] = inp.hy_oas

    final_threshold = thr
    meta["threshold_components"]["final_threshold"] = final_threshold

    # -------------------------------------------------------
    # 2. Deploy 규칙
    # -------------------------------------------------------

    # 2-1. DD 정규화
    dd = _normalize_dd(inp.dd_3y)
    meta["deploy_components"]["dd_3y_raw"] = inp.dd_3y
    meta["deploy_components"]["dd_3y_normalized"] = dd

    if dd < final_threshold:
        deploy_raw = 0.0
        meta["deploy_components"]["below_threshold"] = True
    else:
        meta["deploy_components"]["below_threshold"] = False
        denom = (0.60 - final_threshold)
        if denom <= 0.0:
            # 방어 코드
            deploy_raw = 1.0
            meta["deploy_components"]["denom_zero_guard"] = True
        else:
            deploy_raw = (dd - final_threshold) / denom
        deploy_raw = clip(deploy_raw, 0.0, 1.0)

    meta["deploy_components"]["deploy_raw_after_linear"] = deploy_raw

    # 2-2. Deep crash enhancement (DD ≥ 40%)
    if dd >= 0.40:
        deploy_raw = min(1.0, deploy_raw + 0.20)
        meta["deploy_components"]["deep_crash_boost_applied"] = True
    else:
        meta["deploy_components"]["deep_crash_boost_applied"] = False

    meta["deploy_components"]["deploy_raw_after_deep_crash"] = deploy_raw

    # 2-3. ML_Risk penalty (ML_Risk ≥ 0.75)
    if inp.ml_risk >= 0.75:
        deploy_raw *= 0.50
        meta["deploy_components"]["ml_risk_penalty_applied"] = True
    else:
        meta["deploy_components"]["ml_risk_penalty_applied"] = False

    meta["deploy_components"]["ml_risk"] = inp.ml_risk
    meta["deploy_components"]["deploy_raw_after_ml_risk"] = deploy_raw

    # 2-4. 최종 clip
    deploy_factor = clip(deploy_raw, 0.0, 1.0)
    meta["deploy_components"]["deploy_factor"] = deploy_factor

    return CmaTasOutput(
        deploy_factor=deploy_factor,
        final_threshold=final_threshold,
        meta=meta,
    )
