from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

DATA_DIR = Path("data")

# =========================
# TAS God Mode (Official)
# =========================
# base_threshold = 18%
# VIX >=30 -> 16%, VIX <=15 -> 20%, else 18%
# long_term_dd <= -15% (10Y peak-to-now) -> +4%
# HY_OAS >= 500 -> +2%
# deploy_raw = (DD - thr)/(0.60 - thr), clip 0..1
# if DD >= 40% -> +0.20 (cap 1.0)
# if ML_Risk >= 0.75 -> *0.5
# (DD is positive magnitude here, e.g., 0.22 for -22%)
# (10Y dd input is negative, e.g., -0.17)

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def tas_threshold(vix: float, long_term_dd_10y: float, hy_oas: float) -> float:
    if vix >= 30:
        thr = 0.16
    elif vix <= 15:
        thr = 0.20
    else:
        thr = 0.18

    if long_term_dd_10y <= -0.15:
        thr += 0.04
    if hy_oas >= 500:
        thr += 0.02
    return thr

def tas_deploy_factor(dd_mag: float, thr: float, ml_risk: float) -> float:
    if dd_mag < thr:
        return 0.0
    denom = (0.60 - thr)
    if denom <= 1e-9:
        return 0.0
    deploy = (dd_mag - thr) / denom
    deploy = _clip(deploy, 0.0, 1.0)

    if dd_mag >= 0.40:
        deploy = min(1.0, deploy + 0.20)

    if ml_risk >= 0.75:
        deploy *= 0.50

    return _clip(deploy, 0.0, 1.0)

# =========================
# FXW scaler (BUY only)
# fx_scale = 0.7 + 0.6*FXW
# =========================
def fx_scale_from_fxw(fxw: float) -> float:
    return 0.7 + 0.6 * _clip(fxw, 0.0, 1.0)

# =========================
# CMA State (minimal)
# =========================
@dataclass
class CMAState:
    ref_base_krw: float
    s0_count: int
    asof_yyyymm: str

def load_cma_state(path: Path = DATA_DIR / "cma_state.json") -> Optional[CMAState]:
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    return CMAState(
        ref_base_krw=float(obj.get("ref_base_krw", 0.0)),
        s0_count=int(obj.get("s0_count", 0)),
        asof_yyyymm=str(obj.get("asof_yyyymm", "")),
    )

def save_cma_state(st: CMAState, path: Path = DATA_DIR / "cma_state.json") -> None:
    path.write_text(
        json.dumps(
            {
                "ref_base_krw": round(float(st.ref_base_krw), 2),
                "s0_count": int(st.s0_count),
                "asof_yyyymm": st.asof_yyyymm,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

# =========================
# Rolling Expansion
# - Operator-ref_base 고정 정책과 충돌하므로 기본 비활성화
# =========================
def maybe_roll_ref_base(
    total_cma: float,
    dd_mag: float,
    state_name: str,
    ml_risk: float,
    systemic_bucket: str,
    st: CMAState,
) -> Tuple[CMAState, float]:
    # Disabled by design (operator inputs ref_base_krw)
    return st, 0.0

# =========================
# Execution constraints
# BUY: min ticket 5,000,000 KRW, 5M rounding
# CAP: min(0.15*total, 20M)
# Deadband: |delta| < 5M -> No Action
#
# SELL: min ticket 10,000,000 KRW, CAP min(0.10*total, 20M)
# =========================
def _round_to_5m(x: float) -> float:
    return round(x / 5_000_000) * 5_000_000

def plan_cma_action(
    asof_yyyymm: str,
    deployed_krw: float,
    cash_krw: float,
    operator_ref_base_krw: float,   # ← single source input (from insert→data)
    # engine signals
    fxw: float,
    vix: float,
    hy_oas: float,
    dd_mag_3y: float,          # magnitude, e.g. 0.22
    long_term_dd_10y: float,   # negative, e.g. -0.17
    ml_risk: float,
    systemic_bucket: str,
    final_state_name: str,
    # for SELL gating
    prev_cma_state: Optional[CMAState],
) -> Dict:
    total = deployed_krw + cash_krw

    # === ref_base policy ===
    # Priority:
    # 1) operator_ref_base_krw > 0 : operator fixed baseline
    # 2) prev_state.ref_base_krw > 0 : keep existing baseline (post-first-buy fixed)
    # 3) else 0 : baseline not set -> CMA buy inactive by design
    op = float(operator_ref_base_krw or 0.0)
    prev_ref = float(prev_cma_state.ref_base_krw) if (prev_cma_state and prev_cma_state.ref_base_krw) else 0.0

    if op > 0:
        ref_base = op
        ref_base_mode = "operator_fixed"
    elif prev_ref > 0:
        ref_base = prev_ref
        ref_base_mode = "state_fixed"
    else:
        ref_base = 0.0
        ref_base_mode = "not_set"

    # init/update state (only s0_count + asof tracking; ref_base follows policy above)
    st = prev_cma_state or CMAState(ref_base_krw=ref_base, s0_count=0, asof_yyyymm=asof_yyyymm)

    # update s0_count (SELL gate용: S0 연속 개월 수)
    if final_state_name.startswith("S0"):
        st.s0_count_
