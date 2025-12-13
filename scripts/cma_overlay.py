from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import math
import numpy as np

# 데이터 경로 설정
DATA_DIR = Path("data")

# =========================
# TAS God Mode (Official)
# =========================
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
# CMA State (minimal, SELL gating only)
# =========================
@dataclass
class CMAState:
    ref_base_krw: float
    s0_count: int
    last_s0_yyyymm: str
    asof_yyyymm: str


def load_cma_state(path: Path = DATA_DIR / "cma_state.json") -> Optional[CMAState]:
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    return CMAState(
        ref_base_krw=float(obj.get("ref_base_krw", 0.0)),
        s0_count=int(obj.get("s0_count", 0)),
        last_s0_yyyymm=str(obj.get("last_s0_yyyymm", "")),
        asof_yyyymm=str(obj.get("asof_yyyymm", "")),
    )


def save_cma_state(st: CMAState, path: Path = DATA_DIR / "cma_state.json") -> None:
    payload = {
        "ref_base_krw": round(float(st.ref_base_krw), 2),
        "s0_count": int(st.s0_count),
        "last_s0_yyyymm": str(st.last_s0_yyyymm),
        "asof_yyyymm": str(st.asof_yyyymm),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
# =========================
def _round_to_5m(x: float) -> float:
    return round(x / 5_000_000) * 5_000_000


def plan_cma_action(
    asof_yyyymm: str,
    deployed_krw: float,
    cash_krw: float,
    operator_ref_base_krw: float,  
    fxw: float,
    vix: float,
    hy_oas: float,
    dd_mag_3y: float,         
    long_term_dd_10y: float, 
    ml_risk: float,
    systemic_bucket: str,
    final_state_name: str,
    prev_cma_state: Optional[CMAState],
) -> Dict:
    total = float(deployed_krw) + float(cash_krw)

    ref_base = float(operator_ref_base_krw) if operator_ref_base_krw is not None else 0.0
    if ref_base < 0:
        ref_base = 0.0
    ref_base_mode = "operator_fixed" if ref_base > 0 else "not_set"

    st = prev_cma_state or CMAState(
        ref_base_krw=ref_base,
        s0_count=0,
        last_s0_yyyymm="",
        asof_yyyymm=asof_yyyymm,
    )

    st.ref_base_krw = ref_base
    st.asof_yyyymm = asof_yyyymm

    if final_state_name.startswith("S0"):
        if st.last_s0_yyyymm != asof_yyyymm:
            st.s0_count += 1
            st.last_s0_yyyymm = asof_yyyymm
    else:
        st.s0_count = 0
        st.last_s0_yyyymm = ""

    thr = tas_threshold(vix=vix, long_term_dd_10y=long_term_dd_10y, hy_oas=hy_oas)
    deploy_factor = tas_deploy_factor(dd_mag=dd_mag_3y, thr=thr, ml_risk=ml_risk)

    target_deploy = ref_base * deploy_factor
    delta_raw = target_deploy - float(deployed_krw)

    deadband = 5_000_000
    exec_delta = 0.0

    fx_scale = fx_scale_from_fxw(fxw)
    buy_cap = min(0.15 * total, 20_000_000)
    sell_cap = min(0.10 * total, 20_000_000)

    if abs(delta_raw) < deadband:
        exec_delta = 0.0
    elif delta_raw > 0:
        buy_amt = min(delta_raw, float(cash_krw), buy_cap)
        buy_amt_scaled = _round_to_5m(buy_amt * fx_scale)
        if buy_amt_scaled >= 5_000_000:
            exec_delta = max(0.0, min(buy_amt_scaled, float(cash_krw)))
        else:
            exec_delta = 0.0
    else:
        if st.s0_count >= 2 and systemic_bucket in ("C0", "C1"):
            sell_amt = min(abs(delta_raw), sell_cap)
            sell_amt = round(sell_amt / 10_000_000) * 10_000_000
            if sell_amt >= 10_000_000:
                exec_delta = -sell_amt
            else:
                exec_delta = 0.0
        else:
            exec_delta = 0.0

    return {
        "asof_yyyymm": asof_yyyymm,
        "cma_snapshot": {
            "deployed_krw": float(deployed_krw),
            "cash_krw": float(cash_krw),
            "total_cma_krw": float(total),
            "ref_base_krw": float(ref_base),
            "ref_base_mode": ref_base_mode,
            "s0_count": int(st.s0_count),
            "last_s0_yyyymm": str(st.last_s0_yyyymm),
        },
        "tas": {
            "threshold": float(thr),
            "deploy_factor": float(deploy_factor),
            "target_deploy_krw": float(target_deploy),
            "delta_raw_krw": float(delta_raw),
        },
        "execution": {
            "fxw": float(fxw),
            "fx_scale": float(fx_scale),
            "buy_cap_krw": float(buy_cap),
            "sell_cap_krw": float(sell_cap),
            "suggested_exec_krw": float(exec_delta),
        },
        "_state_obj": st,
    }


def allocate_risk_on(exec_buy_krw: float, target_weights: Dict[str, float]) -> Dict[str, float]:
    basket = ["SPX", "NDX", "DIV", "EM", "ENERGY"]
    denom = sum(max(0.0, float(target_weights.get(k, 0.0))) for k in basket)
    if exec_buy_krw <= 0 or denom <= 0:
        return {k: 0.0 for k in basket}
    out: Dict[str, float] = {}
    for k in basket:
        w = max(0.0, float(target_weights.get(k, 0.0)))
        out[k] = exec_buy_krw * (w / denom)
    return out
