from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import json
import math
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
# Spec: :contentReference[oaicite:3]{index=3}

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
# Rolling Expansion (ref_base 방식)
# eligible = total - ref_base
# add = eligible * 0.25
# 조건: DD>=20%, state=S1_MILD, ML_Risk<0.70, systemic C0/C1
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
        ref_base_krw=float(obj["ref_base_krw"]),
        s0_count=int(obj.get("s0_count", 0)),
        asof_yyyymm=str(obj.get("asof_yyyymm", "")),
    )

def save_cma_state(st: CMAState, path: Path = DATA_DIR / "cma_state.json") -> None:
    path.write_text(json.dumps({
        "ref_base_krw": round(st.ref_base_krw, 2),
        "s0_count": int(st.s0_count),
        "asof_yyyymm": st.asof_yyyymm
    }, ensure_ascii=False, indent=2), encoding="utf-8")

def maybe_roll_ref_base(
    total_cma: float,
    dd_mag: float,
    state_name: str,
    ml_risk: float,
    systemic_bucket: str,
    st: CMAState,
) -> Tuple[CMAState, float]:
    eligible = max(0.0, total_cma - st.ref_base_krw)
    add = 0.0

    cond = (
        dd_mag >= 0.20 and
        state_name == "S1_MILD" and
        ml_risk < 0.70 and
        systemic_bucket in ("C0", "C1")
    )
    if cond and eligible > 0:
        add = eligible * 0.25
        st.ref_base_krw += add

    return st, add

# =========================
# Execution constraints
# BUY: min ticket 5,000,000 KRW, 5M rounding
# CAP: min(0.15*total, 20M)
# Deadband: |delta| < 5M -> No Action
#
# SELL: min ticket 10,000,000 KRW, CAP min(0.10*total, 20M)
# (SELL 조건은 “S0_NORMAL 2~3개월 연속 + systemic C0/C1 + min ticket”)
# =========================
def _round_to_5m(x: float) -> float:
    return round(x / 5_000_000) * 5_000_000

def plan_cma_action(
    asof_yyyymm: str,
    deployed_krw: float,
    cash_krw: float,
    # engine signals
    fxw: float,
    vix: float,
    hy_oas: float,
    dd_mag_3y: float,          # magnitude, e.g. 0.22
    long_term_dd_10y: float,   # negative, e.g. -0.17 (없으면 0으로 넣어도 됨)
    ml_risk: float,
    systemic_bucket: str,
    final_state_name: str,
    # for SELL gating
    prev_cma_state: Optional[CMAState],
) -> Dict:
    total = deployed_krw + cash_krw

    # init cma_state
    st = prev_cma_state or CMAState(ref_base_krw=total, s0_count=0, asof_yyyymm=asof_yyyymm)

    # update s0_count (2~3 months consecutive)
    if final_state_name == "S0_NORMAL":
        st.s0_count += 1
    else:
        st.s0_count = 0

    # rolling expansion (partial)
    st.asof_yyyymm = asof_yyyymm
    st, ref_base_add = maybe_roll_ref_base(
        total_cma=total,
        dd_mag=dd_mag_3y,
        state_name=final_state_name,
        ml_risk=ml_risk,
        systemic_bucket=systemic_bucket,
        st=st,
    )

    ref_base = st.ref_base_krw

    # ---------- BUY (TAS God Mode) ----------
    thr = tas_threshold(vix=vix, long_term_dd_10y=long_term_dd_10y, hy_oas=hy_oas)
    deploy_factor = tas_deploy_factor(dd_mag=dd_mag_3y, thr=thr, ml_risk=ml_risk)

    target_deploy = ref_base * deploy_factor
    delta_raw = target_deploy - deployed_krw  # + => BUY, - => SELL target

    deadband = 5_000_000
    action = "NO_ACTION"
    exec_delta = 0.0

    fx_scale = fx_scale_from_fxw(fxw)
    buy_cap = min(0.15 * total, 20_000_000)
    sell_cap = min(0.10 * total, 20_000_000)

    # BUY execution (only if cash available)
    if abs(delta_raw) < deadband:
        action = "NO_ACTION"
        exec_delta = 0.0
    elif delta_raw > 0:
        # planned buy
        buy_amt = min(delta_raw, cash_krw, buy_cap)
        buy_amt = _round_to_5m(buy_amt)
        if buy_amt >= 5_000_000:
            # FXW applies to BUY only
            buy_amt_scaled = _round_to_5m(buy_amt * fx_scale)
            exec_delta = max(0.0, min(buy_amt_scaled, cash_krw))
            action = "BUY" if exec_delta >= 5_000_000 else "NO_ACTION"
        else:
            action = "NO_ACTION"
            exec_delta = 0.0
    else:
        # SELL gating: very rare
        # condition: S0_NORMAL 2~3개월 연속 + systemic C0/C1 + min ticket
        if st.s0_count >= 2 and systemic_bucket in ("C0", "C1"):
            sell_amt = min(abs(delta_raw), sell_cap)
            # SELL min ticket 10M
            sell_amt = round(sell_amt / 10_000_000) * 10_000_000
            if sell_amt >= 10_000_000:
                exec_delta = -sell_amt
                action = "SELL"
            else:
                action = "NO_ACTION"
                exec_delta = 0.0
        else:
            action = "NO_ACTION"
            exec_delta = 0.0

    return {
        "asof_yyyymm": asof_yyyymm,
        "cma_snapshot": {
            "deployed_krw": deployed_krw,
            "cash_krw": cash_krw,
            "total_cma_krw": total,
            "ref_base_krw": ref_base,
            "ref_base_add_krw": ref_base_add,
            "s0_count": st.s0_count
        },
        "tas": {
            "threshold": thr,
            "deploy_factor": deploy_factor,
            "target_deploy_krw": target_deploy,
            "delta_raw_krw": delta_raw
        },
        "execution": {
            "fxw": fxw,
            "fx_scale": fx_scale,
            "buy_cap_krw": buy_cap,
            "sell_cap_krw": sell_cap,
            "deadband_krw": deadband,
            "action": action,
            "exec_delta_krw": exec_delta
        },
        "_state_obj": st  # caller saves
    }

def allocate_risk_on(exec_buy_krw: float, target_weights: Dict[str, float]) -> Dict[str, float]:
    """
    CMA는 Risk-on basket only: SPX/NDX/DIV/EM/ENERGY
    분배는 '현재비중'이 아니라 '엔진 타겟 weight' 기준.
    """
    basket = ["SPX", "NDX", "DIV", "EM", "ENERGY"]
    denom = sum(max(0.0, float(target_weights.get(k, 0.0))) for k in basket)
    if exec_buy_krw <= 0 or denom <= 0:
        return {k: 0.0 for k in basket}
    out = {}
    for k in basket:
        w = max(0.0, float(target_weights.get(k, 0.0)))
        out[k] = exec_buy_krw * (w / denom)
    return out
