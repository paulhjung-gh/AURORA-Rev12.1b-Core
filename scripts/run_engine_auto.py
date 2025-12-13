import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone
from scipy.stats import gaussian_kde
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from engine.aurora_engine import AuroraX121
from engine.ml_layer import (
    compute_ml_risk,
    compute_ml_opp,
    compute_ml_regime,
    clip,
    norm,
)
from engine.systemic_layer import (
    compute_systemic_level,
    determine_systemic_bucket,
)
from scripts.cma_overlay import (
    load_cma_state,
    save_cma_state,
    plan_cma_action,
    allocate_risk_on,
)

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

GOV_MAX_SGOV = 0.80
GOV_MAX_SAT = 0.20
GOV_MAX_DUR = 0.30

MONTHLY_ISA_KRW = 1_650_000
MONTHLY_DIRECT_KRW = 2_350_000
MONTHLY_TOTAL_KRW = MONTHLY_ISA_KRW + MONTHLY_DIRECT_KRW
MONTHLY_GOLD_ISA_KRW = 100_000
MONTHLY_GOLD_DIRECT_KRW = 100_000
MONTHLY_GOLD_TOTAL_KRW = MONTHLY_GOLD_ISA_KRW + MONTHLY_GOLD_DIRECT_KRW
GOLD_SLEEVE_WEIGHT = MONTHLY_GOLD_TOTAL_KRW / MONTHLY_TOTAL_KRW  # 0.05

def _fail(msg: str) -> None:
    raise RuntimeError(msg)

def compute_macro_score(ism: float, pmi: float, cpi_yoy: float, unemployment: float) -> float:
    ism_n = norm(ism, 45, 60)
    pmi_n = norm(pmi, 45, 60)
    cpi_n = 1 - norm(cpi_yoy, 2, 8)
    unemp_n = 1 - norm(unemployment, 3, 7)
    return clip(0.25 * (ism_n + pmi_n + cpi_n + unemp_n), 0, 1)

def compute_fx_kde_anchor_and_stats(fx_hist_130d: list) -> Dict[str, Any]:
    data = np.array(fx_hist_130d)
    kde = gaussian_kde(data)
    x = np.linspace(data.min() - 100, data.max() + 100, 1000)
    density = kde(x)
    anchor = x[np.argmax(density)]
    return {"kde_anchor": anchor}

# Rev12.4 알파 틸트 복원
def calculate_alpha(asset: str, sig: Dict[str, float]) -> float:
    vix = sig["vix"]
    if vix >= 30:
        vix_alpha = -0.05
    elif vix <= 15:
        vix_alpha = 0.05
    else:
        vix_alpha = 0.0

    drawdown = sig["drawdown"]
    if drawdown <= -0.30:
        dd_alpha = -0.05
    elif drawdown >= 0:
        dd_alpha = 0.05
    else:
        dd_alpha = 0.0

    fxw = sig["fxw"]
    if fxw < 0.3:
        fxw_alpha = -0.05
    elif fxw > 0.7:
        fxw_alpha = 0.05
    else:
        fxw_alpha = 0.0

    ml_risk = sig["ml_risk"]
    if ml_risk >= 0.75:
        ml_alpha = -0.05
    elif ml_risk <= 0.40:
        ml_alpha = 0.05
    else:
        ml_alpha = 0.0

    systemic_bucket = sig["systemic_bucket"]
    if systemic_bucket in ["C2", "C3"]:
        sys_alpha = -0.05
    elif systemic_bucket == "C0":
        sys_alpha = 0.05
    else:
        sys_alpha = 0.0

    if asset == "SPX":
        alpha = vix_alpha + dd_alpha + fxw_alpha + ml_alpha + sys_alpha
    elif asset == "NDX":
        alpha = (vix_alpha * 1.5) + (dd_alpha * 1.2) + (fxw_alpha * 1.3) + (ml_alpha * 1.4) + (sys_alpha * 1.2)
    elif asset == "DIV":
        alpha = (vix_alpha * 0.5) + (dd_alpha * 0.8) + (fxw_alpha * 0.7) + (ml_alpha * 0.6) + (sys_alpha * 0.5)
    else:
        alpha = 0.0

    return max(-0.10, min(0.10, alpha))

def compute_portfolio_target(sig: Dict[str, float]) -> Dict[str, float]:
    eng = AuroraX121()
    fxw = sig["fxw"]
    fx_rate = sig["fx_rate"]
    ffr_upper = sig["ffr_upper"]
    ml_risk = sig["ml_risk"]
    ml_opp = sig["ml_opp"]
    macro_score = sig["macro_score"]
    systemic_bucket = sig["systemic_bucket"]
    gold_w = GOLD_SLEEVE_WEIGHT
    remaining = 1.0 - gold_w

    sgov_floor = eng.sgov_floor(fxw=fxw, fx_rate=fx_rate, ffr=ffr_upper, ml_risk=ml_risk, systemic=systemic_bucket)
    sat_weight = eng.satellite_target(systemic=systemic_bucket, ml_opp=ml_opp, fxw=fxw)
    dur_weight = eng.duration_target(macro_score=macro_score, fxw=fxw, ml_risk=ml_risk)

    sgov_floor = clip(sgov_floor, 0.22, GOV_MAX_SGOV)
    sat_weight = clip(sat_weight, 0.0, GOV_MAX_SAT)
    dur_weight = clip(dur_weight, 0.0, GOV_MAX_DUR)

    tri_sum = sgov_floor + sat_weight + dur_weight
    if tri_sum > remaining > 0:
        scale = remaining / tri_sum
        sgov_floor *= scale
        sat_weight *= scale
        dur_weight *= scale

    core_weight = remaining - (sgov_floor + sat_weight + dur_weight)

    # 알파 틸트 적용
    alpha_spx = calculate_alpha("SPX", sig)
    alpha_ndx = calculate_alpha("NDX", sig)
    alpha_div = calculate_alpha("DIV", sig)

    core_base = {
        "SPX": 0.525 + alpha_spx,
        "NDX": 0.245 + alpha_ndx,
        "DIV": 0.230 + alpha_div,
    }
    core_sum = sum(core_base.values())
    core_alloc = {k: v / core_sum for k, v in core_base.items()}

    em_w = sat_weight * (2/3)
    en_w = sat_weight * (1/3)

    weights = {
        "SPX": core_alloc["SPX"] * core_weight,
        "NDX": core_alloc["NDX"] * core_weight,
        "DIV": core_alloc["DIV"] * core_weight,
        "EM": em_w,
        "ENERGY": en_w,
        "DURATION": dur_weight,
        "SGOV": sgov_floor,
        "GOLD": gold_w,
    }

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights

# 나머지 함수는 이전 수정본과 동일 (CMA 등)

if __name__ == "__main__":
    main()
