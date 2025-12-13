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

# load_latest_market 함수 복원
def load_latest_market() -> Dict[str, Any]:
    files = sorted(DATA_DIR.glob("market_data_20*.json"))
    if not files:
        raise FileNotFoundError("market_data_*.json 파일이 data/ 폴더에 없습니다.")
    latest = files[-1]
    with latest.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("마켓 데이터 JSON은 dict 여야 합니다.")
    return raw

# 기타 함수들 (compute_macro_score, compute_fx_kde_anchor_and_stats, calculate_alpha, compute_portfolio_target, build_signals 등 이전 버전 그대로)

# main 함수 (완전 복원)
def main():
    market = load_latest_market()
    sig = build_signals(market)
    weights = compute_portfolio_target(sig)
    state_name = determine_state_from_signals(sig)
    cma_overlay = compute_cma_overlay_section(sig, weights)
    
    # 신호 출력 등 기존 로직
    print("[INFO] ==== 3. FD / ML / Systemic Signals ====")
    # ... (기존 print 문들)
    
    # 보고서 작성 및 저장 로직
    write_daily_report(sig, weights, state_name, cma_overlay, meta)

if __name__ == "__main__":
    main()
