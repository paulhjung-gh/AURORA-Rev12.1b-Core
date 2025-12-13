import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone

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

# === Governance Protocol Clamp Values (운영 cap) ===
GOV_MAX_SGOV = 0.80
GOV_MAX_SAT = 0.20
GOV_MAX_DUR = 0.30

# === Monthly flow rule (ISA 165 / Direct 235 유지, Gold ISA 10 / Direct 10) ===
MONTHLY_ISA_KRW = 1_650_000
MONTHLY_DIRECT_KRW = 2_350_000
MONTHLY_TOTAL_KRW = MONTHLY_ISA_KRW + MONTHLY_DIRECT_KRW  # 4,000,000
MONTHLY_GOLD_ISA_KRW = 100_000
MONTHLY_GOLD_DIRECT_KRW = 100_000
MONTHLY_GOLD_TOTAL_KRW = MONTHLY_GOLD_ISA_KRW + MONTHLY_GOLD_DIRECT_KRW  # 200,000
# Gold sleeve weight = 200k / 4,000k = 5%
GOLD_SLEEVE_WEIGHT = MONTHLY_GOLD_TOTAL_KRW / MONTHLY_TOTAL_KRW  # 0.05

def _fail(msg: str) -> None:
    raise RuntimeError(msg)

# =========================
# P0-4: Unit Guards + Deterministic Normalization
# =========================
def _assert_range(name: str, x: float, lo: float, hi: float) -> float:
    if not (lo <= x <= hi):
        _fail(f"UNIT_GUARD_FAIL: {name}={x} (expected range {lo}..{hi})")
    return x

def _clean_float_series(xs: list, name: str) -> list[float]:
    if not isinstance(xs, list):
        _fail(f"MarketData invalid type: {name} must be list")
    out: list[float] = []
    for v in xs:
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isfinite(fv):
            continue
        if fv <= 0:
            continue
        out.append(fv)
    return out

def _normalize_pct_from_fraction(x: float, name: str) -> float:
    if 0.0 < x < 1.0:
        old = x
        x = x * 100.0
        print(f"[UNIT] {name} normalized: fraction->pct ({old} -> {x})")
    return x

def _normalize_policy_rate_pct(x: float, name: str) -> float:
    if 0.0 < x < 1.0:
        old = x
        x = x * 100.0
        print(f"[UNIT] {name} normalized: fraction->pct ({old} -> {x})")
    elif 50.0 <= x < 1000.0:
        old = x
        x = x / 100.0
        print(f"[UNIT] {name} normalized: bps->pct ({old} -> {x})")
    return x

def _normalize_macro_pct(x: float, name: str) -> float:
    if 0.0 < x < 1.0:
        old = x
        x = x * 100.0
        print(f"[UNIT] {name} normalized: fraction->pct ({old} -> {x})")
    return x

def build_signals(market: Dict[str, Any]) -> Dict[str, Any]:
    """
    마켓 데이터를 처리하고, 각종 신호를 계산합니다.
    Rev12.4 추가: yc_spread (10Y - 2Y) 계산
    """
    signals = {}
    signals['fxw'] = market.get('fxw', 0.0)
    signals['vix'] = market.get('vix', 0.0)
    signals['drawdown'] = market.get('drawdown', 0.0)
    signals['ml_risk'] = market.get('ml_risk', 0.0)
    signals['macro_score'] = market.get('macro_score', 0.0)
    signals['systemic_bucket'] = market.get('systemic_bucket', 'C0')
    signals['fx_rate'] = market['fx']['usdkrw']
    signals['ffr_upper'] = market['rates']['ffr_upper']
    signals['ml_opp'] = market.get('ml_opp', 0.5)

    # Rev12.4 추가: Yield Curve Spread (10Y - 2Y bps)
    dgs10 = market['rates'].get('dgs10', 0.0)
    dgs2 = market['rates'].get('dgs2', 0.0)
    signals['yc_spread'] = (dgs10 - dgs2) * 100  # bps 단위

    return signals

def calculate_alpha(asset: str, signals: Dict[str, float]) -> float:
    """
    자산군에 대한 알파를 계산합니다 (Rev12.4)
    """
    alpha = 0.0

    vix = signals["vix"]
    if vix >= 30:
        vix_alpha = -0.05
    elif vix <= 15:
        vix_alpha = 0.05
    else:
        vix_alpha = 0.0

    drawdown = signals["drawdown"]
    if drawdown <= -0.3:
        dd_alpha = -0.05
    elif drawdown >= 0:
        dd_alpha = 0.05
    else:
        dd_alpha = 0.0

    fxw = signals["fxw"]
    if fxw < 0.3:
        fxw_alpha = -0.05
    elif fxw > 0.7:
        fxw_alpha = 0.05
    else:
        fxw_alpha = 0.0

    ml_risk = signals["ml_risk"]
    if ml_risk >= 0.75:
        ml_alpha = -0.05
    elif ml_risk <= 0.4:
        ml_alpha = 0.05
    else:
        ml_alpha = 0.0

    systemic_bucket = signals["systemic_bucket"]
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

    alpha = max(-0.1, min(alpha, 0.1))
    return alpha

def compute_portfolio_target(sig: Dict[str, float]) -> Dict[str, float]:
    """
    Rev12.4: Core 내부 tilt 적용 버전
    """
    eng = AuroraX121()
    fxw = sig["fxw"]
    fx_rate = sig["fx_rate"]
    ffr_upper = sig["ffr_upper"]
    ml_risk = sig["ml_risk"]
    ml_opp = sig["ml_opp"]
    macro_score = sig["macro_score"]
    systemic_bucket = sig["systemic_bucket"]

    gold_w = float(GOLD_SLEEVE_WEIGHT)
    remaining = 1.0 - gold_w

    sgov_floor = eng.sgov_floor(
        fxw=fxw,
        fx_rate=fx_rate,
        ffr=ffr_upper,
        ml_risk=ml_risk,
        systemic=systemic_bucket,
    )
    sat_weight = eng.satellite_target(
        systemic=systemic_bucket,
        ml_opp=ml_opp,
        fxw=fxw,
    )
    dur_weight = eng.duration_target(
        macro_score=macro_score,
        fxw=fxw,
        ml_risk=ml_risk,
    )

    sgov_floor = max(0.0, min(GOV_MAX_SGOV, sgov_floor))
    sat_weight = max(0.0, min(GOV_MAX_SAT, sat_weight))
    dur_weight = max(0.0, min(GOV_MAX_DUR, dur_weight))

    tri_sum = sgov_floor + sat_weight + dur_weight
    if tri_sum > remaining and tri_sum > 0:
        scale = remaining / tri_sum
        sgov_floor *= scale
        sat_weight *= scale
        dur_weight *= scale

    core_weight = max(0.0, remaining - (sgov_floor + sat_weight + dur_weight))

    # Rev12.4 Core tilt 적용
    alpha_spx = calculate_alpha("SPX", sig)
    alpha_ndx = calculate_alpha("NDX", sig)
    alpha_div = calculate_alpha("DIV", sig)

    core_base = {
        "SPX": 0.525 + alpha_spx,
        "NDX": 0.245 + alpha_ndx,
        "DIV": 0.230 + alpha_div,
    }

    # Core 내부 normalize (총합 1 보장)
    core_sum = sum(core_base.values())
    core_alloc = {k: v / core_sum for k, v in core_base.items()}

    em_w = sat_weight * (2.0 / 3.0)
    en_w = sat_weight * (1.0 / 3.0)

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
        scale = 1.0 / total
        weights = {k: v * scale for k, v in weights.items()}

    return weights

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

def write_daily_report(
    sig: Dict[str, Any],
    final_weights: Dict[str, float],
    state_name: str,
    cma_overlay: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().date()
    yyyymmdd = today.strftime("%Y%m%d")
    out_path = REPORTS_DIR / f"aurora_daily_report_{yyyymmdd}.md"
    lines = []
    lines.append("# AURORA Rev12.4 Daily Report")
    lines.append("")
    lines.append(f"- Report Date: {today.isoformat()}")
    lines.append(f"- Engine Version: {meta.get('engine_version')}")
    lines.append(f"- Git Commit: {meta.get('git_commit')}")
    lines.append(f"- Run ID: {meta.get('run_id')}")
    lines.append(f"- Timestamp(UTC): {meta.get('timestamp_utc')}")
    lines.append("")
    lines.append("## 1. Market Data Summary (FD inputs)")
    if sig.get("fx_rate"):
        lines.append(f"- USD/KRW (Sell Rate): {sig['fx_rate']:.2f}")
    if sig.get("vix"):
        lines.append(f"- VIX: {sig['vix']:.2f}")
    if sig.get("hy_oas"):
        lines.append(f"- HY OAS: {sig['hy_oas']:.2f} bps")
    lines.append("")
    lines.append("## 2. CMA State and Overlay")
    lines.append(f"- Ref Base KRW: {cma_overlay['cma_snapshot']['ref_base_krw']}")
    lines.append(f"- S0 Count: {cma_overlay['cma_snapshot']['s0_count']}")
    lines.append(f"- Last S0 YYYMM: {cma_overlay['cma_snapshot']['last_s0_yyyymm']}")
    lines.append(f"- CMA Total: {cma_overlay['cma_snapshot']['total_cma_krw']}")
    lines.append(f"- Target Deploy (KRW): {cma_overlay['tas']['target_deploy_krw']}")
    lines.append(f"- Suggested Execution: {cma_overlay['execution']['suggested_exec_krw']}")
    lines.append("")
    lines.append("## 3. Target Weights (Portfolio 100%)")
    lines.append("| Asset | Weight (%) |")
    lines.append("|---------|-----------:|")
    total = 0.0
    for key in ["SPX", "NDX", "DIV", "EM", "ENERGY", "DURATION", "SGOV", "GOLD"]:
        w = final_weights.get(key, 0.0)
        total += w
        lines.append(f"| {key:7s} | {w*100:10.2f} |")
    lines.append(f"| **Total** | **{total*100:10.2f}** |")
    lines.append("")
    lines.append("## 4. CMA Overlay Allocation")
    for key, value in cma_overlay['risk_on_target_weights'].items():
        lines.append(f"- {key}: {value*100:.2f}%")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[REPORT] Daily report written to: {out_path}")
    
def determine_state_from_signals(sig: Dict[str, float]) -> str:
        """
        엔진 상태 결정 함수 (테스트용 + 실전용)
        - 간단하게 Systemic과 ML_Risk 기준으로 상태 반환
        - 필요 시 나중에 더 세밀하게 확장 가능
        """
        ml_risk = sig.get("ml_risk", 0.5)
        systemic = sig.get("systemic_bucket", "C0")

        if systemic == "C3" or ml_risk > 0.85:
            return "S3_HARD"
        if systemic == "C2" or ml_risk > 0.75:
            return "S3_SOFT"
        if sig.get("vix", 0) > 35:
            return "S2_HIGH_VOL"
        if ml_risk > 0.65:
            return "S1_MILD"
        return "S0_NORMAL"
    
    def compute_cma_overlay_section(sig: Dict[str, float], weights: Dict[str, float]) -> Dict[str, Any]:
        """
        실제 CMA Overlay 계산 (cma_overlay.py 함수들 활용)
        """
        cma_state = load_cma_state()

        tas_output = plan_cma_action(
            vix=sig.get("vix", 15.0),
            long_term_dd_10y=sig.get("drawdown", 0.0),
            hy_oas=sig.get("hy_oas", 300.0),
            ml_risk=sig.get("ml_risk", 0.5),
            state="S0_NORMAL",
            systemic_bucket=sig.get("systemic_bucket", "C0"),
        )

        exec_delta = tas_output.suggested_exec_krw(cma_state.cash_krw)

        risk_on_alloc = allocate_risk_on(exec_delta, weights)

        return {
            "cma_snapshot": {
                "ref_base_krw": cma_state.ref_base_krw,
                "s0_count": cma_state.s0_count,
                "last_s0_yyyymm": cma_state.last_s0_yyyymm,
                "total_cma_krw": cma_state.deployed_krw + cma_state.cash_krw,
            },
            "tas": {
                "threshold": tas_output.final_threshold,
                "deploy_factor": tas_output.deploy_factor,
                "target_deploy_krw": tas_output.deploy_factor * (cma_state.deployed_krw + cma_state.cash_krw),
            },
            "execution": {
                "suggested_exec_krw": exec_delta,
            },
            "risk_on_target_weights": risk_on_alloc,
        }

def main():
    market = load_latest_market()
    sig = build_signals(market)
    weights = compute_portfolio_target(sig)
    state_name = determine_state_from_signals(sig)  # 기존 함수 가정
    cma_overlay = compute_cma_overlay_section(sig, weights)  # 기존 함수 가정

    print("[INFO] ==== 3. FD / ML / Systemic Signals ====")
    print(f"FXW (KDE): {sig['fxw']:.3f}")
    print(f"FX Vol (21D σ): {sig.get('fx_vol', 0.0):.4f}")
    print(f"SPX 3Y Drawdown: {sig['drawdown']*100:.2f}%")
    print(f"MacroScore: {sig['macro_score']:.3f}")
    print(f"ML_Risk / ML_Opp / ML_Regime: {sig['ml_risk']:.3f} / {sig['ml_opp']:.3f} / {sig.get('ml_regime', 0.5):.3f}")
    print(f"Systemic Level / Bucket: {sig['systemic_level']:.3f} / {sig['systemic_bucket']}")
    print(f"Yield Curve Spread (10Y-2Y bps): {sig.get('yc_spread', 0.0):.1f}")

    print("[INFO] ==== 4. Engine State ====")
    print(f"Final State: {state_name}")

    print("[INFO] ==== 5. CMA Overlay Snapshot ====")
    print(f"Threshold: {cma_overlay['tas']['threshold']*100:.1f}%")
    print(f"Deploy Factor: {cma_overlay['tas']['deploy_factor']*100:.1f}%")

    today = datetime.now().strftime("%Y%m%d")
    out_path = DATA_DIR / f"aurora_target_weights_{today}.json"

    meta = {
        "engine_version": "AURORA-Rev12.4",
        "git_commit": os.getenv("GITHUB_SHA", ""),
        "run_id": os.getenv("GITHUB_RUN_ID", ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    out = {
        "date": today,
        "meta": meta,
        "signals": {**sig, "state": state_name},
        "weights": weights,
        "cma_overlay": cma_overlay,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[OK] Target weights JSON saved to: {out_path}")

    write_daily_report(sig, weights, state_name, cma_overlay, meta)

if __name__ == "__main__":
    main()
