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
GOV_MAX_SAT  = 0.20
GOV_MAX_DUR  = 0.30

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
# - "정규화 가능한 범위"에서는 자동 정규화(로그 남김)
# - 그 외는 strict fail
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
    """
    Some sources deliver rates as fractions (e.g., 0.0413) instead of percent (4.13).
    Deterministic rule:
      - if 0 < x < 1: treat as fraction and convert to percent (×100)
    """
    if 0.0 < x < 1.0:
        old = x
        x = x * 100.0
        print(f"[UNIT] {name} normalized: fraction->pct ({old} -> {x})")
    return x


def _normalize_policy_rate_pct(x: float, name: str) -> float:
    """
    FFR Upper may appear as:
      - percent: 3.75
      - bps: 375
      - fraction: 0.0375
    Deterministic rule (no guessing):
      - if 0 < x < 1: fraction -> percent (×100)
      - elif 50 <= x < 1000: bps -> percent (/100)
      - else: assume percent
    """
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
    """
    CPI YoY / Unemployment may appear as:
      - percent: 3.02
      - fraction: 0.0302
    Deterministic rule:
      - if 0 < x < 1: fraction -> percent (×100)
    """
    if 0.0 < x < 1.0:
        old = x
        x = x * 100.0
        print(f"[UNIT] {name} normalized: fraction->pct ({old} -> {x})")
    return x


def calculate_alpha(asset: str, signals: Dict[str, float]) -> float:
    """
    자산군에 대한 알파를 계산합니다.
    - asset: 'SPX', 'NDX', 'DIV' 중 하나
    - signals: {'vix': vix_value, 'drawdown': drawdown_value, 'fxw': fxw_value, 'ml_risk': ml_risk_value, 'systemic_bucket': systemic_bucket_value, ...}
    """
   
    # 기본 알파 값은 0
    alpha = 0.0
   
    # VIX에 따른 조정
    vix = signals["vix"]
    if vix >= 30:
        vix_alpha = -0.05  # VIX가 높으면 리스크가 커져서 비중을 낮춤
    elif vix <= 15:
        vix_alpha = 0.05  # VIX가 낮으면 비중을 늘림
    else:
        vix_alpha = 0.0  # VIX가 중간이면 변화 없음
   
    # Drawdown에 따른 조정
    drawdown = signals["drawdown"]
    if drawdown <= -0.3:
        dd_alpha = -0.05  # 큰 하락은 자산 비중을 줄임
    elif drawdown >= 0:
        dd_alpha = 0.05  # 하락이 없거나 상승하면 비중을 늘림
    else:
        dd_alpha = 0.0
   
    # FXW에 따른 조정
    fxw = signals["fxw"]
    if fxw < 0.3:
        fxw_alpha = -0.05  # FXW가 낮으면 리스크가 커져서 비중을 줄임
    elif fxw > 0.7:
        fxw_alpha = 0.05  # FXW가 높으면 안전하므로 비중을 늘림
    else:
        fxw_alpha = 0.0
   
    # ML_Risk에 따른 조정 (합성 리스크 지표 반영)
    ml_risk = signals["ml_risk"]
    if ml_risk >= 0.75:
        ml_alpha = -0.05  # ML_Risk 높으면 전체 리스크 커져 비중 줄임
    elif ml_risk <= 0.4:
        ml_alpha = 0.05  # ML_Risk 낮으면 안전하므로 비중 늘림
    else:
        ml_alpha = 0.0
   
    # Systemic Bucket에 따른 조정 (시스템 리스크 반영)
    systemic_bucket = signals["systemic_bucket"]
    if systemic_bucket in ["C2", "C3"]:
        sys_alpha = -0.05  # C2/C3 고위험 버킷에서 비중 줄임
    elif systemic_bucket == "C0":
        sys_alpha = 0.05  # C0 안전 버킷에서 비중 늘림
    else:
        sys_alpha = 0.0  # C1 중립 버킷에서 변화 없음
   
    # 각 자산군에 따른 가중치 계산 (ML_Risk / Systemic Bucket 반영 추가)
    if asset == "SPX":
        alpha = vix_alpha + dd_alpha + fxw_alpha + ml_alpha + sys_alpha
    elif asset == "NDX":
        alpha = (vix_alpha * 1.5) + (dd_alpha * 1.2) + (fxw_alpha * 1.3) + (ml_alpha * 1.4) + (sys_alpha * 1.2)  # NDX는 공격적, 리스크 조정 강하게
    elif asset == "DIV":
        alpha = (vix_alpha * 0.5) + (dd_alpha * 0.8) + (fxw_alpha * 0.7) + (ml_alpha * 0.6) + (sys_alpha * 0.5)  # DIV는 방어적, 리스크 조정 약하게
   
    # 최종 알파는 -0.1에서 0.1 사이로 제한 (과도한 tilt 방지)
    alpha = max(-0.1, min(alpha, 0.1))
   
    return alpha


def compute_portfolio_target(sig: Dict[str, float]) -> Dict[str, float]:
    """
    엔진 산출(SGOV/Satellite/Duration/Core)을 'Gold sleeve' 제외 나머지(=95%)에 적용.
    Gold는 월납입 규칙 기반 고정 5%로 반영.
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
    sat_weight = max(0.0, min(GOV_MAX_SAT,  sat_weight))
    dur_weight = max(0.0, min(GOV_MAX_DUR,  dur_weight))

    tri_sum = sgov_floor + sat_weight + dur_weight
    if tri_sum > remaining and tri_sum > 0:
        scale = remaining / tri_sum
        sgov_floor *= scale
        sat_weight *= scale
        dur_weight *= scale

    core_weight = max(0.0, remaining - (sgov_floor + sat_weight + dur_weight))

    core_config = {"SPX": 0.525, "NDX": 0.245, "DIV": 0.230}
    core_alloc = {k: core_weight * w for k, w in core_config.items()}

    em_w = sat_weight * (2.0 / 3.0)
    en_w = sat_weight * (1.0 / 3.0)

    weights = {
        "SPX": core_alloc["SPX"],
        "NDX": core_alloc["NDX"],
        "DIV": core_alloc["DIV"],
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


def load_cma_balance(path: Path = None) -> Dict[str, Any]:
    if path is None:
        path = DATA_DIR / "cma_balance.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 이 없습니다. 월초에 (a,b) 입력 파일을 커밋해야 합니다.")
    obj = json.loads(path.read_text(encoding="utf-8"))
    deployed = float(obj.get("deployed_krw", 0))
    cash = float(obj.get("cash_krw", 0))
    asof = str(obj.get("asof_yyyymm", ""))
    ref_base = float(obj.get("ref_base_krw", 0))
    return {
        "asof_yyyymm": asof,
        "deployed_krw": deployed,
        "cash_krw": cash,
        "ref_base_krw": ref_base,
    }


def _find_latest_cma_state_file() -> Path | None:
    files = sorted(DATA_DIR.glob("cma_state_20*.json"))
    return files[-1] if files else None


def compute_cma_overlay_section(
    sig: Dict[str, float],
    target_weights: Dict[str, float],
) -> Dict[str, Any]:
    bal = load_cma_balance()

    today = datetime.now().strftime("%Y%m%d")
    cma_state_path = DATA_DIR / f"cma_state_{today}.json"

    if cma_state_path.exists():
        prev_state = load_cma_state(cma_state_path)
    else:
        latest_prev = _find_latest_cma_state_file()
        prev_state = load_cma_state(latest_prev) if latest_prev else None

    dd_mag_3y = abs(float(sig.get("drawdown", 0.0)))
    dd_10y = float(sig.get("dd_10y", 0.0))
    state_name = determine_state_from_signals(sig)

    out = plan_cma_action(
        asof_yyyymm=bal["asof_yyyymm"],
        deployed_krw=bal["deployed_krw"],
        cash_krw=bal["cash_krw"],
        operator_ref_base_krw=float(bal.get("ref_base_krw", 0.0)),
        fxw=float(sig["fxw"]),
        vix=float(sig["vix"]),
        hy_oas=float(sig["hy_oas"]),
        dd_mag_3y=dd_mag_3y,
        long_term_dd_10y=dd_10y,
        ml_risk=float(sig["ml_risk"]),
        systemic_bucket=str(sig["systemic_bucket"]),
        final_state_name=state_name,
        prev_cma_state=prev_state,
    )

    save_cma_state(out["_state_obj"], cma_state_path)

    suggested_exec = float(out["execution"]["suggested_exec_krw"])
    alloc = allocate_risk_on(max(0.0, suggested_exec), target_weights)

    basket = ["SPX", "NDX", "DIV", "EM", "ENERGY"]
    denom = sum(max(0.0, float(target_weights.get(k, 0.0))) for k in basket)
    risk_on_w = {
        k: (max(0.0, float(target_weights.get(k, 0.0))) / denom if denom > 0 else 0.0)
        for k in basket
    }

    return {
        "state": state_name,
        "snapshot": out["cma_snapshot"],
        "tas": out["tas"],
        "execution": out["execution"],
        "allocation": alloc,
        "risk_on_target_weights": risk_on_w,
    }


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

    usdkrw = sig.get("fx_rate")
    vix = sig.get("vix")
    hy_oas = sig.get("hy_oas")
    ust2y = sig.get("dgs2")
    ust10y = sig.get("dgs10")

    fxw = sig.get("fxw")
    fx_vol = sig.get("fx_vol")
    drawdown = sig.get("drawdown")
    macro_score = sig.get("macro_score")
    ml_risk = sig.get("ml_risk")
    ml_opp = sig.get("ml_opp")
    ml_regime = sig.get("ml_regime")
    systemic_level = sig.get("systemic_level")
    systemic_bucket = sig.get("systemic_bucket")

    lines = []
    lines.append("# AURORA Rev12.1b Daily Report")
    lines.append("")
    lines.append(f"- Report Date: {today.isoformat()}")
    lines.append(f"- Engine Version: {meta.get('engine_version')}")
    lines.append(f"- Git Commit: {meta.get('git_commit')}")
    lines.append(f"- Run ID: {meta.get('run_id')}")
    lines.append(f"- Timestamp(UTC): {meta.get('timestamp_utc')}")
    lines.append("")

    lines.append("## 1. Market Data Summary (FD inputs)")
    if usdkrw is not None:
        lines.append(f"- USD/KRW (Sell Rate): {float(usdkrw):.2f}")
    if vix is not None:
        lines.append(f"- VIX: {float(vix):.2f}")
    if hy_oas is not None:
        lines.append(f"- HY OAS: {float(hy_oas):.2f} bps")
    if ust2y is not None and ust10y is not None:
        lines.append(f"- UST 2Y / 10Y: {float(ust2y):.2f}% / {float(ust10y):.2f}%")
    lines.append("")

    lines.append("## 2. Target Weights (Portfolio 100%)")
    lines.append("")
    lines.append("| Asset   | Weight (%) |")
    lines.append("|---------|-----------:|")
    total = 0.0
    for key in ["SPX", "NDX", "DIV", "EM", "ENERGY", "DURATION", "SGOV", "GOLD"]:
        w = float(final_weights.get(key, 0.0))
        total += w
        lines.append(f"| {key:7s} | {w*100:10.2f} |")
    lines.append(f"| **Total** | **{total*100:10.2f}** |")
    lines.append("")

    lines.append("## 3. FD / ML / Systemic Signals")
    lines.append("")

    fx_kde = sig.get("fx_kde", {})
    fx_rate = sig.get("fx_rate")

    if fx_kde:
        lines.append("## FXW Anchor Distribution (USD/KRW, 130D KDE)")
        lines.append("")
        lines.append(f"- KDE Anchor (Mode): **{fx_kde['anchor']:.1f}**")
        lines.append(
            f"- Distribution: "
            f"min={fx_kde['min']:.1f}, "
            f"P05={fx_kde['p05']:.1f}, "
            f"P25={fx_kde['p25']:.1f}, "
            f"P50={fx_kde['p50']:.1f}, "
            f"P75={fx_kde['p75']:.1f}, "
            f"P95={fx_kde['p95']:.1f}, "
            f"max={fx_kde['max']:.1f}"
        )
        if fx_rate is not None:
            pos = "below anchor (KRW strong)" if float(fx_rate) < float(fx_kde["anchor"]) else "above anchor (KRW weak)"
            lines.append(f"- Current FX: {float(fx_rate):.2f} → **{pos}**")
        lines.append("")

    if fxw is not None:
        lines.append(f"- FXW (KDE): {float(fxw):.3f}")
    if fx_vol is not None:
        lines.append(f"- FX Vol (21D σ): {float(fx_vol):.4f}")
    if drawdown is not None:
        lines.append(f"- SPX 3Y Drawdown: {float(drawdown)*100:.2f}%")
    if macro_score is not None:
        lines.append(f"- MacroScore: {float(macro_score):.3f}")
    if ml_risk is not None and ml_opp is not None and ml_regime is not None:
        lines.append(
            f"- ML_Risk / ML_Opp / ML_Regime: {float(ml_risk):.3f} / {float(ml_opp):.3f} / {float(ml_regime):.3f}"
        )
    if systemic_level is not None and systemic_bucket is not None:
        lines.append(f"- Systemic Level / Bucket: {float(systemic_level):.3f} / {systemic_bucket}")
    lines.append("")

    lines.append("## 4. Engine State")
    lines.append("")
    lines.append(f"- Final State: {state_name}")
    lines.append("")

    lines.append("## 5. CMA Overlay (TAS Dynamic Threshold) Snapshot")
    lines.append("")
    tas = cma_overlay.get("tas", {})
    exe = cma_overlay.get("execution", {})
    snap = cma_overlay.get("snapshot", {})
    alloc = cma_overlay.get("allocation", {})

    thr = tas.get("threshold")
    df = tas.get("deploy_factor")
    td = tas.get("target_deploy_krw")
    dr = tas.get("delta_raw_krw")

    fx_scale = exe.get("fx_scale")
    suggested = exe.get("suggested_exec_krw")

    lines.append(
        f"- CMA Snapshot (KRW): deployed={snap.get('deployed_krw',0):.0f}, cash={snap.get('cash_krw',0):.0f}, "
        f"total={snap.get('total_cma_krw',0):.0f}, ref_base={snap.get('ref_base_krw',0):.0f}, s0_count={snap.get('s0_count',0)}"
    )
    if thr is not None:
        lines.append(f"- Threshold: {float(thr)*100:.1f}%")
    if df is not None:
        lines.append(f"- Deploy Factor: {float(df)*100:.1f}%")
    if td is not None:
        lines.append(f"- Target Deploy (KRW): {float(td):.0f}")
    if dr is not None:
        lines.append(f"- Delta Raw (KRW): {float(dr):.0f}")
    if fx_scale is not None:
        lines.append(f"- FX Scale (BUY only): {float(fx_scale):.3f}")

    total_cma = float(snap.get("total_cma_krw", 0.0))
    suggested_v = float(suggested) if suggested is not None else 0.0
    pct = (abs(suggested_v) / total_cma * 100.0) if total_cma > 0 else 0.0
    direction = "BUY" if suggested_v > 0 else ("SELL" if suggested_v < 0 else "HOLD")
    lines.append(f"- Suggested Exec: {direction} {suggested_v:.0f} KRW ({pct:.2f}% of total CMA)")
    lines.append("")

    lines.append("| CMA Allocation (KRW, based on Suggested Exec) | Amount |")
    lines.append("|----------------------------------------------|-------:|")
    for k in ["SPX", "NDX", "DIV", "EM", "ENERGY"]:
        lines.append(f"| {k:20s} | {float(alloc.get(k,0.0)):.0f} |")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[REPORT] Daily report written to: {out_path}")


def main():
    market = load_latest_market()
    sig = build_signals(market)
    weights = compute_portfolio_target(sig)
    state_name = determine_state_from_signals(sig)

    cma_overlay = compute_cma_overlay_section(sig, weights)

    print("[INFO] ==== 3. FD / ML / Systemic Signals ====")
    print(f"FXW (KDE): {sig['fxw']:.3f}")
    print(f"FX Vol (21D σ): {sig['fx_vol']:.4f}")
    print(f"SPX 3Y Drawdown: {sig['drawdown']*100:.2f}%")
    print(f"MacroScore: {sig['macro_score']:.3f}")
    print(f"ML_Risk / ML_Opp / ML_Regime: {sig['ml_risk']:.3f} / {sig['ml_opp']:.3f} / {sig['ml_regime']:.3f}")
    print(f"Systemic Level / Bucket: {sig['systemic_level']:.3f} / {sig['systemic_bucket']}")
    print("[INFO] ==== 4. Engine State ====")
    print(f"Final State: {state_name}")

    print("[INFO] ==== 5. CMA Overlay Snapshot ====")
    print(f"Threshold: {cma_overlay['tas']['threshold']*100:.1f}%")
    print(f"Deploy Factor: {cma_overlay['tas']['deploy_factor']*100:.1f}%")

    today = datetime.now().strftime("%Y%m%d")
    out_path = DATA_DIR / f"aurora_target_weights_{today}.json"

    meta = {
        "engine_version": "AURORA-Rev12.1b",
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
