import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
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

# CMA Overlay (new)
# - cma_overlay.py 위치에 따라 import 경로를 맞춰야 함.
#   권장: scripts/cma_overlay.py 로 두고, scripts 폴더에 __init__.py 추가.
from scripts.cma_overlay import (
    load_cma_state,
    save_cma_state,
    plan_cma_action,
    allocate_risk_on,
)

DATA_DIR = Path("data")
REPORTS_DIR = ROOT / "reports"

# === Governance Protocol Rev12.1b Clamp Values ===
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

# 고정 Gold sleeve 비중 = 200k / 4,000k = 5%
GOLD_SLEEVE_WEIGHT = MONTHLY_GOLD_TOTAL_KRW / MONTHLY_TOTAL_KRW  # 0.05


def load_latest_market() -> Dict[str, Any]:
    files = sorted(DATA_DIR.glob("market_data_20*.json"))
    if not files:
        raise FileNotFoundError("data/ 폴더에 market_data_*.json 이 없습니다.")

    latest = files[-1]
    with latest.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("market_data_* JSON 최상위 구조는 dict 여야 합니다.")

    market: Dict[str, Any] = dict(raw)

    # FX
    usdkrw_val = raw.get("usdkrw") or raw.get("usdkrw_sell") or raw.get("fx_usdkrw")
    fx_val = float(usdkrw_val) if usdkrw_val is not None else 0.0

    hist_21d = (
        raw.get("usdkrw_history_21d")
        or raw.get("usdkrw_hist_21d")
        or raw.get("usdkrw_hist_21")
        or raw.get("fx_hist_21d")
    )
    hist_130d = (
        raw.get("usdkrw_history_130d")
        or raw.get("usdkrw_hist_130d")
        or raw.get("usdkrw_hist_130")
        or raw.get("fx_hist_130d")
    )

    market.setdefault("fx", {})
    fx_block = market["fx"]
    if not isinstance(fx_block, dict):
        fx_block = {}
        market["fx"] = fx_block

    fx_block.setdefault("usdkrw", fx_val)
    fx_block.setdefault("latest", fx_val)
    fx_block.setdefault("usdkrw_history_21d", hist_21d or [])
    fx_block.setdefault("usdkrw_history_130d", hist_130d or [])

    # SPX
    market.setdefault("spx", {})
    spx_block = market["spx"]
    if not isinstance(spx_block, dict):
        spx_block = {}
        market["spx"] = spx_block

    spx_block.setdefault("exposure", raw.get("spx_exposure"))
    spx_block.setdefault("drawdown_3y", raw.get("spx_drawdown_3y"))
    spx_block.setdefault("history_3y", raw.get("spx_history_3y", raw.get("spx_history", [])))

    # RISK
    market.setdefault("risk", {})
    risk_block = market["risk"]
    if not isinstance(risk_block, dict):
        risk_block = {}
        market["risk"] = risk_block

    vix = raw.get("vix")
    hy_oas = raw.get("hy_oas")
    yc_spread = raw.get("yc_spread") or raw.get("yc_10y_2y_spread")

    risk_block.setdefault("vix", float(vix) if vix is not None else 0.0)
    risk_block.setdefault("hy_oas", float(hy_oas) if hy_oas is not None else 0.0)
    risk_block.setdefault("yc_spread", float(yc_spread) if yc_spread is not None else 0.0)

    # RATES
    market.setdefault("rates", {})
    rates_block = market["rates"]
    if not isinstance(rates_block, dict):
        rates_block = {}
        market["rates"] = rates_block

    dgs2 = raw.get("dgs2") or raw.get("ust2y")
    dgs10 = raw.get("dgs10") or raw.get("ust10y")
    ffr_upper = raw.get("ffr_upper") or raw.get("ffr")

    rates_block.setdefault("dgs2", float(dgs2) if dgs2 is not None else 0.0)
    rates_block.setdefault("dgs10", float(dgs10) if dgs10 is not None else 0.0)
    rates_block.setdefault("ffr_upper", float(ffr_upper) if ffr_upper is not None else 0.0)

    # MACRO
    market.setdefault("macro", {})
    macro_block = market["macro"]
    if not isinstance(macro_block, dict):
        macro_block = {}
        market["macro"] = macro_block

    ism = raw.get("ism_mfg") or raw.get("ism")
    pmi = raw.get("pmi_sp_global") or raw.get("pmi") or ism
    cpi_yoy = raw.get("cpi_yoy")
    unemp = raw.get("unemployment") or raw.get("unemployment_rate")

    macro_block.setdefault("ism", float(ism) if ism is not None else 50.0)
    macro_block.setdefault("pmi", float(pmi) if pmi is not None else 50.0)
    macro_block.setdefault("pmi_markit", float(pmi) if pmi is not None else 50.0)
    macro_block.setdefault("cpi_yoy", float(cpi_yoy) if cpi_yoy is not None else 2.0)
    macro_block.setdefault("unemployment", float(unemp) if unemp is not None else 4.0)

    print(f"[INFO] Loaded market data JSON: {latest}")
    return market


def compute_fx_vol(fx_hist_21d):
    import math
    if not fx_hist_21d or len(fx_hist_21d) < 2:
        return 0.0
    rets = []
    for a, b in zip(fx_hist_21d[:-1], fx_hist_21d[1:]):
        if a <= 0 or b <= 0:
            continue
        rets.append(math.log(b / a))
    if not rets:
        return 0.0
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / max(1, len(rets) - 1)
    return var ** 0.5


def compute_drawdown(spx_block: Dict[str, Any]) -> float:
    hist = spx_block.get("history_3y") or spx_block.get("history_1095d") or spx_block.get("history")
    if not hist:
        return 0.0
    peak = max(hist)
    last = hist[-1]
    if peak <= 0:
        return 0.0
    return (last - peak) / peak  # negative in drawdown


def compute_spx_drawdown_10y() -> float:
    try:
        import yfinance as yf
    except ImportError:
        return 0.0

    data = yf.download("^GSPC", period="10y", progress=False)
    if "Close" not in data or data["Close"].empty:
        return 0.0

    prices = data["Close"]
    peak = float(prices.max())
    last = float(prices.iloc[-1])
    if peak <= 0:
        return 0.0
    return (last - peak) / peak  # negative


def compute_macro_score_from_market(pmi: float, cpi_yoy: float, unemployment: float) -> float:
    ism = pmi  # ISM -> PMI substitute
    ism_n = norm(ism, 45.0, 60.0)
    pmi_n = norm(pmi, 45.0, 60.0)
    cpi_n = 1.0 - norm(cpi_yoy, 3.0, 8.0)
    unemp_n = 1.0 - norm(unemployment, 3.0, 7.0)
    macro = 0.25 * (ism_n + pmi_n + cpi_n + unemp_n)
    return clip(macro, 0.0, 1.0)


def build_signals(market: Dict[str, Any]) -> Dict[str, float]:
    fx_block = market["fx"]
    spx_block = market["spx"]
    risk = market["risk"]
    rates = market["rates"]
    macro = market["macro"]

    fx_rate = fx_block["usdkrw"]
    fx_hist_21d = fx_block.get("usdkrw_history_21d", [])
    fx_vol = compute_fx_vol(fx_hist_21d)

    vix = risk["vix"]
    hy_oas = risk["hy_oas"]

    dgs2 = rates["dgs2"]
    dgs10 = rates["dgs10"]
    ffr_upper = rates["ffr_upper"]
    yc_spread_bps = (dgs10 - dgs2) * 100.0

    pmi = macro["pmi_markit"]
    cpi_yoy = macro["cpi_yoy"]
    unemployment = macro["unemployment"]

    drawdown = compute_drawdown(spx_block)
    dd_10y = compute_spx_drawdown_10y()

    # FXW (KDE)
    engine = AuroraX121()
    fx_hist_130d = fx_block.get("usdkrw_history_130d", [])
    for px in fx_hist_130d[:-1]:
        engine.fxw(px)
    fxw = engine.fxw(fx_rate)

    macro_score = compute_macro_score_from_market(pmi, cpi_yoy, unemployment)

    ml_risk = compute_ml_risk(
        vix=vix,
        hy_oas=hy_oas,
        fxw=fxw,
        drawdown=drawdown,
        yc_spread=yc_spread_bps,
    )
    ml_opp = compute_ml_opp(
        vix=vix,
        hy_oas=hy_oas,
        fxw=fxw,
        drawdown=drawdown,
    )
    ml_regime = compute_ml_regime(ml_risk=ml_risk, ml_opp=ml_opp)

    systemic_level = compute_systemic_level(
        hy_oas=hy_oas,
        yc_spread=yc_spread_bps,
        macro_score=macro_score,
        ml_regime=ml_regime,
        drawdown=drawdown,
    )
    systemic_bucket = determine_systemic_bucket(systemic_level)

    return {
        "fx_rate": fx_rate,
        "fxw": fxw,
        "fx_vol": fx_vol,
        "vix": vix,
        "hy_oas": hy_oas,
        "drawdown": drawdown,
        "dd_10y": dd_10y,
        "dgs2": dgs2,
        "dgs10": dgs10,
        "yc_spread_bps": yc_spread_bps,
        "ffr_upper": ffr_upper,
        "pmi": pmi,
        "cpi_yoy": cpi_yoy,
        "unemployment": unemployment,
        "macro_score": macro_score,
        "ml_risk": ml_risk,
        "ml_opp": ml_opp,
        "ml_regime": ml_regime,
        "systemic_level": systemic_level,
        "systemic_bucket": systemic_bucket,
    }


def determine_state_from_signals(sig: Dict[str, float]) -> str:
    systemic_bucket = sig["systemic_bucket"]
    systemic_level = sig["systemic_level"]
    vix = sig["vix"]
    ml_risk = sig["ml_risk"]
    dd = sig["drawdown"]  # negative

    if systemic_bucket in ("C2", "C3") or systemic_level >= 0.70:
        return "S3_HARD"

    if ml_risk >= 0.80 or vix >= 30.0 or dd <= -0.30:
        return "S2_HIGH_VOL"

    if ml_risk >= 0.60 or vix >= 22.0 or dd <= -0.10:
        return "S1_MILD"

    return "S0_NORMAL"


def compute_portfolio_target(sig: Dict[str, float]) -> Dict[str, float]:
    """
    엔진 산출(SGOV/Satellite/Duration/Core)을 'Gold sleeve' 제외 나머지(=95%)에 적용.
    Gold는 월납입 규칙(ISA10 + Direct10) 기반 고정 5%로 반영.
    """
    eng = AuroraX121()

    fxw = sig["fxw"]
    fx_rate = sig["fx_rate"]
    ffr_upper = sig["ffr_upper"]
    ml_risk = sig["ml_risk"]
    ml_opp = sig["ml_opp"]
    macro_score = sig["macro_score"]
    systemic_bucket = sig["systemic_bucket"]

    # Gold fixed sleeve
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

    # Governance caps
    sgov_floor = max(0.0, min(GOV_MAX_SGOV, sgov_floor))
    sat_weight = max(0.0, min(GOV_MAX_SAT,  sat_weight))
    dur_weight = max(0.0, min(GOV_MAX_DUR,  dur_weight))

    # Fit within remaining budget (95%)
    tri_sum = sgov_floor + sat_weight + dur_weight
    if tri_sum > remaining and tri_sum > 0:
        scale = remaining / tri_sum
        sgov_floor *= scale
        sat_weight *= scale
        dur_weight *= scale

    core_weight = max(0.0, remaining - (sgov_floor + sat_weight + dur_weight))

    # Core split (RuleSet)
    core_config = {"SPX": 0.525, "NDX": 0.245, "DIV": 0.230}
    core_alloc = {k: core_weight * w for k, w in core_config.items()}

    # Satellite split (EM:ENERGY = 2:1)
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

    # Normalize tiny float drift
    total = sum(weights.values())
    if total > 0:
        scale = 1.0 / total
        weights = {k: v * scale for k, v in weights.items()}

    return weights


def load_cma_balance(path: Path = Path("insert") / "cma_balance.json") -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("data/cma_balance.json 이 없습니다. 월초에 (a,b) 입력 파일을 커밋해야 합니다.")
    obj = json.loads(path.read_text(encoding="utf-8"))
    deployed = float(obj.get("deployed_krw", 0))
    cash = float(obj.get("cash_krw", 0))
    asof = str(obj.get("asof_yyyymm", ""))
    return {"asof_yyyymm": asof, "deployed_krw": deployed, "cash_krw": cash}


def compute_cma_overlay_section(
    sig: Dict[str, float],
    target_weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    CMA Overlay 계산:
    - 입력 (a,b)은 data/cma_balance.json
    - state 저장/갱신은 data/cma_state.json
    - 출력은 숫자 기반 (threshold, deploy_factor, target_deploy, delta_raw, fx_scale, suggested_exec, allocation)
    """
    bal = load_cma_balance()
    prev_state = load_cma_state()

    # dd_mag is magnitude (positive)
    dd_mag_3y = abs(float(sig.get("drawdown", 0.0)))
    dd_10y = float(sig.get("dd_10y", 0.0))

    state_name = determine_state_from_signals(sig)

    out = plan_cma_action(
        asof_yyyymm=bal["asof_yyyymm"],
        deployed_krw=bal["deployed_krw"],
        cash_krw=bal["cash_krw"],
        fxw=float(sig["fxw"]),
        vix=float(sig["vix"]),
        hy_oas=float(sig["hy_oas"]),
        dd_mag_3y=dd_mag_3y,
        long_term_dd_10y=dd_10y,  # negative
        ml_risk=float(sig["ml_risk"]),
        systemic_bucket=str(sig["systemic_bucket"]),
        final_state_name=state_name,
        prev_cma_state=prev_state,
    )

    # persist state
    save_cma_state(out["_state_obj"])

    suggested_exec = float(out["execution"]["suggested_exec_krw"])
    alloc = allocate_risk_on(max(0.0, suggested_exec), target_weights)

    return {
        "state": state_name,
        "snapshot": out["cma_snapshot"],
        "tas": out["tas"],
        "execution": out["execution"],
        "allocation": alloc,
    }


def write_daily_report(
    sig: Dict[str, Any],
    final_weights: Dict[str, float],
    state_name: str,
    cma_overlay: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    """
    Daily Report markdown:
    경로 고정: ROOT/reports/aurora_daily_report_YYYYMMDD.md
    섹션: 1 Market, 2 Weights, 3 Signals, 4 State, 5 CMA Overlay
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().date()
    yyyymmdd = today.strftime("%Y%m%d")
    out_path = REPORTS_DIR / f"aurora_daily_report_{yyyymmdd}.md"

    # Market summary
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

    # CMA Overlay (daily)
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

    lines.append(f"- CMA Snapshot (KRW): deployed={snap.get('deployed_krw',0):.0f}, cash={snap.get('cash_krw',0):.0f}, total={snap.get('total_cma_krw',0):.0f}, ref_base={snap.get('ref_base_krw',0):.0f}, s0_count={snap.get('s0_count',0)}")
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
    if suggested is not None:
        lines.append(f"- Suggested Exec (KRW): {float(suggested):.0f}")
    lines.append("")

    # Allocation table (risk-on only)
    lines.append("| CMA Allocation (KRW) | Amount |")
    lines.append("|----------------------|-------:|")
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

    # CMA overlay (daily, based on cma_balance.json)
    cma_overlay = compute_cma_overlay_section(sig, weights)

    # Console summary (optional)
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

    # Output JSON (with meta)
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

    # Report
    write_daily_report(sig, weights, state_name, cma_overlay, meta)


if __name__ == "__main__":
    main()
