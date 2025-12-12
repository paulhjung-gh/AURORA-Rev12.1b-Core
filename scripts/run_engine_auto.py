import os
import sys
import json
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


def load_latest_market() -> Dict[str, Any]:
    """
    Canonical input: data/market_data_YYYYMMDD.json
    - 절대 외부 다운로드로 보충하지 않음 (Determinism)
    - key는 build_market_json.py가 만든 스키마를 1순위로 사용
    - 구버전 key는 최소한으로만 alias 처리
    """
    files = sorted(DATA_DIR.glob("market_data_20*.json"))
    if not files:
        raise FileNotFoundError("data/ 폴더에 market_data_*.json 이 없습니다.")

    latest = files[-1]
    with latest.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("market_data_* JSON 최상위 구조는 dict 여야 합니다.")

    # ✅ Data freshness (DATE ONLY, ignore time)
    # market_data_YYYYMMDD.json 내부에 meta.generated_yyyymmdd 가 있으면 오늘 날짜와 일치해야 함
    today = datetime.now().strftime("%Y%m%d")
    meta = raw.get("meta", {})
    if isinstance(meta, dict):
        generated = meta.get("generated_yyyymmdd") or meta.get("date") or meta.get("generated_date")
        if generated is not None and str(generated) != today:
            _fail(f"STALE market data: meta.generated_yyyymmdd={generated}, today={today}, file={latest.name}")

    market: Dict[str, Any] = dict(raw)

    # --- FX block ---
    fx = raw.get("fx", {})
    if not isinstance(fx, dict):
        fx = {}
    usdkrw = fx.get("usdkrw") or raw.get("usdkrw") or raw.get("usdkrw_sell") or raw.get("fx_usdkrw")
    if usdkrw is None:
        _fail("MarketData missing: fx.usdkrw")

    hist_21d = fx.get("usdkrw_history_21d") or raw.get("usdkrw_history_21d") or raw.get("fx_hist_21d")
    hist_130d = fx.get("usdkrw_history_130d") or raw.get("usdkrw_history_130d") or raw.get("fx_hist_130d")

    if not isinstance(hist_21d, list) or len(hist_21d) < 2:
        _fail("MarketData missing/insufficient: fx.usdkrw_history_21d (need>=2)")
    if not isinstance(hist_130d, list) or len(hist_130d) < 130:
        _fail("MarketData missing/insufficient: fx.usdkrw_history_130d (need>=130)")

    market["fx"] = {
        "usdkrw": float(usdkrw),
        "latest": float(usdkrw),
        "usdkrw_history_21d": [float(x) for x in hist_21d],
        "usdkrw_history_130d": [float(x) for x in hist_130d],
        "fx_vol_21d_sigma": fx.get("fx_vol_21d_sigma", raw.get("fx_vol_21d_sigma")),
    }

    # --- SPX block ---
    spx = raw.get("spx", {})
    if not isinstance(spx, dict):
        spx = {}

    spx_last = spx.get("last") or raw.get("spx_last") or raw.get("spx")
    closes_3y = spx.get("closes_3y_1095") or spx.get("history_3y") or raw.get("spx_3y_closes_1095")
    if spx_last is None:
        _fail("MarketData missing: spx.last")
    if not isinstance(closes_3y, list) or len(closes_3y) < 200:
        _fail("MarketData missing/insufficient: spx.closes_3y_1095 (need meaningful series)")

    market["spx"] = {
        "last": float(spx_last),
        "closes_3y_1095": [float(x) for x in closes_3y],
        "closes_10y": spx.get("closes_10y") or raw.get("spx_10y_closes"),
    }

    # --- RISK block ---
    risk = raw.get("risk", {})
    if not isinstance(risk, dict):
        risk = {}

    vix = risk.get("vix") or raw.get("vix")
    hy_oas = risk.get("hy_oas") or raw.get("hy_oas_bps") or raw.get("hy_oas")
    if vix is None:
        _fail("MarketData missing: risk.vix")
    if hy_oas is None:
        _fail("MarketData missing: risk.hy_oas (bps)")

    market["risk"] = {
        "vix": float(vix),
        "hy_oas": float(hy_oas),
        "yc_spread": float(risk.get("yc_spread", raw.get("yc_spread", 0.0))),
    }

    # --- RATES block ---
    rates = raw.get("rates", {})
    if not isinstance(rates, dict):
        rates = {}

    dgs2 = rates.get("dgs2") or raw.get("dgs2") or raw.get("ust2y_pct") or raw.get("ust2y")
    dgs10 = rates.get("dgs10") or raw.get("dgs10") or raw.get("ust10y_pct") or raw.get("ust10y")
    ffr_upper = rates.get("ffr_upper") or raw.get("ffr_upper_pct") or raw.get("ffr_upper") or raw.get("ffr")
    if dgs2 is None or dgs10 is None or ffr_upper is None:
        _fail("MarketData missing: rates.(dgs2,dgs10,ffr_upper)")

    market["rates"] = {
        "dgs2": float(dgs2),
        "dgs10": float(dgs10),
        "ffr_upper": float(ffr_upper),
    }

    # --- MACRO block ---
    macro = raw.get("macro", {})
    if not isinstance(macro, dict):
        macro = {}

    pmi = macro.get("pmi_markit") or macro.get("pmi") or raw.get("pmi_markit") or raw.get("pmi_sp_global")
    cpi_yoy = macro.get("cpi_yoy") or raw.get("cpi_yoy_pct") or raw.get("cpi_yoy")
    unemp = macro.get("unemployment") or raw.get("unemp_rate_pct") or raw.get("unemployment_rate")

    if pmi is None or cpi_yoy is None or unemp is None:
        _fail("MarketData missing: macro.(pmi_markit,cpi_yoy,unemployment)")

    market["macro"] = {
        "pmi_markit": float(pmi),
        "cpi_yoy": float(cpi_yoy),
        "unemployment": float(unemp),
    }

    # --- ETF block (optional) ---
    etf = raw.get("etf") or raw.get("etf_close") or {}
    if not isinstance(etf, dict):
        etf = {}
    market["etf"] = etf

    print(f"[INFO] Loaded market data JSON: {latest}")
    return market


def compute_fx_vol(fx_hist_21d: list) -> float:
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


def compute_drawdown_from_series(closes: list[float]) -> float:
    if not closes:
        return 0.0
    peak = max(closes)
    last = closes[-1]
    if peak <= 0:
        return 0.0
    return (last - peak) / peak  # negative in drawdown


def compute_macro_score_from_market(pmi: float, cpi_yoy: float, unemployment: float) -> float:
    ism = pmi  # ISM -> PMI substitute (운영 선택)
    ism_n = norm(ism, 45.0, 60.0)
    pmi_n = norm(pmi, 45.0, 60.0)
    cpi_n = 1.0 - norm(cpi_yoy, 2.0, 8.0)       # ✅ 2~8 (official)
    unemp_n = 1.0 - norm(unemployment, 3.0, 7.0)
    macro = 0.25 * (ism_n + pmi_n + cpi_n + unemp_n)
    return clip(macro, 0.0, 1.0)

def compute_fx_kde_anchor_and_stats(fx_hist_130d: list[float]) -> Dict[str, float]:
    import numpy as np
    from scipy.stats import gaussian_kde

    data = np.asarray(fx_hist_130d, dtype=float)

    kde = gaussian_kde(data)
    x = np.linspace(data.min() - 100.0, data.max() + 100.0, 1000)
    density = kde(x)

    anchor = float(x[np.argmax(density)])

    return {
        "anchor": anchor,
        "p05": float(np.percentile(data, 5)),
        "p25": float(np.percentile(data, 25)),
        "p50": float(np.percentile(data, 50)),
        "p75": float(np.percentile(data, 75)),
        "p95": float(np.percentile(data, 95)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def build_signals(market: Dict[str, Any]) -> Dict[str, float]:
    fx_block = market["fx"]
    spx_block = market["spx"]
    risk = market["risk"]
    rates = market["rates"]
    macro = market["macro"]

    fx_rate = float(fx_block["usdkrw"])
    fx_hist_21d = fx_block.get("usdkrw_history_21d", [])
    fx_vol = compute_fx_vol(fx_hist_21d)

    vix = float(risk["vix"])
    hy_oas = float(risk["hy_oas"])

    dgs2 = float(rates["dgs2"])
    dgs10 = float(rates["dgs10"])
    ffr_upper = float(rates["ffr_upper"])
    yc_spread_bps = (dgs10 - dgs2) * 100.0

    pmi = float(macro["pmi_markit"])
    cpi_yoy = float(macro["cpi_yoy"])
    unemployment = float(macro["unemployment"])

    closes_3y = spx_block.get("closes_3y_1095", [])
    drawdown = compute_drawdown_from_series(closes_3y)

    closes_10y = spx_block.get("closes_10y")
    if isinstance(closes_10y, list) and len(closes_10y) >= 200:
        dd_10y = compute_drawdown_from_series([float(x) for x in closes_10y])
    else:
        dd_10y = float(drawdown)

    # =========================
    # FXW (KDE 130D) — strict preload, no neutral fallback
    # =========================
    fx_hist_130d = fx_block.get("usdkrw_history_130d", [])
    if not isinstance(fx_hist_130d, list) or len(fx_hist_130d) < 130:
        _fail("MarketData missing/insufficient: fx.usdkrw_history_130d (need>=130)")

    engine = AuroraX121()

    # ✅ preload: add only (NO compute during preload)
    for px in fx_hist_130d:
        engine.kde.add(float(px))

    # ✅ compute exactly once
    fxw = engine.kde.fxw(fx_rate)

    # =========================
    # MacroScore (주의: 아래 함수도 공식 범위로 수정 권장)
    # =========================
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
    fx_vol = sig.get("fx_vol", 0.0)
    ml_risk = sig["ml_risk"]
    dd = sig["drawdown"]  # negative

    # 1) Systemic hard clamp
    if systemic_bucket in ("C2", "C3") or systemic_level >= 0.70:
        return "S3_HARD"

    # 2) High vol trigger (Governance): FXVol ≥ 0.02 → S2_HIGH_VOL
    if fx_vol >= 0.02:
        return "S2_HIGH_VOL"

    # 3) Other high-vol / crash conditions
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
    # 운영 기준: data/cma_balance.json (월초에 업데이트 후 커밋)
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

    # ✅ prev_state: 오늘 파일이 없으면 "가장 최근 파일"에서 이어받기
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

    # ✅ Suggested Exec: 단일 라인(중복 제거) + 안전 처리
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
