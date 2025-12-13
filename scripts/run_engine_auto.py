# scripts/run_engine_auto.py
# Rev12.3-final baseline + (CMA overlay external) / FXW is computed from 130D KDE, never read from market JSON.
# Note: CMA는 외부 자금이며, 내부 target weights를 변경하지 않는다. (Overlay는 별도 섹션/리포트로만 제공)

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
    Deterministic rule:
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


def load_latest_market() -> Dict[str, Any]:
    """
    Canonical input: data/market_data_YYYYMMDD.json
    - 절대 외부 다운로드로 보충하지 않음 (Determinism)
    """
    files = sorted(DATA_DIR.glob("market_data_20*.json"))
    if not files:
        raise FileNotFoundError("data/ 폴더에 market_data_*.json 이 없습니다.")

    latest = files[-1]
    with latest.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("market_data_* JSON 최상위 구조는 dict 여야 합니다.")

    # ✅ Data freshness (DATE ONLY)
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

    fx_hist_21d = _clean_float_series(hist_21d, "fx.usdkrw_history_21d")
    fx_hist_130d = _clean_float_series(hist_130d, "fx.usdkrw_history_130d")

    if len(fx_hist_21d) < 2:
        _fail("MarketData missing/insufficient: fx.usdkrw_history_21d (need>=2)")
    if len(fx_hist_130d) < 130:
        _fail(f"MarketData missing/insufficient: fx.usdkrw_history_130d (need>=130, got={len(fx_hist_130d)})")

    # P0-2: 반드시 최근 130 trading days만 사용
    fx_hist_130d = fx_hist_130d[-130:]

    market["fx"] = {
        "usdkrw": float(usdkrw),
        "latest": float(usdkrw),
        "usdkrw_history_21d": fx_hist_21d,
        "usdkrw_history_130d": fx_hist_130d,
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
    hy_oas = (
        risk.get("hy_oas_bps")
        or risk.get("hy_oas")
        or raw.get("hy_oas_bps")
        or raw.get("hy_oas")
    )

    if vix is None:
        _fail("MarketData missing: risk.vix")
    if hy_oas is None:
        _fail("MarketData missing: risk.hy_oas (bps)")

    vix_f = float(vix)
    hy_oas_f = float(hy_oas)

    _assert_range("VIX(level)", vix_f, 5.0, 80.0)

    # HY OAS deterministic unit normalization:
    # - If 0 < value < 50, treat as percent points and convert to bps (×100)
    if 0.0 < hy_oas_f < 50.0:
        old = hy_oas_f
        hy_oas_f *= 100.0
        print(f"[UNIT] HY_OAS normalized: percent->bps ({old} -> {hy_oas_f})")

    _assert_range("HY_OAS(bps)", hy_oas_f, 50.0, 2000.0)

    market["risk"] = {
        "vix": vix_f,
        "hy_oas": hy_oas_f,
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

    dgs2_f = float(dgs2)
    dgs10_f = float(dgs10)
    ffr_f = float(ffr_upper)

    dgs2_f = _normalize_pct_from_fraction(dgs2_f, "UST2Y")
    dgs10_f = _normalize_pct_from_fraction(dgs10_f, "UST10Y")
    ffr_f = _normalize_policy_rate_pct(ffr_f, "FFR_Upper")

    _assert_range("UST2Y(%)", dgs2_f, 0.0, 25.0)
    _assert_range("UST10Y(%)", dgs10_f, 0.0, 25.0)
    _assert_range("FFR_Upper(%)", ffr_f, 0.0, 25.0)

    market["rates"] = {
        "dgs2": dgs2_f,
        "dgs10": dgs10_f,
        "ffr_upper": ffr_f,
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

    pmi_f = float(pmi)
    cpi_f = _normalize_macro_pct(float(cpi_yoy), "CPI_YoY")
    unemp_f = _normalize_macro_pct(float(unemp), "Unemployment")

    _assert_range("PMI(points)", pmi_f, 0.0, 100.0)
    _assert_range("CPI_YoY(%)", cpi_f, -20.0, 50.0)
    _assert_range("Unemployment(%)", unemp_f, 0.0, 30.0)

    market["macro"] = {
        "pmi_markit": pmi_f,
        "cpi_yoy": cpi_f,
        "unemployment": unemp_f,
    }

    # --- ETF block (optional) ---
    etf = raw.get("etf") or raw.get("etf_close") or {}
    if not isinstance(etf, dict):
        etf = {}
    market["etf"] = etf

    print(f"[INFO] Loaded market data JSON: {latest}")
    return market


def compute_fx_vol(fx_hist_21d: list) -> float:
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
    # 운영 합의: ISM 제외, PMI를 대체 사용
    ism = pmi
    ism_n = norm(ism, 45.0, 60.0)
    pmi_n = norm(pmi, 45.0, 60.0)
    cpi_n = 1.0 - norm(cpi_yoy, 2.0, 8.0)
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
    anchor = float(x[int(density.argmax())])

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


def build_signals(market: Dict[str, Any]) -> Dict[str, Any]:
    """
    핵심: FXW는 market['fxw']로 읽지 않는다.
    FXW = KDE(anchor from 130D USDKRW) -> engine.kde.fxw(fx_rate)
    """
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

    # Yield Curve: (10Y - 2Y) in bps
    yc_spread_bps = (dgs10 - dgs2) * 100.0
    _assert_range("YieldCurveSpread(bps)", yc_spread_bps, -300.0, 300.0)

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
    # FXW (KDE 130 trading days) — strict & validated
    # =========================
    fx_hist_130d = fx_block.get("usdkrw_history_130d", [])
    if not isinstance(fx_hist_130d, list) or len(fx_hist_130d) < 130:
        _fail("MarketData missing/insufficient: fx.usdkrw_history_130d (need>=130)")

    fx_hist_130d = _clean_float_series(fx_hist_130d, "fx.usdkrw_history_130d")[-130:]
    if len(fx_hist_130d) < 130:
        _fail(f"MarketData invalid: fx.usdkrw_history_130d cleaned length < 130 (got={len(fx_hist_130d)})")

    engine = AuroraX121()
    for px in fx_hist_130d:
        engine.kde.add(float(px))
    fxw = engine.kde.fxw(fx_rate)

    fx_kde = compute_fx_kde_anchor_and_stats(fx_hist_130d)

    # P0-2 guard: anchor sanity
    anchor = float(fx_kde["anchor"])
    p05 = float(fx_kde["p05"])
    if anchor < (p05 - 10.0):
        _fail(
            f"P0-2_GUARD_FAIL: KDE anchor too low vs distribution. "
            f"anchor={anchor:.1f}, p05={p05:.1f}. "
            f"Check USDKRW 130D series window/trading-days integrity."
        )

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
        "fx_kde": fx_kde,
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

    if systemic_bucket in ("C2", "C3") or systemic_level >= 0.70:
        return "S3_HARD"

    if fx_vol >= 0.02:
        return "S2_HIGH_VOL"

    if ml_risk >= 0.80 or vix >= 30.0 or dd <= -0.30:
        return "S2_HIGH_VOL"

    if ml_risk >= 0.60 or vix >= 22.0 or dd <= -0.10:
        return "S1_MILD"

    return "S0_NORMAL"


def compute_portfolio_target(sig: Dict[str, float]) -> Dict[str, float]:
    """
    내부 포트 타겟(ISA+Direct) = 100% 기준.
    Gold sleeve(5%) 고정, 나머지 95%에 엔진(SGOV/Satellite/Duration/Core) 적용.
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

    # Core baseline (Rev12.1b/12.2/12.3 consistent)
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

    # enforce sum=1.0
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
    """
    CMA = 외부 자금.
    - 내부 target_weights를 "변경"하지 않는다.
    - suggested_exec_krw와, 그 실행금을 risk-on basket에 어떻게 배분할지(alloc)만 제공한다.
    """
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

    lines = []
    lines.append("# AURORA Daily Report")
    lines.append("")
    lines.append(f"- Report Date: {today.isoformat()}")
    lines.append(f"- Engine Version: {meta.get('engine_version')}")
    lines.append(f"- Git Commit: {meta.get('git_commit')}")
    lines.append(f"- Run ID: {meta.get('run_id')}")
    lines.append(f"- Timestamp(UTC): {meta.get('timestamp_utc')}")
    lines.append("")

    lines.append("## 1. Market Data Summary (FD inputs)")
    lines.append(f"- USD/KRW (Sell Rate): {float(sig['fx_rate']):.2f}")
    lines.append(f"- VIX: {float(sig['vix']):.2f}")
    lines.append(f"- HY OAS: {float(sig['hy_oas']):.2f} bps")
    lines.append(f"- UST 2Y / 10Y: {float(sig['dgs2']):.2f}% / {float(sig['dgs10']):.2f}%")
    lines.append(f"- Yield Curve Spread (10Y-2Y bps): {float(sig['yc_spread_bps']):.1f}")
    lines.append("")

    lines.append("## 2. Target Weights (Internal Portfolio 100%)")
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

    lines.append("## 3. Signals (FD / ML / Systemic)")
    lines.append(f"- FXW (KDE): {float(sig['fxw']):.3f}")
    lines.append(f"- FX Vol (21D σ): {float(sig['fx_vol']):.4f}")
    lines.append(f"- SPX 3Y Drawdown: {float(sig['drawdown'])*100:.2f}%")
    lines.append(f"- MacroScore: {float(sig['macro_score']):.3f}")
    lines.append(
        f"- ML_Risk / ML_Opp / ML_Regime: "
        f"{float(sig['ml_risk']):.3f} / {float(sig['ml_opp']):.3f} / {float(sig['ml_regime']):.3f}"
    )
    lines.append(f"- Systemic Level / Bucket: {float(sig['systemic_level']):.3f} / {sig['systemic_bucket']}")
    lines.append("")

    fx_kde = sig.get("fx_kde", {})
    if fx_kde:
        lines.append("### FXW Anchor Distribution (USD/KRW, 130D KDE)")
        lines.append(
            f"- KDE Anchor (Mode): {fx_kde['anchor']:.1f} | "
            f"min={fx_kde['min']:.1f}, p05={fx_kde['p05']:.1f}, p50={fx_kde['p50']:.1f}, p95={fx_kde['p95']:.1f}, max={fx_kde['max']:.1f}"
        )
        pos = "below anchor (KRW strong)" if float(sig["fx_rate"]) < float(fx_kde["anchor"]) else "above anchor (KRW weak)"
        lines.append(f"- Current FX: {float(sig['fx_rate']):.2f} → {pos}")
        lines.append("")

    lines.append("## 4. Engine State")
    lines.append(f"- Final State: {state_name}")
    lines.append("")

    lines.append("## 5. CMA Overlay (External) Snapshot")
    tas = cma_overlay.get("tas", {})
    exe = cma_overlay.get("execution", {})
    snap = cma_overlay.get("snapshot", {})
    alloc = cma_overlay.get("allocation", {})

    suggested = float(exe.get("suggested_exec_krw", 0.0))
    total_cma = float(snap.get("total_cma_krw", 0.0))
    pct = (abs(suggested) / total_cma * 100.0) if total_cma > 0 else 0.0
    direction = "BUY" if suggested > 0 else ("SELL" if suggested < 0 else "HOLD")

    lines.append(
        f"- CMA Snapshot (KRW): deployed={snap.get('deployed_krw',0):.0f}, cash={snap.get('cash_krw',0):.0f}, "
        f"total={snap.get('total_cma_krw',0):.0f}, ref_base={snap.get('ref_base_krw',0):.0f}, s0_count={snap.get('s0_count',0)}"
    )
    if tas.get("threshold") is not None:
        lines.append(f"- Threshold: {float(tas['threshold'])*100:.1f}%")
    if tas.get("deploy_factor") is not None:
        lines.append(f"- Deploy Factor: {float(tas['deploy_factor'])*100:.1f}%")
    if tas.get("target_deploy_krw") is not None:
        lines.append(f"- Target Deploy (KRW): {float(tas['target_deploy_krw']):.0f}")
    lines.append(f"- Suggested Exec: {direction} {suggested:.0f} KRW ({pct:.2f}% of total CMA)")
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
    print(f"Yield Curve Spread (10Y-2Y bps): {sig['yc_spread_bps']:.1f}")

    print("[INFO] ==== 4. Engine State ====")
    print(f"Final State: {state_name}")

    print("[INFO] ==== 5. CMA Overlay Snapshot ====")
    print(f"Threshold: {cma_overlay['tas']['threshold']*100:.1f}%")
    print(f"Deploy Factor: {cma_overlay['tas']['deploy_factor']*100:.1f}%")

    today = datetime.now().strftime("%Y%m%d")
    out_path = DATA_DIR / f"aurora_target_weights_{today}.json"

    meta = {
        # 12.4는 폐기했다고 했으니, core 기준은 12.3-final로 두고 overlay만 보고서에 표시
        "engine_version": "AURORA-Rev12.3-final (Core) + CMA Overlay",
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
