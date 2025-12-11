import sys
import json
from pathlib import Path
from typing import Dict, Any
from cma_tas import CmaTasInput, compute_cma_tas
from datetime import datetime  # datetime 사용 위해 추가

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


DATA_DIR = Path("data")


def load_latest_market() -> Dict[str, Any]:
    """data/ 폴더에서 가장 최근 market_data_*.json 파일을 불러와
    엔진이 기대하는 market 구조로 변환한다.
    """
    files = sorted(DATA_DIR.glob("market_data_20*.json"))
    if not files:
        raise FileNotFoundError("data/ 폴더에 market_data_*.json 이 없습니다.")
    
    latest = files[-1]
    with latest.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("market_data_* JSON 최상위 구조는 dict 여야 합니다.")

    # 1) 기본적으로 raw 내용을 그대로 옮겨 놓고
    market: Dict[str, Any] = dict(raw)

    # 2)  FX block
    usdkrw_val = (
        raw.get("usdkrw")
        or raw.get("usdkrw_sell")
        or raw.get("fx_usdkrw")
    )
    fx_val = float(usdkrw_val) if usdkrw_val is not None else 0.0

    # 21D / 130D 히스토리 키 이름 여러 패턴 지원
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


    # 3) SPX 블록 (노멀라이즈용 / drawdown 계산용)
    market.setdefault("spx", {})
    spx_block = market["spx"]
    if not isinstance(spx_block, dict):
        spx_block = {}
        market["spx"] = spx_block

    spx_block.setdefault("exposure", raw.get("spx_exposure"))
    spx_block.setdefault("drawdown_3y", raw.get("spx_drawdown_3y"))
    # 3년 히스토리(옵션)
    spx_block.setdefault(
        "history_3y",
        raw.get("spx_history_3y", raw.get("spx_history", [])),
    )

    # 4) RISK 블록: VIX, HY OAS, YC 스프레드 등
    market.setdefault("risk", {})
    risk_block = market["risk"]
    if not isinstance(risk_block, dict):
        risk_block = {}
        market["risk"] = risk_block

    vix = raw.get("vix")
    hy_oas = raw.get("hy_oas")
    yc_spread = (
        raw.get("yc_spread")
        or raw.get("yc_10y_2y_spread")
    )

    risk_block.setdefault("vix", float(vix) if vix is not None else 0.0)
    risk_block.setdefault("hy_oas", float(hy_oas) if hy_oas is not None else 0.0)
    risk_block.setdefault("yc_spread", float(yc_spread) if yc_spread is not None else 0.0)

    # 5) RATES 블록: 2Y, 10Y, FFR Upper
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

    # 6) MACRO 블록: ISM, PMI, CPI YoY, Unemployment
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
    """21D FX log-return sigma (비연율)."""
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
    """
    3Y SPX long-horizon drawdown 계산.
    history_3y 혹은 history_1095d 리스트가 있으면 사용.
    """
    hist = spx_block.get("history_3y") or spx_block.get("history_1095d") or spx_block.get("history")
    if not hist:
        return 0.0
    peak = max(hist)
    last = hist[-1]
    if peak <= 0:
        return 0.0
    return (last - peak) / peak  # 예: -0.25 = -25%


def compute_macro_score_from_market(pmi: float, cpi_yoy: float, unemployment: float) -> float:
    """
    RuleSet 기반 MacroScore 구현.
    ISM 지표는 S&P Global US Manufacturing PMI 로 대체.

    ism_n = norm(ISM,45,60)
    pmi_n = norm(PMI,45,60)
    cpi_n = 1-norm(CPI_YoY,3,8)
    unemp_n = 1-norm(Unemployment,3,7)
    macro = clip(0.25*(ism_n+pmi_n+cpi_n+unemp_n),0,1)
    """
    ism = pmi  # ISM → PMI 대체
    ism_n = norm(ism, 45.0, 60.0)
    pmi_n = norm(pmi, 45.0, 60.0)
    cpi_n = 1.0 - norm(cpi_yoy, 3.0, 8.0)
    unemp_n = 1.0 - norm(unemployment, 3.0, 7.0)
    macro = 0.25 * (ism_n + pmi_n + cpi_n + unemp_n)
    return clip(macro, 0.0, 1.0)


def build_signals(market: Dict[str, Any]) -> Dict[str, float]:
    """market_data JSON 을 FD/ML/Systemic 엔진 입력으로 변환."""
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

    # FXW 계산 (KDE 엔진 사용)
    engine = AuroraX121()
    fx_hist_130d = fx_block.get("usdkrw_history_130d", [])
    for px in fx_hist_130d[:-1]:
        engine.fxw(px)
    fxw = engine.fxw(fx_rate)

    macro_score = compute_macro_score_from_market(pmi, cpi_yoy, unemployment)

    # ML 레이어
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

    # Systemic 레이어: Rev12.1b 공식 로직
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


def compute_portfolio_target(sig: Dict[str, float]) -> Dict[str, float]:
    """
    AuroraX121 엔진을 이용해 SGOV / Satellite / Duration / Core 비중을 산출하고
    Core/Satellite 를 세부 자산(SPX,NDX,DIV,EM,ENERGY)로 분배한다.
    """
    eng = AuroraX121()

    fxw = sig["fxw"]
    fx_rate = sig["fx_rate"]
    ffr_upper = sig["ffr_upper"]
    ml_risk = sig["ml_risk"]
    ml_opp = sig["ml_opp"]
    macro_score = sig["macro_score"]
    systemic_bucket = sig["systemic_bucket"]

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

    # 안전장치: 음수/1 초과 방지 (필요시 튜닝 가능)
    sgov_floor = max(0.0, min(0.80, sgov_floor))
    sat_weight = max(0.0, min(0.20, sat_weight))
    dur_weight = max(0.0, min(0.30, dur_weight))

    residual = 1.0 - (sgov_floor + sat_weight + dur_weight)
    core_weight = max(0.0, residual)

    # Core 분배: RuleSet 공식 비중
    core_config = {
        "SPX": 0.525,
        "NDX": 0.245,
        "DIV": 0.230,
    }
    core_alloc = {k: core_weight * w for k, w in core_config.items()}

    # Satellite 분배: EM:ENERGY = 2:1
    em_ratio = 2.0 / 3.0
    en_ratio = 1.0 / 3.0
    em_w = sat_weight * em_ratio
    en_w = sat_weight * en_ratio

    # Duration / SGOV 그대로
    weights = {
        "SPX": core_alloc["SPX"],
        "NDX": core_alloc["NDX"],
        "DIV": core_alloc["DIV"],
        "EM": em_w,
        "ENERGY": en_w,
        "DURATION": dur_weight,
        "SGOV": sgov_floor,
    }

    # 합이 1.0 이 되도록 미세 조정
    total = sum(weights.values())
    if total > 0:
        scale = 1.0 / total
        weights = {k: v * scale for k, v in weights.items()}

    return weights


# ==== CMA TAS Dynamic Threshold 연동 ====


def compute_cma_section(sig: Dict[str, float]) -> Dict[str, Any]:
    """
    FD / ML / Systemic 신호(sig)를 받아 CMA TAS 결과 dict를 반환.
    """
    vix = sig["vix"]
    hy_oas = sig["hy_oas"]
    dd_3y = sig["drawdown"]             # engine convention: -0.25 = -25%
    dd_10y = sig.get("dd_10y", dd_3y)   # TODO: 10Y series 도입 시 교체
    ml_risk = sig["ml_risk"]
    systemic_bucket = sig["systemic_bucket"]

    # 새 State 결정 로직 사용
    state_name = determine_state_from_signals(sig)

    tas_input = CmaTasInput(
        vix=vix,
        hy_oas=hy_oas,
        dd_3y=dd_3y,
        dd_10y=dd_10y,
        ml_risk=ml_risk,
        state=state_name,
        systemic_bucket=systemic_bucket,
    )

    tas_output = compute_cma_tas(tas_input)

    return {
        "deploy_factor": tas_output.deploy_factor,
        "threshold": tas_output.final_threshold,
        "meta": tas_output.meta,
    }


def main():
    market = load_latest_market()
    sig = build_signals(market)
    weights = compute_portfolio_target(sig)

    # CMA TAS 계산
    cma = compute_cma_section(sig)

    print("[INFO] ==== FD / ML / Systemic Summary ====")
    print(f"FX: {sig['fx_rate']:.2f}, FXW: {sig['fxw']:.3f}, FX vol(21D): {sig['fx_vol']:.4f}")
    print(f"VIX: {sig['vix']:.2f}, HY OAS(bps): {sig['hy_oas']:.1f}, Drawdown(3Y): {sig['drawdown']:.3f}")
    print(f"YC spread(bps): {sig['yc_spread_bps']:.1f}, FFR Upper: {sig['ffr_upper']:.2f}")
    print(
        f"MacroScore: {sig['macro_score']:.3f}, ML_Risk: {sig['ml_risk']:.3f}, "
        f"ML_Opp: {sig['ml_opp']:.3f}, ML_Regime: {sig['ml_regime']:.3f}"
    )
    print(f"Systemic Level: {sig['systemic_level']:.3f}, Bucket: {sig['systemic_bucket']}")

    print("[INFO] ==== Target Portfolio Weights (AURORA Rev12.1b) ====")
    for k in ["SPX", "NDX", "DIV", "EM", "ENERGY", "DURATION", "SGOV"]:
        print(f"  {k}: {weights[k]*100:5.2f}%")

    print("[INFO] ==== CMA TAS (Dynamic Threshold) ====")
    print(
        f"  Threshold: {cma['threshold']*100:4.1f}%, "
        f"Deploy Factor: {cma['deploy_factor']*100:4.1f}%"
    )

    # 결과를 JSON 으로 저장 (Level-4 자동화 대비)
    today = datetime.now().strftime("%Y%m%d")
    out_path = DATA_DIR / f"aurora_target_weights_{today}.json"
    state_name = determine_state_from_signals(sig)

    out = {
        "date": today,
        "signals": {
            **sig,
            "state": state_name,
        },
        "weights": weights,
        "cma": cma,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[OK] Target weights JSON saved to: {out_path}")


if __name__ == "__main__":
    main()
