import sys
import json
from pathlib import Path
from typing import Dict, Any

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
    """data/ 폴더에서 가장 최근 market_data_*.json 파일을 불러온다."""
    files = sorted(DATA_DIR.glob("market_data_*.json"))
    if not files:
        raise FileNotFoundError("data/ 폴더에 market_data_*.json 이 없습니다.")
    
    latest = files[-1]
    with latest.open("r", encoding="utf-8") as f:
        market = json.load(f)

    # --- FX placeholder ---
    if "fx" not in market:
        market["fx"] = {
            "latest": None
        }

    # --- SPX placeholder ---
    if "spx" not in market:
        market["spx"] = {}

    # --- RISK placeholder ---
    if "risk" not in market:
        # 일단 최소 구조만 만들어 둔다.
        # 나중에 VIX, HY OAS, YC spread 등 실제 값으로 채워넣을 예정.
        market["risk"] = {
            "vix": 0.0,
            "hy_oas": 0.0,
            "yc_spread": 0.0,
        }
    # ----------------------

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

    # Systemic 레이어
    systemic_bucket = determine_systemic_bucket(systemic_level)(
        hy_oas=hy_oas,
        yc_spread=yc_spread_bps,
        macro_score=macro_score,
        ml_regime=ml_regime,
        drawdown=drawdown,
    )
    systemic_bucket = compute_systemic_bucket(systemic_level)

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


def main():
    market = load_latest_market()
    sig = build_signals(market)
    weights = compute_portfolio_target(sig)

    print("[INFO] ==== FD / ML / Systemic Summary ====")
    print(f"FX: {sig['fx_rate']:.2f}, FXW: {sig['fxw']:.3f}, FX vol(21D): {sig['fx_vol']:.4f}")
    print(f"VIX: {sig['vix']:.2f}, HY OAS(bps): {sig['hy_oas']:.1f}, Drawdown(3Y): {sig['drawdown']:.3f}")
    print(f"YC spread(bps): {sig['yc_spread_bps']:.1f}, FFR Upper: {sig['ffr_upper']:.2f}")
    print(f"MacroScore: {sig['macro_score']:.3f}, ML_Risk: {sig['ml_risk']:.3f}, "
          f"ML_Opp: {sig['ml_opp']:.3f}, ML_Regime: {sig['ml_regime']:.3f}")
    print(f"Systemic Level: {sig['systemic_level']:.3f}, Bucket: {sig['systemic_bucket']}")

    print("[INFO] ==== Target Portfolio Weights (AURORA Rev12.1b) ====")
    for k in ["SPX", "NDX", "DIV", "EM", "ENERGY", "DURATION", "SGOV"]:
        print(f"  {k}: {weights[k]*100:5.2f}%")

    # 결과를 JSON 으로 저장 (Level-4 자동화 대비)
    today = datetime.now().strftime("%Y%m%d")
    out_path = DATA_DIR / f"aurora_target_weights_{today}.json"
    out = {
        "date": today,
        "signals": sig,
        "weights": weights,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[OK] Target weights JSON saved to: {out_path}")


if __name__ == "__main__":
    main()
