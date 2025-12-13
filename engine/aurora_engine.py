from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict
import math
import numpy as np
from scipy.stats import gaussian_kde
from collections import deque

# ============================================
# AuroraX Rev12.4 – Core Tilt Edition (based on Rev12.1b-KDE)
# Full Deterministic Edition with KDE Dynamic Anchor + Core Internal Continuous Tilt
#
# Rev12.4 Change Log:
#   - Core Equity 내부 연속형 tilt 추가 (Σα = 0 보장, 총 Core 비중 불변)
#   - 신호: Yield Curve inversion, ML_Risk, VIX (sigmoid 기반 부드러운 반응)
#   - Tilt 범위 제한 ±0.08 (8%)
# ============================================

class State(Enum):
    S3_HARD = auto()
    S3_SOFT = auto()
    S2_HIGH_VOL = auto()
    S1_MILD = auto()
    S0_NORMAL = auto()
    S7_REPAIR = auto()

class ReversionMode(Enum):
    OFF = "off"
    SELL_ONLY = "sell_only"
    LIMITED = "limited"
    MONDAY_ONLY = "monday_only"
    FULL = "full"

@dataclass
class Signals:
    fx: float
    vix: float
    fx_vol: float = 0.0
    fx_daily_change_pct: float = 0.0
    systemic_level: float = 0.0
    drawdown: float = 0.0
    ml_opp: float = 0.5
    ml_risk: float = 0.5
    ml_regime: float = 0.5
    macro_score: float = 0.5
    ffr_upper: float = 5.0
    yc_spread: float = 0.0  # 10Y - 2Y 추가 (Rev12.4 필요)

@dataclass
class Portfolio:
    weights: Dict[str, float]
    cma_balance: float
    state: State = State.S0_NORMAL

@dataclass
class Decision:
    next_state: State
    fxw: float
    systemic_flag: str
    sgov_floor: float
    reversion_mode: ReversionMode
    cma_use: float
    satellite_target: float
    duration_target: float
    updated_weights: Dict[str, float]
    notes: Dict[str, str] = field(default_factory=dict)

# ==================== KDE 동적 Anchor 클래스 ====================
class KDE_AdaptiveFXW:
    def __init__(self, window: int = 130, alpha: float = 0.01):
        self.window = window
        self.alpha = alpha
        self.buffer = deque(maxlen=window)

    def preload(self, fx_series) -> None:
        for fx in fx_series:
            self.add(fx)

    def add(self, fx: float) -> None:
        try:
            self.buffer.append(float(fx))
        except (TypeError, ValueError):
            return

    def fxw(self, fx: float) -> float:
        if len(self.buffer) < 2:
            raise ValueError("FXW buffer not preloaded with sufficient history (need >=2, ideally 130).")
        data = np.asarray(self.buffer, dtype=float)
        kde = gaussian_kde(data)
        x = np.linspace(data.min() - 100.0, data.max() + 100.0, 1000)
        density = kde(x)
        anchor = float(x[np.argmax(density)])
        raw = self.alpha * (float(fx) - anchor)
        fxw_val = 1.0 / (1.0 + math.exp(raw))
        return max(0.0, min(1.0, fxw_val))

# ========================= 메인 엔진 클래스 =========================
class AuroraX121:
    def __init__(self, fx_history_130=None):
        self.kde = KDE_AdaptiveFXW(window=130)
        if fx_history_130 is not None:
            self.kde.preload(fx_history_130)

    def fxw(self, fx_rate: float) -> float:
        self.kde.add(fx_rate)
        return self.kde.fxw(fx_rate)

    def systemic(self, level: float) -> str:
        if level >= 0.80: return "C3"
        if level >= 0.70: return "C2"
        if level >= 0.55: return "C1"
        return "C0"

    def sgov_floor(self, fxw: float, fx_rate: float, ffr: float, ml_risk: float, systemic: str) -> float:
        base = max(0.22, 0.7 * (ffr / 100))
        fx_penalty = (1 - fxw) * 0.12
        k_penalty = 0.05 if fx_rate >= 1600 else 0.02 if fx_rate >= 1550 else 0.0
        ml_penalty = max(0, ml_risk - 0.60) * 0.08
        sys_penalty = {"C0":0, "C1":0.05, "C2":0.10, "C3":0.20}.get(systemic, 0)
        return min(0.80, max(0.22, base + fx_penalty + k_penalty + ml_penalty + sys_penalty))

    def satellite_target(self, systemic: str, ml_opp: float, fxw: float) -> float:
        if systemic != "C0":
            return 0.03
        base = 0.03
        expansion = min(0.06, 0.3 * ml_opp + 0.4 * fxw)
        return base + expansion

    def duration_target(self, macro_score: float, fxw: float, ml_risk: float) -> float:
        if macro_score < 0.70 or fxw < 0.55 or ml_risk > 0.50:
            return 0.0
        return 0.05 + 0.10 * (macro_score - 0.70) / 0.20

    def cma_usage(self, cma_bal: float, drawdown: float, fxw: float, ml_opp: float, ml_risk: float) -> float:
        if drawdown < 0.10 or fxw < 0.35 or ml_opp < 0.65 or ml_risk >= 0.90:
            return 0.0
        raw = (drawdown - 0.10) / 0.25
        raw = max(0.0, min(1.0, raw))
        factor = 0.1 + 0.7 * raw
        if ml_risk >= 0.75:
            factor *= 0.5
        return cma_bal * factor

    def repair_apply(self, w: Dict[str, float], sgov_floor: float, sat_tgt: float) -> Dict[str, float]:
        em = w.get("EM", 0.0)
        en = w.get("ENERGY", 0.0)
        s = em + en
        step = 0.01
        diff = sat_tgt - s
        adj = 0.0
        if abs(diff) > step:
            adj = step if diff > 0.0 else -step
        new_s = max(0.0, min(0.09, s + adj))
        em_ratio = 2.0 / 3.0
        en_ratio = 1.0 / 3.0
        others = 1.0 - s
        if others <= 0.0:
            nw = {k: 0.0 for k in w}
            nw["EM"] = new_s * em_ratio
            nw["ENERGY"] = new_s * en_ratio
            return nw
        new_others = 1.0 - new_s
        scale = new_others / others
        nw: Dict[str, float] = {}
        for k, v in w.items():
            if k == "EM":
                nw[k] = new_s * em_ratio
            elif k == "ENERGY":
                nw[k] = new_s * en_ratio
            else:
                nw[k] = v * scale
        return nw

    def reversion_mode(self, st: State, sys: str) -> ReversionMode:
        if sys in ("C2", "C3"):
            return ReversionMode.OFF
        if st == State.S3_HARD:
            return ReversionMode.SELL_ONLY
        if st in (State.S3_SOFT, State.S2_HIGH_VOL):
            return ReversionMode.LIMITED
        if st == State.S7_REPAIR:
            return ReversionMode.MONDAY_ONLY
        return ReversionMode.FULL

    # ==================== Rev12.4 추가: Core Internal Continuous Tilt ====================
    def _core_tilt_alpha(self, signals: Signals) -> Dict[str, float]:
        """
        Core 내부 연속형 tilt 계산 (Rev12.4)
        - YC inversion 강할수록 DIV +α / NDX -α
        - ML_Risk 높을수록 DIV +α
        - VIX 높을수록 DIV +α
        - sigmoid로 부드러운 반응
        - 총합 α = 0 보장
        """
        def sigmoid(z: float, k: float = 12.0) -> float:
            return 1.0 / (1.0 + math.exp(-k * z))

        # 1. Yield Curve inversion (10Y-2Y < 0 일수록 inversion 강함)
        yc_inv = max(0.0, -signals.yc_spread / 100)  # bps 단위 가정, inversion 강도 0~1+
        yc_alpha = 0.06 * sigmoid(yc_inv - 0.5)  # inversion 강할수록 +α

        # 2. ML_Risk 높을수록 DIV 방어 tilt
        ml_alpha = 0.05 * sigmoid(signals.ml_risk - 0.60)

        # 3. VIX 높을수록 DIV 방어 tilt
        vix_alpha = 0.04 * sigmoid((signals.vix - 20) / 10)

        total_positive = yc_alpha + ml_alpha + vix_alpha

        # SPX는 중립 buffer, NDX는 성장주라 -α 강하게
        alpha_spx = 0.0
        alpha_ndx = -total_positive * 1.2
        alpha_div = total_positive * 1.8  # DIV에 강하게 집중

        # 최대 ±8% 제한
        alpha_spx = max(-0.08, min(0.08, alpha_spx))
        alpha_ndx = max(-0.08, min(0.08, alpha_ndx))
        alpha_div = max(-0.08, min(0.08, alpha_div))

        # Σα = 0 강제 보정
        sum_alpha = alpha_spx + alpha_ndx + alpha_div
        if abs(sum_alpha) > 1e-6:
            alpha_spx -= sum_alpha / 3
            alpha_ndx -= sum_alpha / 3
            alpha_div -= sum_alpha / 3

        return {"SPX": alpha_spx, "NDX": alpha_ndx, "DIV": alpha_div}

    def compute_portfolio_target(self, signals: Signals) -> Dict[str, float]:
        """
        Rev12.4: Core 내부 tilt 적용 버전
        """
        fxw = self.fxw(signals.fx)
        systemic_bucket = self.systemic(signals.systemic_level)
        sgov_floor = self.sgov_floor(
            fxw=fxw,
            fx_rate=signals.fx,
            ffr=signals.ffr_upper,
            ml_risk=signals.ml_risk,
            systemic=systemic_bucket,
        )
        sat_weight = self.satellite_target(
            systemic=systemic_bucket,
            ml_opp=signals.ml_opp,
            fxw=fxw,
        )
        dur_weight = self.duration_target(
            macro_score=signals.macro_score,
            fxw=fxw,
            ml_risk=signals.ml_risk,
        )

        # Rev12.4 Core tilt 적용
        tilt = self._core_tilt_alpha(signals)

        core_base = {
            "SPX": 0.525 + tilt["SPX"],
            "NDX": 0.245 + tilt["NDX"],
            "DIV": 0.230 + tilt["DIV"],
        }

        # Core 내부 normalize (총합 1 보장)
        core_sum = sum(core_base.values())
        core_alloc = {k: v / core_sum for k, v in core_base.items()}

        em_w = sat_weight * (2.0 / 3.0)
        en_w = sat_weight * (1.0 / 3.0)
        gold_w = 0.05

        weights = {
            "SPX": core_alloc["SPX"] * (1.0 - gold_w - sgov_floor - sat_weight - dur_weight),
            "NDX": core_alloc["NDX"] * (1.0 - gold_w - sgov_floor - sat_weight - dur_weight),
            "DIV": core_alloc["DIV"] * (1.0 - gold_w - sgov_floor - sat_weight - dur_weight),
            "EM": em_w,
            "ENERGY": en_w,
            "DURATION": dur_weight,
            "SGOV": sgov_floor,
            "GOLD": gold_w,
        }

        # 최종 전체 normalize (안전장치)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights
