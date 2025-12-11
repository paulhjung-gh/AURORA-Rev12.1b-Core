from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict
import math
import numpy as np
from scipy.stats import gaussian_kde
from collections import deque

# ============================================
# AuroraX Rev12.1b-KDE – Final Synced Engine (Python)
# Full Deterministic Edition with KDE Dynamic Anchor
#
# Matches Governance Protocol (12.1b-KDE),
# Whitepaper 12.1b-KDE, Market Data FD Spec,
# ML Layer 12.1b, Systemic Layer 12.1b,
# and RuleSet 12.1b-KDE.
#
# Notes:
#   - All Signals inputs (fx, vix, fx_vol, drawdown,
#     macro_score, ffr_upper, ml_* fields, systemic_level)
#     are deterministically defined by upstream FD components:
#       * FXW from USD/KRW sell-rate with KDE dynamic anchor
#       * fx_vol from 21D log-return sigma (non-annualized)
#       * drawdown from 3Y SPX window
#       * macro_score from ISM/PMI/CPI YoY/Unemployment
#       * ffr_upper from FRED DFEDTARU
#   - Engine formulas themselves are unchanged from Rev12.1,
#     ensuring strict backward compatibility while removing
#     any external scalar input dependency.
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
    def __init__(self, window: int = 130):
        self.window = window
        self.buffer = deque(maxlen=window)
        self.alpha = 0.01

    def add(self, fx: float):
        self.buffer.append(fx)

    def fxw(self, fx: float) -> float:
        if len(self.buffer) == 0:
            anchor = 1480.0
        else:
            data = np.array(self.buffer)
            kde = gaussian_kde(data)
            x = np.linspace(data.min()-100, data.max()+100, 1000)
            density = kde(x)
            anchor = x[np.argmax(density)]          # ← 순수 KDE mode (가중 평균 0%)
        raw = self.alpha * (fx - anchor)
        return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(raw))))
# ========================= 메인 엔진 클래스 =========================
class AuroraX121:
    def __init__(self):
        self.kde = KDE_AdaptiveFXW(window=130)

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
