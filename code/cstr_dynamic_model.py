"""Standalone CSTR dynamic simulation code extracted from Physics-BO.

The original project only returned final-state values from nested CSTR ODE
functions. This module exposes the ODE right-hand side and full time trajectory
so a student can plot transient and steady-state behavior directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.integrate import odeint


STATE_LABELS = ("Ca", "Cb", "T", "Cc", "Cd")
CONCENTRATION_LABELS = ("Ca", "Cb", "Cc", "Cd")

DEFAULT_PRICES = {
    "PA": 5.0,
    "PB": 19.0,
    "PC": 7.0,
    "PD": 2.0,
    "PHEAT": 0.00001,
    "F": 5.0,
    "UA": 25.0,
}

DEFAULT_T_RANGE = 500.0 - 150.0


@dataclass(frozen=True)
class CSTRConfig:
    """Physical constants for one CSTR model variant."""

    name: str
    k01: float
    k02: float
    DH1: float
    DH2: float
    EA1: float = 21500.0
    EA2: float = 91500.0
    density: float = 1.0
    Cp: float = 1.0
    T_in: float = 300.0
    arrhenius_variant: str = "shifted"
    initial_state: Tuple[float, float, float, float, float] | None = (
        0.0147,
        1.4,
        674.8,
        0.65,
        1.5353,
    )


CONFIGS = {
    # Used by evaluate_bo_system_scaled_trust and evaluate_bo_system_scaled_ifac
    # in the original src/physics_bo/model.py.
    "trust": CSTRConfig(
        name="trust",
        k01=1.0,
        k02=10.0,
        DH1=-1300.0,
        DH2=-400.0,
        arrhenius_variant="shifted",
    ),
    # Used by evaluate_ifac_grid_system_scaled in the original model.py.
    "ifac_grid": CSTRConfig(
        name="ifac_grid",
        k01=1e10,
        k02=1e11,
        DH1=-600.0,
        DH2=-200.0,
        arrhenius_variant="absolute",
        initial_state=None,
    ),
}


@dataclass
class CSTRSimulationResult:
    t: np.ndarray
    states: np.ndarray
    derivatives: np.ndarray
    profit_profile: np.ndarray
    steady_state: np.ndarray
    steady_profit: float
    settling_time: float
    config: CSTRConfig
    frac_A: float
    Tc_scaled: float
    Tc: float


def get_config(name: str = "trust") -> CSTRConfig:
    """Return one of the original model presets."""

    try:
        return CONFIGS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(CONFIGS))
        raise ValueError(f"Unknown preset {name!r}. Valid presets: {valid}") from exc


def coolant_temperature(Tc_scaled: float, T_range: float = DEFAULT_T_RANGE) -> float:
    return 273.0 + Tc_scaled * T_range


def feed_concentrations(frac_A: float, prices: Dict[str, float]) -> Tuple[float, float]:
    F_in = prices["F"]
    Ca_in = frac_A * F_in
    Cd_in = (1.0 - frac_A) * F_in
    return Ca_in, Cd_in


def initial_state_for(
    config: CSTRConfig,
    frac_A: float,
    prices: Dict[str, float] | None = None,
) -> np.ndarray:
    """Return the initial condition used by the selected original model."""

    prices = DEFAULT_PRICES if prices is None else prices
    if config.initial_state is not None:
        return np.array(config.initial_state, dtype=float)

    Ca_in, Cd_in = feed_concentrations(frac_A, prices)
    return np.array([Ca_in, 0.0, config.T_in, 0.0, Cd_in], dtype=float)


def reaction_rates(state: np.ndarray, config: CSTRConfig) -> Tuple[float, float]:
    Ca, _, T, _, Cd = state

    if config.arrhenius_variant == "shifted":
        k1 = config.k01 * np.exp(-config.EA1 / (8.314 * (1.8 * T + 1000.0)))
        k2 = config.k02 * np.exp(-config.EA2 / (8.314 * (2.8 * T + 1000.0)))
    elif config.arrhenius_variant == "absolute":
        k1 = config.k01 * np.exp(-config.EA1 / (8.314 * T))
        k2 = config.k02 * np.exp(-config.EA2 / (8.314 * T))
    else:
        raise ValueError(f"Unknown Arrhenius variant: {config.arrhenius_variant}")

    R1 = k1 * Ca * Cd
    R2 = k2 * Ca
    return R1, R2


def cstr_rhs(
    state: np.ndarray,
    t: float,
    frac_A: float,
    Tc_scaled: float,
    config: CSTRConfig,
    prices: Dict[str, float] | None = None,
    T_range: float = DEFAULT_T_RANGE,
) -> np.ndarray:
    """ODE right-hand side for the CSTR dynamic model.

    State order: [Ca, Cb, T, Cc, Cd].
    Inputs: frac_A is the feed fraction of A; Tc_scaled is scaled coolant temp.
    """

    prices = DEFAULT_PRICES if prices is None else prices
    Ca, Cb, T, Cc, Cd = state

    F_in = prices["F"]
    F = prices["F"]
    Tc = coolant_temperature(Tc_scaled, T_range)
    Ca_in, Cd_in = feed_concentrations(frac_A, prices)
    R1, R2 = reaction_rates(state, config)

    dCadt = (F_in / 1000.0) * (Ca_in - Ca) - R1 - R2
    dCbdt = (F_in / 1000.0) * (0.0 - Cb) + R1
    dTdt = (
        (-T / 1000.0) * (F_in - F)
        + (F_in * config.T_in - F * T) / 1000.0
        + (1.0 / config.density / config.Cp) * (-config.DH1 * R1 + -config.DH2 * R2)
        + (1.0 / 1000.0 / config.density / config.Cp) * prices["UA"] * (Tc - T)
    )
    dCcdt = (F_in / 1000.0) * (0.0 - Cc) + R2
    dCddt = (F_in / 1000.0) * (Cd_in - Cd) - R1

    return np.array([dCadt, dCbdt, dTdt, dCcdt, dCddt], dtype=float)


def profit_from_state(
    state: np.ndarray,
    frac_A: float,
    Tc_scaled: float,
    prices: Dict[str, float] | None = None,
    T_range: float = DEFAULT_T_RANGE,
) -> float:
    prices = DEFAULT_PRICES if prices is None else prices
    Ca, Cb, T, Cc, Cd = state
    F = prices["F"]
    Tc = coolant_temperature(Tc_scaled, T_range)
    Ca_in, Cd_in = feed_concentrations(frac_A, prices)

    return (
        prices["PB"] * F * Cb
        + prices["PC"] * F * Cc
        - prices["PA"] * F * Ca_in
        - prices["PD"] * F * Cd_in
        - prices["PHEAT"] * prices["UA"] * np.abs(Tc - T)
    )


def estimate_settling_time(
    t: np.ndarray,
    states: np.ndarray,
    tolerance: float = 0.02,
) -> float:
    """Estimate when all states stay within tolerance of final values."""

    steady_state = states[-1]
    scale = np.maximum(np.abs(steady_state), 1.0)
    relative_error = np.max(np.abs((states - steady_state) / scale), axis=1)

    for i in range(len(t)):
        if np.all(relative_error[i:] <= tolerance):
            return float(t[i])
    return float(t[-1])


def simulate_cstr(
    frac_A: float = 0.10,
    Tc_scaled: float = 0.80,
    preset: str = "trust",
    t_end: float = 1500.0,
    n_points: int = 600,
    prices: Dict[str, float] | None = None,
    T_range: float = DEFAULT_T_RANGE,
    settling_tolerance: float = 0.02,
) -> CSTRSimulationResult:
    """Simulate the CSTR and return full transient plus final steady-state data."""

    prices = DEFAULT_PRICES if prices is None else prices
    config = get_config(preset)
    x0 = initial_state_for(config, frac_A, prices)
    t = np.linspace(0.0, t_end, n_points)

    states = odeint(
        cstr_rhs,
        x0,
        t,
        args=(frac_A, Tc_scaled, config, prices, T_range),
    )
    derivatives = np.array(
        [cstr_rhs(state, time, frac_A, Tc_scaled, config, prices, T_range) for state, time in zip(states, t)]
    )
    profit_profile = np.array(
        [profit_from_state(state, frac_A, Tc_scaled, prices, T_range) for state in states]
    )

    steady_state = states[-1]
    steady_profit = float(profit_profile[-1])
    settling_time = estimate_settling_time(t, states, tolerance=settling_tolerance)

    return CSTRSimulationResult(
        t=t,
        states=states,
        derivatives=derivatives,
        profit_profile=profit_profile,
        steady_state=steady_state,
        steady_profit=steady_profit,
        settling_time=settling_time,
        config=config,
        frac_A=frac_A,
        Tc_scaled=Tc_scaled,
        Tc=coolant_temperature(Tc_scaled, T_range),
    )
