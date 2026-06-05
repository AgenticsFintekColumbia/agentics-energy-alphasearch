from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import cvxpy as cp

from agentic_energy.schemas import BatteryParams, DayInputs, SolveRequest, SolveResponse
from .milp_mcp_server import solve_daily_milp
# from agentic_energy.mcp_clients import cost_from_soc


def estimate_bias(actual_hist: np.ndarray, forecast_hist: np.ndarray) -> np.ndarray:
    actual_hist = np.asarray(actual_hist, dtype=float)
    forecast_hist = np.asarray(forecast_hist, dtype=float)
    if actual_hist.shape != forecast_hist.shape:
        raise ValueError("actual_hist and forecast_hist must have same shape")
    return np.mean(actual_hist - forecast_hist, axis=0)


def apply_bias(forecast_next: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(forecast_next, dtype=float) + np.asarray(bias, dtype=float)


def residual_matrix(actual_hist: np.ndarray, forecast_hist: np.ndarray, bias_correct: bool = True) -> np.ndarray:
    actual_hist = np.asarray(actual_hist, dtype=float)
    forecast_hist = np.asarray(forecast_hist, dtype=float)

    if bias_correct:
        b = estimate_bias(actual_hist, forecast_hist)
        forecast_hist = forecast_hist + b[None, :]

    return actual_hist - forecast_hist


def sample_scenarios(base_forecast: np.ndarray, residuals: np.ndarray, n_scenarios: int,
                     random_state: int = 42, block_bootstrap: bool = True) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    residuals = np.asarray(residuals, dtype=float)
    base_forecast = np.asarray(base_forecast, dtype=float)

    N, T = residuals.shape
    out = np.zeros((n_scenarios, T), dtype=float)

    if block_bootstrap:
        idx = rng.integers(0, N, size=n_scenarios)
        for s in range(n_scenarios):
            out[s] = base_forecast + residuals[idx[s]]
    else:
        for s in range(n_scenarios):
            draw = np.array([residuals[rng.integers(0, N), t] for t in range(T)])
            out[s] = base_forecast + draw

    return out


def solve_bias_corrected_deterministic(
    batt: BatteryParams,
    price_forecast_next: np.ndarray,
    demand_forecast_next: np.ndarray,
    price_bias: np.ndarray,
    demand_bias: np.ndarray,
    solver: Optional[str] = None,
    solver_opts: Optional[Dict[str, Any]] = None,
) -> SolveResponse:
    price_corr = apply_bias(price_forecast_next, price_bias)
    demand_corr = apply_bias(demand_forecast_next, demand_bias)

    req = DayInputs(
        prices_buy=price_corr.tolist(),
        prices_sell=price_corr.tolist(),
        demand_MW=demand_corr.tolist(),
        allow_export=True,
        dt_hours=1.0,
        prices_buy_forecast=price_corr.tolist(),
        prices_sell_forecast=price_corr.tolist(),
        demand_MW_forecast=demand_corr.tolist(),
    )
    return solve_daily_milp(batt, req, solver=cp.GUROBI, solver_opts=solver_opts)

def solve_scenario_expected_cost(
    batt: BatteryParams,
    price_scenarios: np.ndarray,
    demand_scenarios: np.ndarray,
    dt_hours: float = 1.0,
    prices_sell_scenarios: Optional[np.ndarray] = None,
    solver: Optional[str] = "GUROBI",
    solver_opts: Optional[Dict[str, Any]] = None,
) -> SolveResponse:

    solver_opts = {} if solver_opts is None else dict(solver_opts)

    P = np.asarray(price_scenarios, dtype=float)
    D = np.asarray(demand_scenarios, dtype=float)
    S, T = P.shape
    Psell = P if prices_sell_scenarios is None else np.asarray(prices_sell_scenarios, dtype=float)

    c = cp.Variable(T, nonneg=True)
    d = cp.Variable(T, nonneg=True)
    soc = cp.Variable(T + 1)
    yc = cp.Variable(T, boolean=True)
    yd = cp.Variable(T, boolean=True)

    imp = cp.Variable((S, T), nonneg=True)
    exp = cp.Variable((S, T), nonneg=True)

    soc0 = float(batt.soc_init)
    # soct = soc0 if batt.soc_target is None else float(batt.soc_target)
    C = float(batt.capacity_MWh)

    cons = [
        soc[0] == soc0,
        # soc[T] == soct,
        soc >= batt.soc_min,
        soc <= batt.soc_max,
    ]

    for t in range(T):
        cons += [
            c[t] <= batt.cmax_MW * yc[t],
            d[t] <= batt.dmax_MW * yd[t],
            yc[t] + yd[t] <= 1,
            soc[t+1] == soc[t] + ((batt.eta_c * c[t] * dt_hours) - ((d[t] * dt_hours )/ batt.eta_d)) / C,
        ]

    losses = []
    for s in range(S):
        for t in range(T):
            cons += [imp[s, t] - exp[s, t] == D[s, t] + c[t] - d[t]]
            # cons += [imp[s, t] - exp[s, t] == c[t] - d[t]] # for ERCOT
        Ls = cp.sum(cp.multiply(P[s], imp[s]) * dt_hours) - cp.sum(cp.multiply(Psell[s], exp[s]) * dt_hours)
        losses.append(Ls)

    mean_loss = (1.0 / S) * cp.sum(cp.hstack(losses))
    prob = cp.Problem(cp.Minimize(mean_loss), cons)
    prob.solve(solver=cp.GUROBI, **solver_opts)

    decision = []
    for t in range(T):
        if yc.value[t] > 0.5:
            decision.append(1)
        elif yd.value[t] > 0.5:
            decision.append(-1)
        else:
            decision.append(0)

    return SolveResponse(
        status=prob.status,
        message=None,
        objective_cost=float(prob.value),
        charge_MW=c.value.tolist(),
        discharge_MW=d.value.tolist(),
        import_MW=None,
        export_MW=None,
        soc=soc.value.tolist(),
        decision=decision,
        confidence=None,
    )


def solve_scenario_cvar(
    batt: BatteryParams,
    price_scenarios: np.ndarray,
    demand_scenarios: np.ndarray,
    beta: float = 0.95,
    lam: float = 0.5,
    dt_hours: float = 1.0,
    prices_sell_scenarios: Optional[np.ndarray] = None,
    solver: Optional[str] = "GUROBI",
    solver_opts: Optional[Dict[str, Any]] = None,
) -> SolveResponse:
    solver_opts = {} if solver_opts is None else dict(solver_opts)

    P = np.asarray(price_scenarios, dtype=float)
    D = np.asarray(demand_scenarios, dtype=float)
    S, T = P.shape

    Psell = P if prices_sell_scenarios is None else np.asarray(prices_sell_scenarios, dtype=float)

    c = cp.Variable(T, nonneg=True)
    d = cp.Variable(T, nonneg=True)
    soc = cp.Variable(T + 1)
    yc = cp.Variable(T, boolean=True)
    yd = cp.Variable(T, boolean=True)

    imp = cp.Variable((S, T), nonneg=True)
    exp = cp.Variable((S, T), nonneg=True)

    alpha = cp.Variable()
    z = cp.Variable(S, nonneg=True)

    soc0 = float(batt.soc_init)
    # soct = soc0 if batt.soc_target is None else float(batt.soc_target)
    C = float(batt.capacity_MWh)

    cons = [
        soc[0] == soc0,
        # soc[T] == soct,
        soc >= batt.soc_min,
        soc <= batt.soc_max,
    ]

    for t in range(T):
        cons += [
            c[t] <= batt.cmax_MW * yc[t],
            d[t] <= batt.dmax_MW * yd[t],
            yc[t] + yd[t] <= 1,
            soc[t+1] == soc[t] + (batt.eta_c * c[t] * dt_hours - d[t] * dt_hours / batt.eta_d) / C,
        ]

    losses = []
    for s in range(S):
        for t in range(T):
            cons += [imp[s, t] - exp[s, t] == D[s, t] + c[t] - d[t]]
            # cons += [imp[s, t] - exp[s, t] == c[t] - d[t]] # for ERCOT
        Ls = cp.sum(cp.multiply(P[s], imp[s]) * dt_hours) - cp.sum(cp.multiply(Psell[s], exp[s]) * dt_hours)
        losses.append(Ls)
        cons += [z[s] >= Ls - alpha]

    mean_loss = (1.0 / S) * cp.sum(cp.hstack(losses))
    cvar = alpha + (1.0 / ((1.0 - beta) * S)) * cp.sum(z)

    prob = cp.Problem(cp.Minimize((1.0 - lam) * mean_loss + lam * cvar), cons)
    prob.solve(solver=cp.GUROBI, **solver_opts)

    decision = []
    for t in range(T):
        if yc.value[t] > 0.5:
            decision.append(1)
        elif yd.value[t] > 0.5:
            decision.append(-1)
        else:
            decision.append(0)

    return SolveResponse(
        status=prob.status,
        message=None,
        objective_cost=float(prob.value),
        charge_MW=c.value.tolist(),
        discharge_MW=d.value.tolist(),
        import_MW=None,
        export_MW=None,
        soc=soc.value.tolist(),
        decision=decision,
        confidence=None,
    )