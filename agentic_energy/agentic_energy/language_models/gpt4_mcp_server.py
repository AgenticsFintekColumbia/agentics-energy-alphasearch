"""agentic_energy/language_models/gpt4_mcp_server.py

MCP server for GPT-4 battery arbitrage scheduling.
It reuses the shared schemas (BatteryParams, DayInputs, SolveRequest,
SolveResponse) and follows the same MCP style as milp_mcp_server.py.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

from agentic_energy.schemas import (
    BatteryParams,
    DayInputs,
    EnergyDataRecord,
    SolveFromRecordsRequest,
    SolveRequest,
    SolveResponse,
)

load_dotenv()
os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")

mcp = FastMCP("GPT4")


def records_to_arrays(records: List[EnergyDataRecord]) -> Tuple[list, list]:
    """Reuse the same convention as milp_mcp_server: sort records by timestamp."""
    rows = [r for r in records if r.prices is not None and r.consumption is not None]
    rows.sort(key=lambda r: r.timestamps)
    prices = [float(r.prices) for r in rows]
    demand = [float(r.consumption) for r in rows]
    return prices, demand


def _arr(x: Optional[List[float]], fallback: List[float], T: int) -> List[float]:
    """Return a length-T float list with fallback handling."""
    if x is None:
        out = fallback
    else:
        out = x
    if len(out) != T:
        raise ValueError(f"Expected length {T}, got {len(out)}")
    return [float(v) for v in out]


def _json_schema_hint(T: int) -> str:
    return f"""
Return ONLY valid JSON with this shape:
{{
  "status": "success",
  "message": "brief strategy",
  "objective_cost": 0.0,
  "soc": [s0, s1, ..., s{T}],
  "charge_MW": [c0, ..., c{T-1}],
  "discharge_MW": [d0, ..., d{T-1}],
  "import_MW": [imp0, ..., imp{T-1}],
  "export_MW": [exp0, ..., exp{T-1}],
  "decision": [u0, ..., u{T-1}]
}}
Lengths: soc has {T+1} values; all other arrays have {T} values.
No markdown. No code fence.
"""


def _build_gpt4_messages(request: SolveRequest) -> List[Dict[str, str]]:
    """Build a compact GPT-4 prompt from existing SolveRequest schemas."""
    batt = request.battery
    day = request.day
    T = len(day.prices_buy)

    p_buy_actual = _arr(day.prices_buy, day.prices_buy, T)
    demand_actual = _arr(day.demand_MW, day.demand_MW, T)
    p_sell_actual = _arr(day.prices_sell, p_buy_actual, T) if day.allow_export else [0.0] * T

    p_buy_forecast = _arr(day.prices_buy_forecast, p_buy_actual, T)
    demand_forecast = _arr(day.demand_MW_forecast, demand_actual, T)
    p_sell_forecast = _arr(day.prices_sell_forecast, p_buy_forecast, T) if day.allow_export else [0.0] * T

    payload = {
        "battery": batt.model_dump(),
        "horizon": {"T": T, "dt_hours": day.dt_hours, "allow_export": day.allow_export},
        "forecast_inputs_for_decision": {
            "prices_buy": p_buy_forecast,
            "prices_sell": p_sell_forecast if day.allow_export else None,
            "demand_MW": demand_forecast,
        },
        "actual_inputs_for_ex_post_evaluation": {
            "prices_buy": p_buy_actual,
            "prices_sell": p_sell_actual if day.allow_export else None,
            "demand_MW": demand_actual,
        },
    }

    system = (
        "You are an expert battery arbitrage optimizer. Use only forecast inputs "
        "to choose the schedule, but report objective_cost under the actual inputs. "
        "Respect battery physics, SoC bounds, power limits, initial/terminal SoC, "
        "and no simultaneous charge/discharge. Prefer charging at low forecast prices "
        "and discharging at high forecast prices."
    )

    user = f"""
Solve this daily battery scheduling problem.

Battery dynamics:
SoC[t+1] = SoC[t] + (eta_c * charge_MW[t] * dt_hours - discharge_MW[t] * dt_hours / eta_d) / capacity_MWh.

Power constraints:
0 <= charge_MW[t] <= cmax_MW; 0 <= discharge_MW[t] <= dmax_MW; not both positive.

Grid balance:
net[t] = demand_MW[t] + charge_MW[t] - discharge_MW[t].
If export is allowed, import[t]-export[t]=net[t] with import/export >= 0.
If export is not allowed, export[t]=0 and import[t]>=net[t].

Problem data JSON:
{json.dumps(payload)}

{_json_schema_hint(T)}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON, tolerating accidental surrounding text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _repair_from_soc(request: SolveRequest, raw_soc: List[float]) -> SolveResponse:
    """Convert an LLM SoC trajectory into feasible flows and realized cost.

    This is intentionally local/minimal: it follows the proposed SoC deltas but clips
    each step to SoC bounds, power limits, and export rules.
    """
    batt, day = request.battery, request.day
    T = len(day.prices_buy)
    dt = float(day.dt_hours)
    C = float(batt.capacity_MWh)

    if len(raw_soc) < T + 1:
        raw_soc = list(raw_soc) + [raw_soc[-1] if raw_soc else batt.soc_init] * (T + 1 - len(raw_soc))
    raw_soc = [float(x) for x in raw_soc[: T + 1]]

    p_buy = np.asarray(day.prices_buy, dtype=float)
    p_sell = np.asarray(day.prices_sell if day.prices_sell is not None else day.prices_buy, dtype=float)
    demand = np.asarray(day.demand_MW, dtype=float)

    soc: List[float] = [float(np.clip(batt.soc_init, batt.soc_min, batt.soc_max))]
    charge: List[float] = []
    discharge: List[float] = []
    imp: List[float] = []
    exp: List[float] = []
    decision: List[int] = []

    for t in range(T):
        current = soc[-1]
        target_next = float(np.clip(raw_soc[t + 1], batt.soc_min, batt.soc_max))
        delta_soc = target_next - current

        c = 0.0
        d = 0.0
        if delta_soc > 1e-9:
            # desired stored-energy increase = delta_soc*C = eta_c*c*dt
            headroom_mwh = max(0.0, (batt.soc_max - current) * C)
            c = min(delta_soc * C / (batt.eta_c * dt), batt.cmax_MW, headroom_mwh / (batt.eta_c * dt))
        elif delta_soc < -1e-9:
            # desired stored-energy decrease = -delta_soc*C = d*dt/eta_d
            available_mwh = max(0.0, (current - batt.soc_min) * C)
            d = min((-delta_soc) * C * batt.eta_d / dt, batt.dmax_MW, available_mwh * batt.eta_d / dt)
            if not day.allow_export:
                # avoid net export in no-export markets
                d = min(d, float(demand[t]) + c)

        next_soc = current + (batt.eta_c * c * dt - d * dt / batt.eta_d) / C
        next_soc = float(np.clip(next_soc, batt.soc_min, batt.soc_max))

        net = float(demand[t] + c - d)
        if day.allow_export:
            im = max(net, 0.0)
            ex = max(-net, 0.0)
        else:
            im = max(net, 0.0)
            ex = 0.0

        charge.append(float(c))
        discharge.append(float(d))
        imp.append(float(im))
        exp.append(float(ex))
        soc.append(next_soc)
        if c > 1e-7:
            decision.append(1)
        elif d > 1e-7:
            decision.append(-1)
        else:
            decision.append(0)

    objective_cost = float(np.sum(p_buy * np.asarray(imp) * dt) - np.sum(p_sell * np.asarray(exp) * dt))

    return SolveResponse(
        status="success",
        message="GPT-4 schedule parsed and locally repaired to satisfy battery feasibility constraints.",
        objective_cost=objective_cost,
        charge_MW=charge,
        discharge_MW=discharge,
        import_MW=imp,
        export_MW=exp if day.allow_export else None,
        soc=soc,
        decision=decision,
    )


def _fallback_response(request: SolveRequest, msg: str) -> SolveResponse:
    T = len(request.day.prices_buy)
    dt = request.day.dt_hours
    cost = float(sum(request.day.prices_buy[t] * request.day.demand_MW[t] * dt for t in range(T)))
    return SolveResponse(
        status="error",
        message=msg,
        objective_cost=cost,
        charge_MW=[0.0] * T,
        discharge_MW=[0.0] * T,
        import_MW=[float(x) for x in request.day.demand_MW],
        export_MW=[0.0] * T if request.day.allow_export else None,
        soc=[request.battery.soc_init] * (T + 1),
        decision=[0] * T,
    )


def solve_daily_gpt4(request: SolveRequest, model: Optional[str] = None, temperature: float = 0.0) -> SolveResponse:
    """Call GPT-4 and return a validated SolveResponse."""
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_response(request, "OPENAI_API_KEY is not set.")

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=_build_gpt4_messages(request),
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        raw = _extract_json(content)

        # Prefer SoC trajectory because it is easier to make feasible deterministically.
        raw_soc = raw.get("soc") or raw.get("SOC_TRAJECTORY") or raw.get("soc_trajectory")
        if raw_soc is None:
            raise ValueError(f"GPT-4 did not return a SoC trajectory. Raw keys={list(raw.keys())}")

        return _repair_from_soc(request, raw_soc)
    except Exception as exc:
        return _fallback_response(request, f"GPT-4 solve failed: {type(exc).__name__}: {exc}")


@mcp.tool()
def gpt4_solve(solverequest: SolveRequest) -> SolveResponse:
    """Run GPT-4 day-ahead battery solver and return schedules + realized cost."""
    opts = solverequest.solver_opts or {}
    model = opts.get("model") or os.getenv("OPENAI_MODEL", "gpt-4o")
    temperature = float(opts.get("temperature", 0.0))
    return solve_daily_gpt4(solverequest, model=model, temperature=temperature)


@mcp.tool()
def gpt4_solve_from_records(solverecordrequest: SolveFromRecordsRequest) -> SolveResponse:
    """Run GPT-4 solver from EnergyDataRecord rows."""
    prices, demand = records_to_arrays(solverecordrequest.records)
    day = DayInputs(
        prices_buy=prices,
        prices_sell=prices,
        demand_MW=demand,
        allow_export=solverecordrequest.allow_export,
        dt_hours=solverecordrequest.dt_hours,
        prices_buy_forecast=prices,
        prices_sell_forecast=prices,
        demand_MW_forecast=demand,
    )
    req = SolveRequest(
        battery=solverecordrequest.battery,
        day=day,
        solver=solverecordrequest.solver or "gpt4",
        solver_opts=solverecordrequest.solver_opts,
    )
    return gpt4_solve(req)


if __name__ == "__main__":
    mcp.run(transport="stdio")
