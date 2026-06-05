"""agentic_energy/language_models/gpt4_mcp_server.py

MCP server for GPT-4 battery arbitrage scheduling.

This version supports a two-stage CoT-style pipeline:
  1) planner call: asks GPT-4 to reason about low/high-price windows,
     constraints, and a candidate SoC trajectory;
  2) finalizer call: converts the plan into strict SolveResponse JSON.

The final output is still a SolveResponse and is locally repaired/recomputed
from the returned SoC trajectory, so downstream code can use it exactly like
MILP/heuristic/RL solvers.
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
    out = fallback if x is None else x
    if len(out) != T:
        raise ValueError(f"Expected length {T}, got {len(out)}")
    return [float(v) for v in out]


def _payload_from_request(request: SolveRequest) -> Dict[str, Any]:
    """Build the compact JSON payload shared by planner and finalizer prompts."""
    batt = request.battery
    day = request.day
    T = len(day.prices_buy)

    p_buy_actual = _arr(day.prices_buy, day.prices_buy, T)
    demand_actual = _arr(day.demand_MW, day.demand_MW, T)
    p_sell_actual = _arr(day.prices_sell, p_buy_actual, T) if day.allow_export else [0.0] * T

    p_buy_forecast = _arr(day.prices_buy_forecast, p_buy_actual, T)
    demand_forecast = _arr(day.demand_MW_forecast, demand_actual, T)
    p_sell_forecast = _arr(day.prices_sell_forecast, p_buy_forecast, T) if day.allow_export else [0.0] * T

    return {
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


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON, tolerating accidental surrounding text."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _planner_schema_hint(T: int) -> str:
    return f"""
Return ONLY valid JSON with this shape:
{{
  "price_summary": "brief description of low/high-price structure",
  "low_price_hours": [0, 1],
  "high_price_hours": [18, 19],
  "constraint_notes": "brief notes about SoC, power limits, terminal target, and export rules",
  "reasoning_steps": [
    "short step 1",
    "short step 2",
    "short step 3"
  ],
  "candidate_soc": [s0, s1, ..., s{T}],
  "candidate_objective_cost": 0.0
}}
Lengths: candidate_soc has {T+1} values.
Keep reasoning_steps concise. No markdown. No code fence.
"""


def _final_schema_hint(T: int) -> str:
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


def _build_direct_messages(request: SolveRequest) -> List[Dict[str, str]]:
    """Single-call structured solver prompt."""
    payload = _payload_from_request(request)
    T = payload["horizon"]["T"]

    system = (
        "You are an expert battery arbitrage optimizer. Use only forecast inputs "
        "to choose the schedule, but report objective_cost under the actual inputs. "
        "Respect battery physics, SoC bounds, power limits, initial/terminal SoC, "
        "and no simultaneous charge/discharge."
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

{_final_schema_hint(T)}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_cot_planner_messages(request: SolveRequest) -> List[Dict[str, str]]:
    """First stage: explicit but concise reasoning plan + candidate SoC."""
    payload = _payload_from_request(request)
    T = payload["horizon"]["T"]

    system = (
        "You are an expert electricity-market battery arbitrage planner. "
        "Use explicit step-by-step reasoning to identify low-price charge windows, "
        "high-price discharge windows, SoC feasibility, and terminal behavior. "
        "Use forecast inputs for decisions and actual inputs only for ex-post cost estimation."
    )
    user = f"""
Create a concise chain-of-thought style planning trace for this battery arbitrage instance.
Do not return the final SolveResponse yet; return only the planning JSON.

Battery dynamics:
SoC[t+1] = SoC[t] + (eta_c * charge_MW[t] * dt_hours - discharge_MW[t] * dt_hours / eta_d) / capacity_MWh.
Constraints: SoC bounds, charge/discharge limits, no simultaneous charge/discharge, initial SoC, terminal target, and export rule.

Problem data JSON:
{json.dumps(payload)}

Planning requirements:
1. Identify low-price hours suitable for charging.
2. Identify high-price hours suitable for discharging.
3. Check SoC headroom/available energy and power limits.
4. Propose a feasible candidate SoC trajectory.
5. Estimate realized cost using actual data after the candidate schedule is fixed.

{_planner_schema_hint(T)}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_cot_finalizer_messages(request: SolveRequest, plan: Dict[str, Any]) -> List[Dict[str, str]]:
    """Second stage: convert CoT plan into strict SolveResponse JSON."""
    payload = _payload_from_request(request)
    T = payload["horizon"]["T"]

    system = (
        "You are a strict JSON finalizer for battery arbitrage schedules. "
        "Convert the planning trace into a final SolveResponse. "
        "Do not include reasoning text outside the message field."
    )
    user = f"""
Use the problem data and planning trace below to produce the final schedule.
The schedule must satisfy battery dynamics, SoC bounds, power limits, no simultaneous charge/discharge, and import/export rules.

Problem data JSON:
{json.dumps(payload)}

Planning trace JSON:
{json.dumps(plan)}

Use the candidate_soc as the main trajectory if feasible; otherwise minimally adjust it.
Compute objective_cost using actual inputs after decisions are fixed.

{_final_schema_hint(T)}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _repair_from_soc(request: SolveRequest, raw_soc: List[float], message: Optional[str] = None) -> SolveResponse:
    """Convert an LLM SoC trajectory into feasible flows and realized cost.

    This follows the proposed SoC deltas but clips each step to SoC bounds,
    power limits, no-export rules, and recomputes realized objective cost.
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
            headroom_mwh = max(0.0, (batt.soc_max - current) * C)
            c = min(delta_soc * C / (batt.eta_c * dt), batt.cmax_MW, headroom_mwh / (batt.eta_c * dt))
        elif delta_soc < -1e-9:
            available_mwh = max(0.0, (current - batt.soc_min) * C)
            d = min((-delta_soc) * C * batt.eta_d / dt, batt.dmax_MW, available_mwh * batt.eta_d / dt)
            if not day.allow_export:
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
        decision.append(1 if c > 1e-7 else (-1 if d > 1e-7 else 0))

    objective_cost = float(np.sum(p_buy * np.asarray(imp) * dt) - np.sum(p_sell * np.asarray(exp) * dt))

    return SolveResponse(
        status="success",
        message=message or "GPT-4 schedule parsed and locally repaired to satisfy battery feasibility constraints.",
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


def _chat_json(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: float) -> Dict[str, Any]:
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return _extract_json(completion.choices[0].message.content or "{}")


def solve_daily_gpt4(
    request: SolveRequest,
    model: Optional[str] = None,
    temperature: float = 0.0,
    use_cot: bool = True,
    return_reasoning_in_message: bool = False,
) -> SolveResponse:
    """Call GPT-4 and return a validated SolveResponse.

    If use_cot=True, uses a two-call CoT-style planner/finalizer pipeline.
    If use_cot=False, uses a single structured solver call.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_response(request, "OPENAI_API_KEY is not set.")

    model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        plan: Optional[Dict[str, Any]] = None
        if use_cot:
            plan = _chat_json(client, model, _build_cot_planner_messages(request), temperature)
            final = _chat_json(client, model, _build_cot_finalizer_messages(request, plan), temperature)
        else:
            final = _chat_json(client, model, _build_direct_messages(request), temperature)

        raw_soc = final.get("soc") or final.get("SOC_TRAJECTORY") or final.get("soc_trajectory")
        if raw_soc is None and plan is not None:
            raw_soc = plan.get("candidate_soc")
        if raw_soc is None:
            raise ValueError(f"GPT-4 did not return a SoC trajectory. Raw keys={list(final.keys())}")

        msg = "GPT-4 CoT planner/finalizer schedule parsed and locally repaired."
        if use_cot and return_reasoning_in_message and plan is not None:
            steps = plan.get("reasoning_steps") or []
            msg += " Plan: " + " | ".join(str(s) for s in steps[:4])

        return _repair_from_soc(request, raw_soc, message=msg)
    except Exception as exc:
        return _fallback_response(request, f"GPT-4 solve failed: {type(exc).__name__}: {exc}")


@mcp.tool()
def gpt4_solve(solverequest: SolveRequest) -> SolveResponse:
    """Run GPT-4 day-ahead battery solver and return schedules + realized cost."""
    opts = solverequest.solver_opts or {}
    model = opts.get("model") or os.getenv("OPENAI_MODEL", "gpt-4o")
    temperature = float(opts.get("temperature", 0.0))
    use_cot = bool(opts.get("use_cot", True))
    return_reasoning = bool(opts.get("return_reasoning_in_message", False))
    return solve_daily_gpt4(
        solverequest,
        model=model,
        temperature=temperature,
        use_cot=use_cot,
        return_reasoning_in_message=return_reasoning,
    )


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
