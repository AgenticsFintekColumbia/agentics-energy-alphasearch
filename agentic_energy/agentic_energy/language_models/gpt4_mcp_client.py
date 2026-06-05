"""scripts/test_gpt4_solve_client.py

Client test for agentic_energy.language_models.gpt4_mcp_server.
Mirrors the style of milp_mcp_client.py and llm_client.py.
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter

from agentic_energy.schemas import BatteryParams, DayInputs, SolveRequest, SolveResponse

load_dotenv()
os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")


def get_tool(tools, name: str):
    for tool in tools:
        if tool.name == name:
            return tool
    raise RuntimeError(f"Tool {name!r} not found. Available: {[t.name for t in tools]}")


def parse_solve_response(raw) -> SolveResponse:
    if isinstance(raw, dict):
        return SolveResponse.model_validate(raw)
    if isinstance(raw, str):
        return SolveResponse.model_validate(json.loads(raw))
    return SolveResponse.model_validate(raw)


def build_dummy_request() -> SolveRequest:
    battery = BatteryParams(
        capacity_MWh=20.0,
        soc_init=0.5,
        soc_min=0.10,
        soc_max=0.90,
        cmax_MW=6.0,
        dmax_MW=6.0,
        eta_c=0.95,
        eta_d=0.95,
        soc_target=0.5,
    )

    actual_prices = [0.12] * 6 + [0.15] * 6 + [0.22] * 6 + [0.16] * 6
    forecast_prices = [0.11] * 6 + [0.16] * 6 + [0.21] * 6 + [0.17] * 6
    demand = [0.9] * 24

    day = DayInputs(
        prices_buy=actual_prices,
        prices_sell=actual_prices,
        demand_MW=demand,
        prices_buy_forecast=forecast_prices,
        prices_sell_forecast=forecast_prices,
        demand_MW_forecast=demand,
        allow_export=True,
        dt_hours=1.0,
    )

    return SolveRequest(
        battery=battery,
        day=day,
        solver="gpt4",
        solver_opts={"model": os.getenv("OPENAI_MODEL", "gpt-4o"), "temperature": 0.0},
    )


def call_gpt4_solve_via_mcp(solve_request: SolveRequest) -> SolveResponse:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_energy.language_models.gpt4_mcp_server"],
        env=os.environ,
    )

    with MCPServerAdapter(params) as tools:
        print("Available tools:", [t.name for t in tools])
        gpt4_solve = get_tool(tools, "gpt4_solve")
        call_fn = getattr(gpt4_solve, "call", None) or getattr(gpt4_solve, "run", None) or getattr(gpt4_solve, "__call__", None)
        if call_fn is None:
            raise RuntimeError("gpt4_solve tool has no callable interface")
        raw = call_fn(solverequest=solve_request.model_dump(exclude_none=True))
        return parse_solve_response(raw)


if __name__ == "__main__":
    req = build_dummy_request()
    resp = call_gpt4_solve_via_mcp(req)
    print("Status:", resp.status)
    print("Objective cost:", resp.objective_cost)
    print("len(soc):", len(resp.soc or []))
    print("total charge:", sum(resp.charge_MW or []))
    print("total discharge:", sum(resp.discharge_MW or []))
