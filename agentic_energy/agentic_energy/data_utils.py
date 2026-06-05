# agentic_energy_app/data_utils.py

import asyncio
from typing import Tuple, List, Sequence, Optional

import pandas as pd
import numpy as np

from agentic_energy.data_loader import EnergyDataLoader
from agentic_energy.schemas import DayInputs, PlotResponse, BatteryParams
from agentic_energy.milp.milp_mcp_server import records_to_arrays

from agentic_energy.mcp_clients import run_price_forecast_plot


async def _load_energy_day_async(
    region: str = "ITALY",
    date_str: str = "2018-01-01",
    forecast_type: str = "LSTM",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Async helper for EnergyDataLoader."""
    actual_obj = EnergyDataLoader(region=region, data_version="actual")
    actual_obj.load_region_data()
    actual = await actual_obj.get_filtered_data(date_str, date_str)

    forecast_obj = EnergyDataLoader(
        region=region,
        data_version="forecast",
        forecast_type=forecast_type,
    )
    forecast_obj.load_region_data()
    forecast = await forecast_obj.get_filtered_data(date_str, date_str)

    return actual, forecast


def load_energy_day(
    region: str,
    date_str: str,
    forecast_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sync wrapper."""
    return asyncio.run(_load_energy_day_async(region, date_str, forecast_type))



def make_day_inputs_from_forecast(
    prices: List[float],
    demand: List[float],
    dt_hours: float = 1.0,
) -> DayInputs:
    """Create DayInputs using forecast prices and (actual) demand."""
    return DayInputs(
        prices_buy=prices,
        prices_sell=prices,
        demand_MW=demand,
        allow_export=True,
        dt_hours=dt_hours,
        prices_buy_forecast=prices,
        prices_sell_forecast=prices,
        demand_MW_forecast=demand,
    )


def run_forecast_step(
    region: str,
    date_str: str,
    forecast_type: str,
    forecast_plot_path: str,
) -> Tuple[DayInputs, pd.DataFrame, pd.DataFrame, PlotResponse]:
    """
    High-level helper used by Streamlit app:

    1) Load actual + forecast from EnergyDataLoader
    2) Extract prices + demand
    3) Build DayInputs
    4) Call visualization MCP to make price forecast plot
    """
    actual_df, forecast_df = load_energy_day(region, date_str, forecast_type)
    forecast_prices, forecast_demand = records_to_arrays(forecast_df)
    actual_prices, actual_demand = records_to_arrays(actual_df)

    dt_hours = 1.0  # 🔧 infer from timestamps if you want
    day_inputs = make_day_inputs_from_forecast(forecast_prices,forecast_demand, dt_hours)

    plot = run_price_forecast_plot(
        prices=forecast_prices,
        dt_hours=dt_hours,
        out_path=forecast_plot_path,
        title=f"Price Forecast - {region} {date_str} ({forecast_type})",
    )

    return day_inputs, actual_df, forecast_df, plot

def make_day_inputs_actual_and_forecast(
    actual_prices: List[float],
    actual_demand: List[float],
    forecast_prices: List[float],
    forecast_demand: List[float],
    dt_hours: float = 1.0,
    allow_export: bool = True,
) -> DayInputs:
    """Create DayInputs with actuals for ex-post evaluation and forecasts for decisions."""
    return DayInputs(
        prices_buy=actual_prices,
        prices_sell=actual_prices,
        demand_MW=actual_demand,
        allow_export=allow_export,
        dt_hours=dt_hours,
        prices_buy_forecast=forecast_prices,
        prices_sell_forecast=forecast_prices,
        demand_MW_forecast=forecast_demand,
    )

def run_forecast_step(
    region: str,
    date_str: str,
    forecast_type: str,
    forecast_plot_path: str,
) -> Tuple[DayInputs, pd.DataFrame, pd.DataFrame, PlotResponse]:
    """Load actual + forecast, build DayInputs, and create forecast plot."""
    actual_df, forecast_df = load_energy_day(region, date_str, forecast_type)
    forecast_prices, forecast_demand = records_to_arrays(forecast_df)
    actual_prices, actual_demand = records_to_arrays(actual_df)

    dt_hours = 1.0
    day_inputs = make_day_inputs_actual_and_forecast(
        actual_prices=actual_prices,
        actual_demand=actual_demand,
        forecast_prices=forecast_prices,
        forecast_demand=forecast_demand,
        dt_hours=dt_hours,
        allow_export=True,
    )

    plot = run_price_forecast_plot(
        prices=forecast_prices,
        dt_hours=dt_hours,
        out_path=forecast_plot_path,
        title=f"Price Forecast - {region} {date_str} ({forecast_type})",
    )
    return day_inputs, actual_df, forecast_df, plot

