from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Optional
import math
import json
import numpy as np
import pandas as pd

# Adjust these imports if your package path differs.
from agentic_energy.schemas import BatteryParams
from agentic_energy.reinforcementlearning.trainer import train_rllib
from agentic_energy.reinforcementlearning.evaluator import rollout_day


def cost_from_soc(
    soc: Sequence[float],
    prices_buy: Sequence[float],
    demand_MW: Sequence[float],
    *,
    battery: BatteryParams,
    prices_sell: Optional[Sequence[float]] = None,
    allow_export: bool = False,
    dt_hours: float = 1.0,
):
    soc = np.asarray(soc, dtype=float)
    assert len(soc) >= 2, "SOC must include at least t=0 and t=1"
    T = len(soc) - 1

    prices_buy = np.asarray(prices_buy, dtype=float)
    demand_MW = np.asarray(demand_MW, dtype=float)
    assert len(prices_buy) == T and len(demand_MW) == T

    if prices_sell is None:
        prices_sell = prices_buy
    prices_sell = np.asarray(prices_sell, dtype=float)
    assert len(prices_sell) == T

    dE = (soc[1:] - soc[:-1]) * battery.capacity_MWh

    charge_MW = np.maximum(dE, 0.0) / (battery.eta_c * dt_hours)
    discharge_MW = np.maximum(-dE, 0.0) * (battery.eta_d / dt_hours)

    charge_MW = np.minimum(charge_MW, battery.cmax_MW)
    discharge_MW = np.minimum(discharge_MW, battery.dmax_MW)

    net = demand_MW + charge_MW - discharge_MW
    imp = np.maximum(net, 0.0)
    exp = np.maximum(-net, 0.0) if allow_export else np.zeros_like(net)

    cost = float(np.sum(prices_buy * imp * dt_hours) - np.sum(prices_sell * exp * dt_hours))

    return {
        "charge_MW": charge_MW,
        "discharge_MW": discharge_MW,
        "import_MW": imp,
        "export_MW": exp,
        "net_MW": net,
        "objective_cost": cost,
    }


def build_eval_dataframe(eval_req, res, battery: BatteryParams, *, dt_hours: Optional[float] = None) -> pd.DataFrame:
    day = eval_req.day
    allow_export = bool(day.allow_export)
    dt = float(dt_hours if dt_hours is not None else day.dt_hours)

    prices_actual = np.asarray(day.prices_buy, dtype=float)
    prices_forecast = np.asarray(
        day.prices_buy_forecast if day.prices_buy_forecast is not None else day.prices_buy,
        dtype=float,
    )
    actual_demand = np.asarray(day.demand_MW, dtype=float)
    forecast_demand = np.asarray(
        day.demand_MW_forecast if day.demand_MW_forecast is not None else day.demand_MW,
        dtype=float,
    )
    prices_sell = np.asarray(
        day.prices_sell if day.prices_sell is not None else day.prices_buy,
        dtype=float,
    )

    out = cost_from_soc(
        soc=res.soc,
        prices_buy=prices_actual,
        demand_MW=actual_demand,
        battery=battery,
        prices_sell=prices_sell,
        allow_export=allow_export,
        dt_hours=dt,
    )

    T = len(prices_actual)
    soc = np.asarray(res.soc, dtype=float)
    if len(soc) != T + 1:
        raise ValueError(f"Expected SOC length T+1={T+1}, got {len(soc)}")

    # timestamps if present, else simple integer step index
    timestamps = getattr(day, "timestamps", None)
    if timestamps is None:
        step = np.arange(T)
        timestamp_col = step
    else:
        timestamp_col = timestamps[:T]

    df = pd.DataFrame({
        "t": np.arange(T),
        "timestamp": timestamp_col,
        "prices_actual": prices_actual,
        "prices_forecast": prices_forecast,
        "actual_demand": actual_demand,
        "forecast_demand": forecast_demand,
        "soc": soc[:-1],
        "soc_next": soc[1:],
        "charge_MW": out["charge_MW"],
        "discharge_MW": out["discharge_MW"],
        "import_MW": out["import_MW"],
        "export_MW": out["export_MW"],
        "net_MW": out["net_MW"],
    })

    df["profit_step"] = (df["discharge_MW"] - df["charge_MW"]) * df["prices_actual"] * dt
    # df["cost_step"] = df["prices_actual"] * df["import_MW"] * dt - df["prices_sell"] * df["export_MW"] * dt
    df["cum_profit"] = df["profit_step"].cumsum()
    # df["cum_cost"] = df["cost_step"].cumsum()
    df["objective_cost_recomputed"] = out["objective_cost"]
    df["objective_cost_policy"] = float(res.objective_cost)
    # df["allow_export"] = allow_export
    # df["dt_hours"] = dt

    return df


def tcrit_95(df: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    }
    if df <= 0:
        return float("nan")
    return table.get(df, 1.96)


def summarize_seed_scores(scores: Sequence[float]) -> dict:
    x = np.asarray(scores, dtype=float)
    n = len(x)
    mean = float(np.mean(x)) if n else float("nan")
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else float("nan")
    tcrit = tcrit_95(n - 1) if n > 1 else float("nan")
    ci_low = mean - tcrit * se if n > 1 else mean
    ci_high = mean + tcrit * se if n > 1 else mean
    return {
        "n_seeds": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def run_multi_seed_experiment_detailed(
    *,
    train_req,
    train_days,
    test_days: Iterable,
    settings=None,
    num_iterations: int = 50,
    obs_mode: str = "compact",
    obs_window: int = 24,
    save_root: str | Path = "runs/rllib_battery_multiseed_detailed",
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
):
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    trajectories_dir = save_root / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    battery = train_req.battery
    seed_rows = []
    daily_rows = []

    for seed in seeds:
        print(f"\n===== TRAINING SEED {seed} =====")
        ckpt = train_rllib(
            train_req,
            train_days,
            settings=settings,
            num_iterations=num_iterations,
            save_dir=str(save_root / f"seed_{seed}"),
            obs_mode=obs_mode,
            obs_window=obs_window,
            seed=seed,
        )
        ckpt = str(save_root / f"seed_{seed}")
        day_costs = []
        day_profits = []

        for day_idx, day in enumerate(test_days):
            eval_req = train_req.model_copy(update={"day": day})
            res = rollout_day(
                ckpt,
                eval_req,
                obs_mode=obs_mode,
                obs_window=obs_window,
            )

            df = build_eval_dataframe(eval_req, res, battery)
            traj_path = trajectories_dir / f"seed_{seed}_day_{day_idx}.csv"
            df.to_csv(traj_path, index=False)

            objective_cost = float(df["objective_cost_recomputed"].iloc[0])
            total_profit = float(df["profit_step"].sum())
            # total_cost_from_steps = float(df["cost_step"].sum())

            day_costs.append(objective_cost)
            day_profits.append(total_profit)

            daily_rows.append({
                "day_idx": day_idx,
                "dataframe": df,
                "trajectory_csv": str(traj_path),
            })
        
        seed_rows.append({
            "seed": seed,
            "daily_rows": daily_rows,
        })

            # daily_rows.append({
            #     "seed": seed,
            #     "day_idx": day_idx,
            #     "trajectory_csv": str(traj_path),
            #     "checkpoint": str(ckpt),
            #     "objective_cost": objective_cost,
            #     "objective_cost_policy": float(df["objective_cost_policy"].iloc[0]),
            #     "total_cost_from_steps": total_cost_from_steps,
            #     "total_profit": total_profit,
            #     "n_steps": len(df),
            # })

        # seed_rows.append({
        #     "seed": seed,
        #     "checkpoint": str(ckpt),
        #     "n_test_days": len(day_costs),
        #     "mean_objective_cost": float(np.mean(day_costs)),
        #     "std_objective_cost_across_days": float(np.std(day_costs, ddof=1)) if len(day_costs) > 1 else 0.0,
        #     "mean_profit": float(np.mean(day_profits)),
        #     "std_profit_across_days": float(np.std(day_profits, ddof=1)) if len(day_profits) > 1 else 0.0,
        # })

    # seed_df = pd.DataFrame(seed_rows).sort_values("seed").reset_index(drop=True)
    # daily_df = pd.DataFrame(daily_rows).sort_values(["seed", "day_idx"]).reset_index(drop=True)
    # summary_df = pd.DataFrame([summarize_seed_scores(seed_df["mean_profit"].tolist())])

    # seed_df.to_csv(save_root / "seed_results.csv", index=False)
    # daily_df.to_csv(save_root / "daily_results.csv", index=False)
    # summary_df.to_csv(save_root / "summary.csv", index=False)

    # manifest = {
    #     "save_root": str(save_root),
    #     "seeds": list(seeds),
    #     "num_iterations": num_iterations,
    #     "obs_mode": obs_mode,
    #     "obs_window": obs_window,
    #     "n_train_days": len(train_days),
    #     "n_test_days": len(list(test_days)) if not isinstance(test_days, list) else len(test_days),
    #     "trajectory_dir": str(trajectories_dir),
    #     "files": {
    #         "seed_results": str(save_root / "seed_results.csv"),
    #         "daily_results": str(save_root / "daily_results.csv"),
    #         "summary": str(save_root / "summary.csv"),
    #     },
    # }
    # with open(save_root / "manifest.json", "w") as f:
    #     json.dump(manifest, f, indent=2)

    # print("\n===== SEED-LEVEL RESULTS =====")
    # print(seed_df.to_string(index=False))
    # print("\n===== SUMMARY ACROSS SEEDS =====")
    # print(summary_df.to_string(index=False))
    # print(f"\nDetailed per-day trajectory CSVs saved under: {trajectories_dir}")

    return seed_rows, daily_rows


if __name__ == "__main__":
    print(
        "Import run_multi_seed_experiment_detailed from this file and call it from your notebook."
    )
