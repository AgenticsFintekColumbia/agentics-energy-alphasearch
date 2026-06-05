# # =============================================================================
# # eoh_stress_plots.py  –  End-of-Horizon Stress Test
# #
# # Exposes late-horizon charging behaviour: methods that continue to charge
# # near the end of the decision window accumulate stranded energy, incur
# # terminal SOC inconsistencies, and require post-processing repairs.
# # =============================================================================

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.ticker as mticker
# from matplotlib.colors import LinearSegmentedColormap
# from matplotlib.gridspec import GridSpec
# import warnings
# warnings.filterwarnings("ignore")

# # ─────────────────────────────────────────────────────────────────────────────
# # 0.  FILE MAP  (edit paths to point at your local CSVs)
# # ─────────────────────────────────────────────────────────────────────────────
# FILE_MAP = {
#     "MILP":         "MILP_forecast_RF_one_year.csv",
#     "MILP(BC)":     "bias_corrected_milp_ITALY_2019.csv",
#     "MILP(Stoch)":  "bias_milp_stochastic_expected_results_ITALY_2019.csv",
#     r"MILP(CVaR$_{\beta=0.95,\lambda=0.5}$)": "cvar_milp_results_ITALY_2019.csv",
#     "HeurTime":     "HeurTime_forecast_RF_one_year.csv",
#     "HeurQuantile": "HeurQuantile_forecast_RF_one_year.csv",
#     "GPT-4":        "gpt4_hourly_arbitrage_corrected3.csv",
#     "RL-PPO":       "rlPPO_forecast_RF_one_year.csv",
# }
# METHODS = list(FILE_MAP.keys())

# SHORT_LABEL = {
#     "MILP":           "MILP(For.)",
#     "MILP(BC)":       "MILP\n(Bias-Corr.)",
#     "MILP(Stoch)":    "MILP\n(Stoch)",
#     r"MILP(CVaR$_{\beta=0.95,\lambda=0.5}$)": "MILP\n(CVaR)",
#     "HeurTime":       "HeurTime",
#     "HeurQuantile":   "HeurQtl",
#     "GPT-4":          "GPT-4",
#     "RL-PPO":         "RL-PPO",
# }

# CAPACITY_MWH = 10.78          # assumed battery capacity
# EOH_WINDOW   = 4              # hours before midnight classified as end-of-horizon
# LATE_CHARGE_HOURS = [18,19,20 ,21, 22, 23]  # hours considered "late charging"

# # ─────────────────────────────────────────────────────────────────────────────
# # 1.  DESIGN TOKENS
# # ─────────────────────────────────────────────────────────────────────────────
# METHOD_COLOR = {
#     "MILP":           "#2563EB",
#     "HeurTime":       "#16A34A",
#     "HeurQuantile":   "#9333EA",
#     "RL-PPO":         "#EA580C",
#     "GPT-4":          "#DC2626",
#     "MILP(BC)":       "#0EA5E9",
#     "MILP(Stoch)":    "#6366F1",
#     r"MILP(CVaR$_{\beta=0.95,\lambda=0.5}$)": "#0F766E",
# }
# METHOD_LS = {m: ("--" if "(BC)" in m else
#                  ("-." if "(Stoch)" in m else
#                   (":" if "CVaR" in m else "-")))
#              for m in METHODS}
# METHOD_MARKER = {
#     "MILP": "o", "HeurTime": "s", "HeurQuantile": "^",
#     "RL-PPO": "D", "GPT-4": "v",
#     "MILP(BC)": "P", "MILP(Stoch)": "X",
#     r"MILP(CVaR$_{\beta=0.95,\lambda=0.5}$)": "*",
# }

# plt.rcParams.update({
#     "font.family":        "serif",
#     "font.size":          10,
#     "axes.titlesize":     12,
#     "axes.labelsize":     10,
#     "axes.spines.top":    False,
#     "axes.spines.right":  False,
#     "axes.linewidth":     0.8,
#     "grid.color":         "#E5E7EB",
#     "grid.linewidth":     0.6,
#     "legend.frameon":     False,
#     "legend.fontsize":    9,
#     "xtick.major.size":   3,
#     "ytick.major.size":   3,
#     "figure.facecolor":   "white",
#     "axes.facecolor":     "white",
#     "savefig.facecolor":  "white",
#     "savefig.bbox":       "tight",
#     "savefig.dpi":        200,
# })

# def _fmt_gbp(v, _): return f"£{v:,.0f}"
# def _short(m):      return SHORT_LABEL.get(m, m)

# OUT = "eoh_figures"
# os.makedirs(OUT, exist_ok=True)

# # ─────────────────────────────────────────────────────────────────────────────
# # 2.  DATA LOADING & EOH FEATURE ENGINEERING
# # ─────────────────────────────────────────────────────────────────────────────
# def load_hourly(path, method):
#     df = pd.read_csv(path).copy()
#     n  = len(df)
#     df["day_index"] = df["day"].astype(int) if "day" in df.columns else np.arange(n) // 24
#     df["hour"]      = np.arange(n) % 24
#     df["method"]    = method
#     return df[["method", "day_index", "hour",
#                "soc", "charge_MW", "discharge_MW",
#                "profit_step", "prices_actual"]]

# print("Loading CSVs …")
# all_h = pd.concat(
#     [load_hourly(p, m) for m, p in FILE_MAP.items()],
#     ignore_index=True
# )
# print("  Loaded all 8 methods.\n")

# # ── Hourly SOC / charge profiles (mean across all days) ──────────────────────
# soc_profile    = all_h.groupby(["method","hour"])["soc"].mean().reset_index()
# charge_profile = all_h.groupby(["method","hour"])["charge_MW"].mean().reset_index()
# profit_profile = all_h.groupby(["method","hour"])["profit_step"].mean().reset_index()

# # ── Terminal SOC (hour 23) ────────────────────────────────────────────────────
# term_soc = (all_h[all_h["hour"] == 23]
#             .groupby(["method","day_index"])["soc"]
#             .last()
#             .reset_index()
#             .rename(columns={"soc": "terminal_soc"}))
# term_soc["stranded_MWh"] = term_soc["terminal_soc"] * CAPACITY_MWH

# last_price = (all_h[all_h["hour"] == 23]
#               [["method","day_index","prices_actual"]]
#               .rename(columns={"prices_actual": "last_price"}))
# term_soc = term_soc.merge(last_price, on=["method","day_index"])
# term_soc["repair_cost_gbp"] = term_soc["stranded_MWh"] * term_soc["last_price"]
# term_soc["is_stranded"]     = term_soc["terminal_soc"] > 0.01

# # ── Late-charging features (last EOH_WINDOW hours) ───────────────────────────
# eoh = all_h[all_h["hour"] >= (24 - EOH_WINDOW)].copy()
# eoh_feats = eoh.groupby(["method","day_index"]).agg(
#     eoh_charge_total =("charge_MW",    "sum"),
#     eoh_charge_max   =("charge_MW",    "max"),
#     eoh_profit_sum   =("profit_step",  "sum"),
#     eoh_price_mean   =("prices_actual","mean"),
# ).reset_index()
# eoh_feats = eoh_feats.merge(term_soc, on=["method","day_index"])

# # Late-charge flag (hours 22–23)
# late = all_h[all_h["hour"].isin(LATE_CHARGE_HOURS)].copy()
# late_charge = (late.groupby(["method","day_index"])["charge_MW"]
#                .sum().reset_index()
#                .rename(columns={"charge_MW": "late_charge_total"}))
# late_charge["is_late_charger"] = late_charge["late_charge_total"] > 0.01
# eoh_feats = eoh_feats.merge(late_charge, on=["method","day_index"])

# # ── Per-method summary ────────────────────────────────────────────────────────
# eoh_summary = eoh_feats.groupby("method").agg(
#     mean_terminal_soc  =("terminal_soc",     "mean"),
#     p90_terminal_soc   =("terminal_soc",     lambda x: np.quantile(x, 0.90)),
#     max_terminal_soc   =("terminal_soc",     "max"),
#     mean_stranded_MWh  =("stranded_MWh",     "mean"),
#     total_stranded_MWh =("stranded_MWh",     "sum"),
#     mean_repair_cost   =("repair_cost_gbp",  "mean"),
#     total_repair_cost  =("repair_cost_gbp",  "sum"),
#     late_charge_days   =("is_late_charger",  "sum"),
#     late_charge_freq   =("is_late_charger",  "mean"),
#     mean_eoh_charge    =("eoh_charge_total", "mean"),
#     mean_eoh_profit    =("eoh_profit_sum",   "mean"),
# ).reindex(METHODS).reset_index()

# print("EOH Summary:")
# print(eoh_summary[["method","mean_terminal_soc","mean_stranded_MWh",
#                     "mean_repair_cost","late_charge_freq"]].to_string(index=False))

# x8    = np.arange(len(METHODS))
# xlbls = [_short(m) for m in METHODS]

# # ─────────────────────────────────────────────────────────────────────────────
# # HELPER
# # ─────────────────────────────────────────────────────────────────────────────
# def _violin_box(ax, data_list, positions, colors, widths=0.55):
#     parts = ax.violinplot(data_list, positions=positions, widths=widths,
#                           showmeans=False, showmedians=False, showextrema=False)
#     for body, color in zip(parts["bodies"], colors):
#         body.set_facecolor(color); body.set_alpha(0.35); body.set_edgecolor("none")
#     bp = ax.boxplot(
#         data_list, positions=positions, widths=widths*0.28, patch_artist=True,
#         medianprops =dict(color="black", linewidth=1.8),
#         whiskerprops =dict(linewidth=0.9),
#         capprops     =dict(linewidth=0.9),
#         flierprops   =dict(marker=".", markersize=2, alpha=0.3, color="#6B7280"),
#     )
#     for patch, color in zip(bp["boxes"], colors):
#         patch.set_facecolor(color); patch.set_alpha(0.88)

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE E1 – Mean SOC & charge profiles (hours 0–23)
# #             Two panels: full-day trajectory + zoomed last 8 hours
# # ═════════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle("Intra-Day Battery State-of-Charge and Charging Profiles",
#              fontsize=14, fontweight="bold", y=1.01)

# for ax, (col, ylabel, title, zoom) in zip(axes, [
#         ("soc",       "Mean State of Charge",     "(a) Mean SOC Profile (full day)",          False),
#         ("charge_MW", "Mean Charge Rate (MW)",     "(b) Mean Charging Rate — Last 8 Hours",    True),
# ]):
#     df_plot = soc_profile if col == "soc" else charge_profile
#     hours   = range(16, 24) if zoom else range(0, 24)

#     for method in METHODS:
#         sub = df_plot[(df_plot["method"] == method) & (df_plot["hour"].isin(hours))]
#         lbl = _short(method).replace("\n", " ")
#         ax.plot(sub["hour"], sub[col],
#                 color=METHOD_COLOR[method], linewidth=1.8,
#                 linestyle=METHOD_LS[method], marker=METHOD_MARKER[method],
#                 markersize=4.5, label=lbl, zorder=3)

#     if zoom:
#         # shade the "danger zone" last 2 hours
#         ax.axvspan(21.5, 23.5, color="#FEE2E2", alpha=0.45, zorder=0,
#                    label="Late-charge window")
#         ax.set_xticks(range(16, 24))
#         ax.set_xticklabels([f"h{h}" for h in range(16, 24)])
#     else:
#         # shade last 4 hours
#         ax.axvspan(19.5, 23.5, color="#FEE2E2", alpha=0.30, zorder=0,
#                    label="EOH window (4 h)")
#         ax.set_xticks(range(0, 24, 2))

#     ax.set_xlabel("Hour of Day")
#     ax.set_ylabel(ylabel)
#     ax.set_title(title, fontsize=11, pad=8)
#     ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)

# axes[0].legend(fontsize=8, ncol=2, loc="upper right")
# axes[1].legend(fontsize=8, ncol=1, loc="upper left")

# plt.tight_layout()
# plt.savefig(f"{OUT}/figE1_soc_charge_profiles.png"); plt.close(); print("Fig E1 saved.")

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE E2 – Terminal SOC distribution (violin + box) per method
# # ═════════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle("Terminal State-of-Charge and Stranded Energy at End of Day",
#              fontsize=14, fontweight="bold", y=1.01)

# for ax, (col, ylabel, title) in zip(axes, [
#         ("terminal_soc",  "Terminal SOC (fraction)",    "(a) Terminal SOC Distribution"),
#         ("stranded_MWh",  "Stranded Energy (MWh)",      "(b) Stranded Energy Distribution"),
# ]):
#     data   = [eoh_feats[eoh_feats["method"]==m][col].values for m in METHODS]
#     colors = [METHOD_COLOR[m] for m in METHODS]
#     _violin_box(ax, data, x8, colors, widths=0.65)

#     ax.set_xticks(x8); ax.set_xticklabels(xlbls, fontsize=8.5)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title, fontsize=11, pad=8)
#     ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
#     ax.axhline(0, color="#9CA3AF", linewidth=0.7, linestyle="--")

#     # Annotate mean values above each violin
#     for xi, method in enumerate(METHODS):
#         mean_val = eoh_feats[eoh_feats["method"]==method][col].mean()
#         ax.text(xi, ax.get_ylim()[1]*0.97, f"{mean_val:.3f}",
#                 ha="center", va="top", fontsize=7.5, color="#374151", fontweight="bold")

# plt.tight_layout()
# plt.savefig(f"{OUT}/figE2_terminal_soc_stranded.png"); plt.close(); print("Fig E2 saved.")

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE E3 – EOH summary bar chart: 4 metrics in 2×2 grid
# # ═════════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(2, 2, figsize=(14, 9))
# fig.suptitle("End-of-Horizon Stress Test: Summary Metrics Across Methods",
#              fontsize=14, fontweight="bold", y=1.01)

# colors_bar = [METHOD_COLOR[m] for m in METHODS]
# specs = [
#     ("mean_terminal_soc", "Mean Terminal SOC",         "(a) Mean Terminal SOC",              False),
#     ("late_charge_freq",  "Late-Charging Frequency",   "(b) Late-Charging Frequency (h22–23)", True),
#     ("mean_stranded_MWh", "Mean Stranded Energy (MWh)","(c) Mean Stranded Energy",            False),
#     ("mean_repair_cost",  "Mean Repair Cost (£)",       "(d) Mean Estimated Repair Cost",     False),
# ]
# for ax, (col, ylabel, title, is_pct) in zip(axes.flatten(), specs):
#     vals = eoh_summary.set_index("method").reindex(METHODS)[col].values
#     bars = ax.bar(x8, vals, color=colors_bar, alpha=0.88,
#                   edgecolor="white", linewidth=0.5, zorder=3)

#     # value labels on bars
#     for bar, val in zip(bars, vals):
#         fmt = f"{val:.1%}" if is_pct else (f"£{val:,.1f}" if "cost" in col.lower() else f"{val:.3f}")
#         ax.text(bar.get_x() + bar.get_width()/2,
#                 bar.get_height() + ax.get_ylim()[1]*0.01,
#                 fmt, ha="center", va="bottom", fontsize=7.5, fontweight="bold")

#     if is_pct:
#         ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
#     elif "cost" in col.lower():
#         ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_gbp))

#     ax.set_xticks(x8); ax.set_xticklabels(xlbls, fontsize=8.5)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title, fontsize=11, pad=8)
#     ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)

# plt.tight_layout()
# plt.savefig(f"{OUT}/figE3_eoh_summary_bars.png"); plt.close(); print("Fig E3 saved.")

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE E4 – Annotated heatmap: EOH metrics (methods × metrics)
# # ═════════════════════════════════════════════════════════════════════════════
# cmap_heat = LinearSegmentedColormap.from_list("eoh", ["#F0FDF4","#991B1B"])

# eoh_heat_cols = {
#     "mean_terminal_soc":  "Mean\nTerminal SOC",
#     "p90_terminal_soc":   "P90\nTerminal SOC",
#     "late_charge_freq":   "Late-Charge\nFrequency",
#     "mean_stranded_MWh":  "Mean Stranded\nEnergy (MWh)",
#     "mean_repair_cost":   "Mean Repair\nCost (£)",
# }

# heat_df = eoh_summary.set_index("method").reindex(METHODS)[list(eoh_heat_cols.keys())]
# # normalise each column to [0,1] for colour mapping
# heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-9)

# fig, ax = plt.subplots(figsize=(12, 5.5))
# im = ax.imshow(heat_norm.values, cmap=cmap_heat, aspect="auto",
#                vmin=0, vmax=1)

# ax.set_xticks(range(len(eoh_heat_cols)))
# ax.set_xticklabels(list(eoh_heat_cols.values()), fontsize=10)
# ax.set_yticks(range(len(METHODS)))
# ax.set_yticklabels(xlbls, fontsize=9)
# ax.tick_params(length=0)

# for i in range(len(METHODS)):
#     for j, col in enumerate(eoh_heat_cols):
#         raw = heat_df.iloc[i, j]
#         norm_val = heat_norm.iloc[i, j]
#         txt_color = "white" if norm_val > 0.55 else "#1E293B"
#         # format by column type
#         if "freq" in col:
#             txt = f"{raw:.1%}"
#         elif "cost" in col:
#             txt = f"£{raw:,.1f}"
#         elif "MWh" in col:
#             txt = f"{raw:.3f}"
#         else:
#             txt = f"{raw:.4f}"
#         ax.text(j, i, txt, ha="center", va="center",
#                 fontsize=9, color=txt_color, fontweight="bold")

# cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="Normalised severity (higher = worse)")
# cb.ax.tick_params(labelsize=8)
# ax.set_title("End-of-Horizon Stress Severity Heatmap",
#              fontsize=13, fontweight="bold", pad=10)

# plt.tight_layout()
# plt.savefig(f"{OUT}/figE4_eoh_heatmap.png"); plt.close(); print("Fig E4 saved.")

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE E5 – Scatter: EOH charge total vs daily profit (late-chargers flagged)
# # ═════════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle("Late-Horizon Charging Behaviour vs. Daily Profit",
#              fontsize=14, fontweight="bold", y=1.01)

# for ax, (method_subset, title) in zip(axes, [
#         (["HeurQuantile","HeurTime","RL-PPO","GPT-4"],
#          "(a) Heuristic & RL Methods"),
#         (["MILP","MILP(BC)","MILP(Stoch)",
#           r"MILP(CVaR$_{\beta=0.95,\lambda=0.5}$)"],
#          "(b) MILP Family"),
# ]):
#     for method in method_subset:
#         sub = eoh_feats[eoh_feats["method"]==method]
#         lbl = _short(method).replace("\n"," ")

#         # Non-late-charge days
#         sub_ok = sub[~sub["is_late_charger"]]
#         ax.scatter(sub_ok["eoh_charge_total"], sub_ok["eoh_profit_sum"],
#                    color=METHOD_COLOR[method], alpha=0.45, s=18,
#                    marker=METHOD_MARKER[method], linewidths=0, zorder=3, label=lbl)

#         # Late-charge days (red ring)
#         sub_late = sub[sub["is_late_charger"]]
#         if len(sub_late):
#             ax.scatter(sub_late["eoh_charge_total"], sub_late["eoh_profit_sum"],
#                        facecolors="none", edgecolors=METHOD_COLOR[method],
#                        s=60, marker="o", linewidths=1.5, zorder=4)

#     ax.set_xlabel(f"Total EOH Charge (MW, last {EOH_WINDOW} h)")
#     ax.set_ylabel("EOH Profit (£)")
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_gbp))
#     ax.set_title(title, fontsize=11, pad=8)
#     ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
#     ax.axhline(0, color="#9CA3AF", linewidth=0.8, linestyle="--")
#     ax.legend(fontsize=8.5, ncol=1)

# # shared annotation
# fig.text(0.5, -0.02, "Circled points = late-charging days (charge > 0 in hours 22–23)",
#          ha="center", fontsize=9, color="#6B7280", style="italic")
# plt.tight_layout()
# plt.savefig(f"{OUT}/figE5_eoh_charge_vs_profit.png"); plt.close(); print("Fig E5 saved.")

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE E6 – Repair cost distribution + cumulative repair cost over days
# # ═════════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle("Estimated Terminal Repair Cost: Distribution and Cumulative Burden",
#              fontsize=14, fontweight="bold", y=1.01)

# # Left: violin + box of daily repair cost
# data   = [eoh_feats[eoh_feats["method"]==m]["repair_cost_gbp"].values for m in METHODS]
# colors = [METHOD_COLOR[m] for m in METHODS]
# _violin_box(axes[0], data, x8, colors, widths=0.65)
# axes[0].set_xticks(x8); axes[0].set_xticklabels(xlbls, fontsize=8.5)
# axes[0].set_ylabel("Repair Cost per Day (£)")
# axes[0].yaxis.set_major_formatter(plt.FuncFormatter(_fmt_gbp))
# axes[0].set_title("(a) Daily Repair Cost Distribution", fontsize=11, pad=8)
# axes[0].yaxis.grid(True, zorder=0); axes[0].set_axisbelow(True)

# # annotate total repair cost
# for xi, method in enumerate(METHODS):
#     total = eoh_feats[eoh_feats["method"]==method]["repair_cost_gbp"].sum()
#     axes[0].text(xi, axes[0].get_ylim()[1]*0.96,
#                  f"Σ£{total:,.0f}", ha="center", va="top",
#                  fontsize=7, color="#374151", fontweight="bold")

# # Right: cumulative repair cost over the year
# for method in METHODS:
#     sub = eoh_feats[eoh_feats["method"]==method].sort_values("day_index")
#     cum = sub["repair_cost_gbp"].cumsum().values
#     axes[1].plot(np.arange(len(cum)), cum,
#                  color=METHOD_COLOR[method], linewidth=1.8,
#                  linestyle=METHOD_LS[method],
#                  label=_short(method).replace("\n"," "))

# axes[1].set_xlabel("Day of Year")
# axes[1].set_ylabel("Cumulative Repair Cost (£)")
# axes[1].yaxis.set_major_formatter(plt.FuncFormatter(_fmt_gbp))
# axes[1].set_title("(b) Cumulative Repair Cost Over Time", fontsize=11, pad=8)
# axes[1].yaxis.grid(True, zorder=0); axes[1].set_axisbelow(True)
# axes[1].legend(fontsize=8.5, ncol=2, loc="upper left")

# plt.tight_layout()
# plt.savefig(f"{OUT}/figE6_repair_cost.png"); plt.close(); print("Fig E6 saved.")

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE E7 – EOH profit profile (mean hourly profit, hours 18–23)
# #             + net-of-repair-cost comparison
# # ═════════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle("End-of-Horizon Profit Profile and Net-of-Repair Performance",
#              fontsize=14, fontweight="bold", y=1.01)

# # Left: mean hourly profit for hours 18–23
# eoh_hours = list(range(18, 24))
# for method in METHODS:
#     sub = profit_profile[(profit_profile["method"]==method) &
#                           (profit_profile["hour"].isin(eoh_hours))]
#     axes[0].plot(sub["hour"], sub["profit_step"],
#                  color=METHOD_COLOR[method], linewidth=1.8,
#                  linestyle=METHOD_LS[method], marker=METHOD_MARKER[method],
#                  markersize=5, label=_short(method).replace("\n"," "), zorder=3)

# axes[0].axvspan(21.5, 23.5, color="#FEE2E2", alpha=0.40, zorder=0, label="Late-charge window")
# axes[0].axhline(0, color="#9CA3AF", linewidth=0.8, linestyle="--")
# axes[0].set_xticks(eoh_hours)
# axes[0].set_xticklabels([f"h{h}" for h in eoh_hours])
# axes[0].set_xlabel("Hour of Day")
# axes[0].set_ylabel("Mean Hourly Profit (£)")
# axes[0].yaxis.set_major_formatter(plt.FuncFormatter(_fmt_gbp))
# axes[0].set_title("(a) Mean Profit Profile — Last 6 Hours", fontsize=11, pad=8)
# axes[0].yaxis.grid(True, zorder=0); axes[0].set_axisbelow(True)
# axes[0].legend(fontsize=8, ncol=2)

# # Right: total annual profit vs net-of-repair-cost (side by side bars)
# daily_profit_total = eoh_feats.groupby("method")["eoh_profit_sum"].sum().reindex(METHODS)
# total_repair       = eoh_feats.groupby("method")["repair_cost_gbp"].sum().reindex(METHODS)

# # Use full-year day profit from all_h
# full_day_profit = (all_h.groupby(["method","day_index"])["profit_step"]
#                    .sum().groupby("method").sum().reindex(METHODS))
# net_profit = full_day_profit - total_repair

# w2 = 0.35
# axes[1].bar(x8 - w2/2, full_day_profit.values / 1000, w2,
#             color=[METHOD_COLOR[m] for m in METHODS], alpha=0.88,
#             edgecolor="white", label="Gross Profit", zorder=3)
# axes[1].bar(x8 + w2/2, net_profit.values / 1000, w2,
#             color=[METHOD_COLOR[m] for m in METHODS], alpha=0.50,
#             edgecolor=[ METHOD_COLOR[m] for m in METHODS],
#             linewidth=1.2, hatch="///", label="Net of Repair Cost", zorder=3)

# axes[1].set_xticks(x8); axes[1].set_xticklabels(xlbls, fontsize=8.5)
# axes[1].set_ylabel("Annual Profit (£ thousands)")
# axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"£{v:,.0f}k"))
# axes[1].set_title("(b) Gross vs Net-of-Repair Annual Profit", fontsize=11, pad=8)
# axes[1].yaxis.grid(True, zorder=0); axes[1].set_axisbelow(True)

# # legend once
# handles = [
#     mpatches.Patch(facecolor="#6B7280", alpha=0.88, label="Gross Profit"),
#     mpatches.Patch(facecolor="#6B7280", alpha=0.50, hatch="///", label="Net of Repair Cost"),
# ]
# axes[1].legend(handles=handles, fontsize=9)

# plt.tight_layout()
# plt.savefig(f"{OUT}/figE7_eoh_profit_and_net.png"); plt.close(); print("Fig E7 saved.")

# # ─────────────────────────────────────────────────────────────────────────────
# # Save summary CSVs
# # ─────────────────────────────────────────────────────────────────────────────
# eoh_summary.to_csv(f"{OUT}/eoh_summary.csv", index=False)
# eoh_feats.to_csv(f"{OUT}/eoh_daily_features.csv", index=False)
# print(f"\nAll EOH figures + CSVs written to ./{OUT}/")

# =============================================================================
# eoh_stress_plots_minimal.py
# Keep only:
# (1) Mean Charging Rate — Last 9 Hours
# (2) Late-Charging Frequency
# (3) Terminal SoC Distribution
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. FILE MAP
# ─────────────────────────────────────────────────────────────────────────────
FILE_MAP = {
    "MILP":         "MILP_forecast_RF_one_year.csv",
    "MILP(BC)":     "bias_corrected_milp_ITALY_2019.csv",
    "MILP(Stoch)":  "bias_milp_stochastic_expected_results_ITALY_2019.csv",
    r"MILP(CVaR$_{\beta=0.95,\lambda=0.5}$)": "cvar_milp_results_ITALY_2019.csv",
    "HeurTime":     "HeurTime_forecast_RF_one_year.csv",
    "HeurQuantile": "HeurQuantile_forecast_RF_one_year.csv",
    "GPT-4":        "gpt4_hourly_arbitrage_corrected3.csv",
    "RL-PPO":       "rlPPO_forecast_RF_one_year.csv",
}
METHODS = list(FILE_MAP.keys())

SHORT_LABEL = {
    "MILP":           "MILP(For.)",
    "MILP(BC)":       "MILP\n(Bias-Corr.)",
    "MILP(Stoch)":    "MILP\n(Stoch)",
    r"MILP(CVaR$_{\beta=0.95,\lambda=0.5}$)": "MILP\n(CVaR)",
    "HeurTime":       "HeurTime",
    "HeurQuantile":   "HeurQtl",
    "GPT-4":          "GPT-4",
    "RL-PPO":         "RL-PPO",
}

CAPACITY_MWH = 10.78
LATE_CHARGE_HOURS = [22, 23]   # keep this strict if you want true late charging
OUT = "eoh_figures_minimal"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. STYLE
# ─────────────────────────────────────────────────────────────────────────────
METHOD_COLOR = {
    "MILP":           "#2563EB",
    "HeurTime":       "#16A34A",
    "HeurQuantile":   "#9333EA",
    "RL-PPO":         "#EA580C",
    "GPT-4":          "#DC2626",
    "MILP(BC)":       "#0EA5E9",
    "MILP(Stoch)":    "#6366F1",
    r"MILP(CVaR$_{\beta=0.95,\lambda=0.5}$)": "#0F766E",
}
METHOD_LS = {
    m: (
        "--" if "(BC)" in m else
        "-." if "(Stoch)" in m else
        ":" if "CVaR" in m else
        "-"
    )
    for m in METHODS
}
METHOD_MARKER = {
    "MILP": "o",
    "HeurTime": "s",
    "HeurQuantile": "^",
    "RL-PPO": "D",
    "GPT-4": "v",
    "MILP(BC)": "P",
    "MILP(Stoch)": "X",
    r"MILP(CVaR$_{\beta=0.95,\lambda=0.5}$)": "*",
}
# plt.rcParams.update({
#     "font.size": 16,          # base font size
#     "axes.labelsize": 18,     # x/y label size
#     "axes.titlesize": 20,     # title size
#     "xtick.labelsize": 14,    # tick size
#     "ytick.labelsize": 14,
#     "legend.fontsize": 16,    # legend text size
#     "font.weight": "bold",    # make all text bold
#     "axes.labelweight": "bold",
#     "axes.titleweight": "bold",
# })

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 16,
    "xtick.major.size": 14,
    "ytick.major.size": 14,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "font.weight": "bold",    # make all text bold
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

def _short(m):
    return SHORT_LABEL.get(m, m)

def _violin_box(ax, data_list, positions, colors, widths=0.55):
    parts = ax.violinplot(
        data_list,
        positions=positions,
        widths=widths,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor("none")

    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=widths * 0.28,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=0.9),
        capprops=dict(linewidth=0.9),
        flierprops=dict(marker=".", markersize=2, alpha=0.3, color="#6B7280"),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_hourly(path, method):
    df = pd.read_csv(path).copy()
    n = len(df)
    df["day_index"] = df["day"].astype(int) if "day" in df.columns else np.arange(n) // 24
    df["hour"] = np.arange(n) % 24
    df["method"] = method
    return df[[
        "method", "day_index", "hour",
        "soc", "charge_MW", "discharge_MW",
        "profit_step", "prices_actual"
    ]]

print("Loading CSVs ...")
all_h = pd.concat(
    [load_hourly(path, method) for method, path in FILE_MAP.items()],
    ignore_index=True
)
print("Loaded all methods.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
charge_profile = all_h.groupby(["method", "hour"])["charge_MW"].mean().reset_index()

term_soc = (
    all_h[all_h["hour"] == 23]
    .groupby(["method", "day_index"])["soc"]
    .last()
    .reset_index()
    .rename(columns={"soc": "terminal_soc"})
)

late = all_h[all_h["hour"].isin(LATE_CHARGE_HOURS)].copy()
late_charge = (
    late.groupby(["method", "day_index"])["charge_MW"]
    .sum()
    .reset_index()
    .rename(columns={"charge_MW": "late_charge_total"})
)
late_charge["is_late_charger"] = late_charge["late_charge_total"] > 0.01

late_summary = (
    late_charge.groupby("method")
    .agg(late_charge_freq=("is_late_charger", "mean"))
    .reindex(METHODS)
    .reset_index()
)

x8 = np.arange(len(METHODS))
xlbls = [_short(m) for m in METHODS]

# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT 1 — Terminal SoC Distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

data = [term_soc[term_soc["method"] == m]["terminal_soc"].values for m in METHODS]
colors = [METHOD_COLOR[m] for m in METHODS]
_violin_box(ax, data, x8, colors, widths=0.65)

ax.set_xticks(x8)
ax.set_xticklabels(xlbls, fontsize=8.5)
ax.set_ylabel("Terminal SOC (fraction)")
ax.set_title("Terminal SOC Distribution", fontsize=11, pad=8)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.axhline(0, color="#9CA3AF", linewidth=0.7, linestyle="--")

for xi, method in enumerate(METHODS):
    mean_val = term_soc[term_soc["method"] == method]["terminal_soc"].mean()
    ax.text(
        xi,
        ax.get_ylim()[1] * 0.97,
        f"{mean_val:.3f}",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#374151",
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig(f"{OUT}/terminal_soc_distribution.png")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT 2 — Mean Charging Rate — Last 9 Hours
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

hours = range(15, 24)   # last 9 hours

for method in METHODS:
    sub = charge_profile[
        (charge_profile["method"] == method) &
        (charge_profile["hour"].isin(hours))
    ]
    ax.plot(
        sub["hour"],
        sub["charge_MW"],
        color=METHOD_COLOR[method],
        linewidth=1.8,
        linestyle=METHOD_LS[method],
        marker=METHOD_MARKER[method],
        markersize=4.5,
        label=_short(method).replace("\n", " "),
        zorder=3,
    )

ax.axvspan(21.5, 23.5, color="#FEE2E2", alpha=0.45, zorder=0, label="Late-charge window")
ax.set_xticks(range(15, 24))
ax.set_xticklabels([f"h{h}" for h in range(15, 24)])
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Mean Charge Rate (MW)")
ax.set_title("Mean Charging Rate — Last 9 Hours", fontsize=11, pad=8)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.legend(fontsize=8, ncol=2, loc="upper left")

plt.tight_layout()
plt.savefig(f"{OUT}/mean_charging_last_9_hours.png")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOT 3 — Late-Charging Frequency
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

vals = late_summary.set_index("method").reindex(METHODS)["late_charge_freq"].values
bars = ax.bar(x8, vals, color=[METHOD_COLOR[m] for m in METHODS], alpha=0.88, edgecolor="white")

for bar, val in zip(bars, vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{val:.1%}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold"
    )

ax.set_xticks(x8)
ax.set_xticklabels(xlbls, fontsize=8.5)
ax.set_ylabel("Late-Charging Frequency")
ax.set_title("Late-Charging Frequency (h22–23)", fontsize=11, pad=8)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.yaxis.grid(True)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{OUT}/late_charging_frequency.png")
plt.show()

print(f"\nSaved only these 3 plots to ./{OUT}/")