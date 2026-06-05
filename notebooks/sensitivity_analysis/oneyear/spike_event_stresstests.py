# # =============================================================================
# # stress_plots.py  –  Publication-quality stress-event figures
# #                     Works with all 8 methods (5 original + 3 MILP variants)
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
# # 0.  FILE MAP
# # ─────────────────────────────────────────────────────────────────────────────
# START_DATE  = "2023-01-01"


# OUT = "spike_event_figures"
# os.makedirs(OUT, exist_ok=True)

# # FILE_MAP = {
# #     "MILP":         "MILP_forecast_RF_one_year.csv",
# #     "MILP(BC)":     "bias_corrected_milp_ITALY_2019.csv",
# #     "MILP(Stoch)":  "bias_milp_stochastic_expected_results_ITALY_2019.csv",
# #     r"MILP(CVaR$_{\beta=0.95,\lambda=0.5}$)": "cvar_milp_results_ITALY_2019.csv",
# #     "HeurTime":     "HeurTime_forecast_RF_one_year.csv",
# #     "HeurQuantile": "HeurQuantile_forecast_RF_one_year.csv",
# #     "GPT-4":        "gpt4_hourly_arbitrage_corrected3.csv",
# #     "RL-PPO":       "rlPPO_forecast_RF_one_year.csv",
# # }

# FILE_MAP = {
#     "MILP":         "MILP_forecast_RF_one_year_NEWYORK_2023.csv",
#     "MILP(BC)":     "bias_corrected_milp_NEWYORK_2023.csv",
#     "MILP(Stoch)":  "bias_milp_stochastic_expected_results_NEWYORK_2023.csv",
#     r"MILP(CVaR$_{\beta=0.95,\lambda=0.5}$)": "cvar_milp_results_NEWYORK_2023.csv",
#     "HeurTime":     "HeurTime_forecast_RF_one_year_NEWYORK_2023.csv",
#     "HeurQuantile": "HeurQuantile_forecast_RF_one_year_NEWYORK_2023.csv",
#     "GPT-4":        "gpt4_hourly_arbitrage_new_york_2023_corrected.csv",
#     "RL-PPO":       "rlPPO_forecast_RF_one_year_NEWYORK_2023.csv",
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

# STRESS_BASE = "MILP"        # method whose prices define stress thresholds

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
# METHOD_LS = {m: ("--" if "(BC)" in m else ("-." if "(Stoch)" in m else (":" if "CVaR" in m else "-")))
#              for m in METHODS}

# REGIME_ORDER  = ["Normal", "Spike", "High Vol", "Spike + High Vol"]
# REGIME_COLOR  = {
#     "Normal":           "#6B7280",
#     "Spike":            "#EF4444",
#     "High Vol":         "#F59E0B",
#     "Spike + High Vol": "#7C3AED",
# }
# REGIME_HATCH = {
#     "Normal":           "",
#     "Spike":            "",
#     "High Vol":         "///",
#     "Spike + High Vol": "xxx",
# }

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 16,
#     "axes.titlesize": 20,
#     "axes.labelsize": 18,
#     "axes.spines.top": False,
#     "axes.spines.right": False,
#     "axes.linewidth": 0.8,
#     "grid.color": "#E5E7EB",
#     "grid.linewidth": 0.6,
#     "legend.frameon": False,
#     "legend.fontsize": 16,
#     "xtick.major.size": 14,
#     "ytick.major.size": 14,
#     "figure.facecolor": "white",
#     "axes.facecolor": "white",
#     "savefig.facecolor": "white",
#     "savefig.bbox": "tight",
#     "savefig.dpi": 200,
#     "font.weight": "bold",    # make all text bold
#     "axes.labelweight": "bold",
#     "axes.titleweight": "bold",
# })

# # plt.rcParams.update({
# #     "font.family":        "serif",
# #     "font.size":          10,
# #     "axes.titlesize":     12,
# #     "axes.labelsize":     10,
# #     "axes.spines.top":    False,
# #     "axes.spines.right":  False,
# #     "axes.linewidth":     0.8,
# #     "grid.color":         "#E5E7EB",
# #     "grid.linewidth":     0.6,
# #     "legend.frameon":     False,
# #     "legend.fontsize":    9,
# #     "xtick.major.size":   3,
# #     "ytick.major.size":   3,
# #     "figure.facecolor":   "white",
# #     "axes.facecolor":     "white",
# #     "savefig.facecolor":  "white",
# #     "savefig.bbox":       "tight",
# #     "savefig.dpi":        200,
# # })

# def _fmt_gbp(v, _): return f"£{v:,.0f}"
# def _short(m): return SHORT_LABEL.get(m, m)


# # ─────────────────────────────────────────────────────────────────────────────
# # 2.  DATA PIPELINE
# # ─────────────────────────────────────────────────────────────────────────────
# def month_to_season(m):
#     if m in [12,1,2]: return "Winter"
#     if m in [3,4,5]:  return "Spring"
#     if m in [6,7,8]:  return "Summer"
#     return "Autumn"

# def prepare_result_df(path, method_name, start_date=START_DATE):
#     df = pd.read_csv(path).copy()
#     n_rows = len(df)
#     n_days = n_rows // 24
#     df["day_index"] = df["day"].astype(int) if "day" in df.columns else np.arange(n_rows)//24
#     df["hour"]      = np.arange(n_rows) % 24
#     dr = pd.date_range(start=start_date, periods=n_days, freq="D")
#     df["date"]        = df["day_index"].map({i: dr[i] for i in range(n_days)})
#     df["is_weekend"]  = df["date"].dt.weekday >= 5
#     df["regime_week"] = np.where(df["is_weekend"], "Weekend", "Weekday")
#     df["season"]      = df["date"].dt.month.apply(month_to_season)
#     df["method"]      = method_name
#     return df

# def make_daily(hourly_df):
#     return hourly_df.groupby(
#         ["method","day_index","date","regime_week","season"], as_index=False
#     ).agg(day_profit=("profit_step","sum"))

# print("Loading CSVs …")
# all_hourly, all_daily = [], []
# for m, p in FILE_MAP.items():
#     h = prepare_result_df(p, m)
#     all_hourly.append(h)
#     all_daily.append(make_daily(h))
#     print(f"  {m}: loaded")

# hourly_all = pd.concat(all_hourly, ignore_index=True)
# daily_all  = pd.concat(all_daily,  ignore_index=True)

# # ── stress features from base method's actual prices ─────────────────────────
# base_h = hourly_all[hourly_all["method"] == STRESS_BASE].copy()
# stress_feat = base_h.groupby(["day_index","date"], as_index=False).agg(
#     price_max  =("prices_actual","max"),
#     price_min  =("prices_actual","min"),
#     price_mean =("prices_actual","mean"),
#     price_std  =("prices_actual","std"),
#     price_range=("prices_actual", lambda x: x.max()-x.min()),
#     abs_change_mean=("prices_actual",
#                      lambda x: np.mean(np.abs(np.diff(x.values))) if len(x)>1 else 0.0),
#     abs_change_max =("prices_actual",
#                      lambda x: np.max(np.abs(np.diff(x.values)))  if len(x)>1 else 0.0),
# )

# thr_spike = stress_feat["price_max"].quantile(0.95)
# thr_vol   = stress_feat["price_std"].quantile(0.90)

# stress_feat["is_spike"]    = stress_feat["price_max"] >= thr_spike
# stress_feat["is_high_vol"] = stress_feat["price_std"] >= thr_vol
# stress_feat["stress_regime"] = "Normal"
# stress_feat.loc[stress_feat["is_spike"],    "stress_regime"] = "Spike"
# stress_feat.loc[stress_feat["is_high_vol"], "stress_regime"] = "High Vol"
# stress_feat.loc[stress_feat["is_spike"] & stress_feat["is_high_vol"],
#                 "stress_regime"] = "Spike + High Vol"

# dws = daily_all.merge(
#     stress_feat[["day_index","date","is_spike","is_high_vol","stress_regime",
#                  "price_max","price_std","price_range","abs_change_mean"]],
#     on=["day_index","date"], how="left"
# )

# # Summary tables
# def _summary(df, groupcol):
#     return df.groupby(["method", groupcol], as_index=False).agg(
#         n_days           =("day_profit","count"),
#         mean_day_profit  =("day_profit","mean"),
#         std_day_profit   =("day_profit","std"),
#         median_day_profit=("day_profit","median"),
#         p5_day_profit    =("day_profit", lambda x: np.quantile(x, 0.05)),
#         total_profit     =("day_profit","sum"),
#     )

# stress_summary = _summary(dws, "stress_regime")
# spike_summary  = stress_summary[stress_summary["stress_regime"]=="Spike"]
# hv_summary     = stress_summary[stress_summary["stress_regime"]=="High Vol"]

# print(f"\nStress thresholds: spike price_max ≥ {thr_spike:.2f}, "
#       f"high-vol price_std ≥ {thr_vol:.2f}")
# print("Data ready.\n")

# xlbls = [_short(m) for m in METHODS]
# x8    = np.arange(len(METHODS))

# # ─────────────────────────────────────────────────────────────────────────────
# # HELPER: violin + boxplot
# # ─────────────────────────────────────────────────────────────────────────────
# def _violin_box(ax, data_list, positions, colors, widths=0.55):
#     parts = ax.violinplot(data_list, positions=positions, widths=widths,
#                           showmeans=False, showmedians=False, showextrema=False)
#     for body, color in zip(parts["bodies"], colors):
#         body.set_facecolor(color); body.set_alpha(0.35); body.set_edgecolor("none")
#     bp = ax.boxplot(
#         data_list, positions=positions, widths=widths*0.27, patch_artist=True,
#         medianprops =dict(color="black", linewidth=1.8),
#         whiskerprops =dict(linewidth=0.9),
#         capprops     =dict(linewidth=0.9),
#         flierprops   =dict(marker=".", markersize=2, alpha=0.3, color="#6B7280"),
#     )
#     for patch, color in zip(bp["boxes"], colors):
#         patch.set_facecolor(color); patch.set_alpha(0.88)

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE S1 – Price distribution on stress days (context figure)
# # ═════════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
# fig.suptitle("Daily Price Characteristics by Stress Regime",
#              fontsize=14, fontweight="bold", y=1.01)

# metrics = [
#     ("price_max",        "Daily Max Price (£/MWh)", "(a) Max Price"),
#     ("price_std",        "Price Std Dev (£/MWh)",   "(b) Price Volatility"),
#     ("abs_change_mean",  "Mean |ΔPrice| (£/MWh)",   "(c) Mean Hourly Price Change"),
# ]
# reg_order_plot = [r for r in REGIME_ORDER if r in stress_feat["stress_regime"].unique()]

# for ax, (col, ylabel, title) in zip(axes, metrics):
#     data   = [stress_feat[stress_feat["stress_regime"]==r][col].values for r in reg_order_plot]
#     colors = [REGIME_COLOR[r] for r in reg_order_plot]
#     _violin_box(ax, data, np.arange(len(reg_order_plot)), colors, widths=0.5)
#     ax.set_xticks(np.arange(len(reg_order_plot)))
#     ax.set_xticklabels(reg_order_plot, fontsize=9)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title, fontsize=11, pad=8)
#     ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)

# # Add threshold annotations on first two panels
# axes[0].axhline(thr_spike, color=REGIME_COLOR["Spike"], linewidth=1.2,
#                 linestyle="--", label=f"q95 = £{thr_spike:.1f}")
# axes[1].axhline(thr_vol,   color=REGIME_COLOR["High Vol"], linewidth=1.2,
#                 linestyle="--", label=f"q90 = {thr_vol:.2f}")
# for ax in axes[:2]:
#     ax.legend(fontsize=8, handlelength=1.5)

# # Regime count annotation
# for ax in axes:
#     for i, r in enumerate(reg_order_plot):
#         n = (stress_feat["stress_regime"]==r).sum()
#         ax.text(i, ax.get_ylim()[0], f"n={n}", ha="center", va="bottom",
#                 fontsize=7.5, color="#374151")

# plt.tight_layout()
# plt.savefig(f"{OUT}/figS1_stress_price_dist.png"); plt.close(); print("Fig S1 saved.")

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE S2 – Mean daily profit by stress regime (grouped bars)
# # ═════════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(1, 2, figsize=(15, 5))
# fig.suptitle("Mean Daily Profit and P5 Downside by Stress Regime",
#              fontsize=14, fontweight="bold", y=1.01)

# specs = [
#     ("mean_day_profit", "Mean Daily Profit (£)", "(a) Mean Daily Profit"),
#     ("p5_day_profit",   "P5 Daily Profit (£)",   "(b) P5 Downside Risk"),
# ]
# regimes_present = [r for r in REGIME_ORDER
#                    if r in stress_summary["stress_regime"].unique()]
# rw   = 0.18
# roff = np.linspace(-(len(regimes_present)-1)/2,
#                     (len(regimes_present)-1)/2, len(regimes_present)) * rw

# for ax, (metric, ylabel, title) in zip(axes, specs):
#     pivot = stress_summary.pivot(index="method", columns="stress_regime",
#                                  values=metric).reindex(METHODS)[regimes_present]
#     for offset, regime in zip(roff, regimes_present):
#         bars = ax.bar(x8 + offset, pivot[regime], rw,
#                       color=REGIME_COLOR[regime], alpha=0.88, label=regime,
#                       edgecolor="white", linewidth=0.4, zorder=3,
#                       hatch=REGIME_HATCH[regime])

#     # error bars for mean
#     if metric == "mean_day_profit":
#         std_piv = stress_summary.pivot(index="method", columns="stress_regime",
#                                        values="std_day_profit").reindex(METHODS)[regimes_present]
#         for offset, regime in zip(roff, regimes_present):
#             ax.errorbar(x8 + offset, pivot[regime], yerr=std_piv[regime],
#                         fmt="none", color="black", capsize=2.5, linewidth=0.7, zorder=4)

#     ax.set_xticks(x8); ax.set_xticklabels(xlbls, fontsize=8.5)
#     ax.set_ylabel(ylabel)
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_gbp))
#     ax.set_title(title, fontsize=11, pad=8)
#     ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
#     ax.axhline(0, color="#9CA3AF", linewidth=0.7, linestyle="--")
#     ax.legend(title="Stress Regime", title_fontsize=9, ncol=2)

# plt.tight_layout()
# plt.savefig(f"{OUT}/figS2_stress_regime_bars.png"); plt.close(); print("Fig S2 saved.")

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE S3 – Annotated heatmaps: mean & P5 profit by stress regime
# # ═════════════════════════════════════════════════════════════════════════════
# cmap_red  = LinearSegmentedColormap.from_list("profit_r", ["#FEF2F2","#991B1B"])
# cmap_blue = LinearSegmentedColormap.from_list("profit_b", ["#EFF6FF","#1D4ED8"])

# fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
# fig.suptitle("Stress Regime Profit Heatmaps", fontsize=14, fontweight="bold", y=1.01)

# for ax, (metric, title, cmap) in zip(axes, [
#         ("mean_day_profit", "(a) Mean Daily Profit (£)", cmap_blue),
#         ("p5_day_profit",   "(b) P5 Daily Profit (£)",   cmap_red)]):

#     pivot = stress_summary.pivot(index="method", columns="stress_regime",
#                                  values=metric).reindex(METHODS)[regimes_present]
#     vmin, vmax = pivot.values.min(), pivot.values.max()
#     im = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
#                    vmin=vmin*0.9, vmax=vmax*1.05)

#     ax.set_xticks(range(len(regimes_present)))
#     ax.set_xticklabels(regimes_present, fontsize=9.5)
#     ax.set_yticks(range(len(METHODS)))
#     ax.set_yticklabels(xlbls, fontsize=9)
#     ax.tick_params(length=0)

#     for i in range(len(METHODS)):
#         for j in range(len(regimes_present)):
#             val = pivot.iloc[i, j]
#             brightness = (val - vmin) / (vmax - vmin + 1e-9)
#             txt_col = "white" if brightness > 0.52 else "#1E293B"
#             ax.text(j, i, f"£{val:,.0f}", ha="center", va="center",
#                     fontsize=8.5, color=txt_col, fontweight="bold")

#     cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
#     cb.ax.tick_params(labelsize=8)
#     ax.set_title(title, fontsize=11, pad=8)

# plt.tight_layout()
# plt.savefig(f"{OUT}/figS3_stress_heatmap.png"); plt.close(); print("Fig S3 saved.")

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE S4 – Violin + boxplot: daily profit distribution per stress regime
# #             (all methods pooled)
# # ═════════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(1, len(regimes_present), figsize=(14, 5), sharey=True)
# fig.suptitle("Daily Profit Distribution by Stress Regime (All Methods Pooled)",
#              fontsize=13, fontweight="bold", y=1.01)

# for ax, regime in zip(axes, regimes_present):
#     data   = [dws[(dws["stress_regime"]==regime) & (dws["method"]==m)]["day_profit"].values
#               for m in METHODS]
#     colors = [METHOD_COLOR[m] for m in METHODS]
#     _violin_box(ax, data, np.arange(len(METHODS)), colors, widths=0.65)
#     ax.set_xticks(np.arange(len(METHODS)))
#     ax.set_xticklabels(xlbls, fontsize=8)
#     ax.set_title(regime, fontsize=11, fontweight="bold", pad=6,
#                  color=REGIME_COLOR[regime])
#     ax.yaxis.grid(True, zorder=0, alpha=0.5); ax.set_axisbelow(True)
#     ax.axhline(0, color="#9CA3AF", linewidth=0.8, linestyle="--")
#     if ax is axes[0]:
#         ax.set_ylabel("Daily Profit (£)")
#         ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_gbp))

# method_patches = [mpatches.Patch(facecolor=METHOD_COLOR[m], label=_short(m).replace("\n"," "))
#                   for m in METHODS]
# fig.legend(handles=method_patches, loc="lower center", ncol=4,
#            frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))

# plt.tight_layout()
# plt.savefig(f"{OUT}/figS4_stress_violin_per_regime.png"); plt.close(); print("Fig S4 saved.")

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE S5 – Per-method violin: profit split by stress regime (2×4 grid)
# # ═════════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)
# fig.suptitle("Per-Method Daily Profit Distribution Across Stress Regimes",
#              fontsize=14, fontweight="bold", y=1.01)

# for ax, method in zip(axes.flatten(), METHODS):
#     data   = [dws[(dws["method"]==method) & (dws["stress_regime"]==r)]["day_profit"].values
#               for r in regimes_present]
#     colors = [REGIME_COLOR[r] for r in regimes_present]
#     _violin_box(ax, data, np.arange(len(regimes_present)), colors, widths=0.55)
#     ax.set_xticks(np.arange(len(regimes_present)))
#     ax.set_xticklabels(regimes_present, fontsize=7.5, rotation=15, ha="right")
#     ax.set_title(_short(method), fontsize=10, fontweight="bold", pad=5)
#     ax.yaxis.grid(True, zorder=0, alpha=0.5); ax.set_axisbelow(True)
#     ax.axhline(0, color="#9CA3AF", linewidth=0.7, linestyle="--")
#     if ax in axes[:,0]:
#         ax.set_ylabel("Daily Profit (£)")
#         ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_gbp))

# regime_patches = [mpatches.Patch(facecolor=REGIME_COLOR[r], label=r)
#                   for r in regimes_present]
# fig.legend(handles=regime_patches, loc="lower center", ncol=4,
#            frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
# plt.tight_layout()
# plt.savefig(f"{OUT}/figS5_stress_per_method_violin.png"); plt.close(); print("Fig S5 saved.")

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE S6 – Spike-day scatter: max price vs daily profit, coloured by method
# # ═════════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle("Profit vs. Market Stress Intensity",
#              fontsize=14, fontweight="bold", y=1.01)

# spike_all = dws[dws["is_spike"] | dws["is_high_vol"]].copy()

# for ax, (xcol, xlabel, title) in zip(axes, [
#         ("price_max",       "Daily Max Price (£/MWh)",   "(a) Spike Days: Profit vs Max Price"),
#         ("price_std",       "Price Std Dev (£/MWh)",     "(b) High-Vol Days: Profit vs Price Volatility"),
# ]):
#     regime_mask = dws["is_spike"] if "Max" in xlabel else dws["is_high_vol"]
#     sub = dws[regime_mask]
#     for method in METHODS:
#         ms = sub[sub["method"]==method]
#         ax.scatter(ms[xcol], ms["day_profit"],
#                    color=METHOD_COLOR[method], alpha=0.65, s=22,
#                    label=_short(method).replace("\n"," "),
#                    linewidths=0, zorder=3)

#     # trend line across all methods
#     xs = sub[xcol].values; ys = sub["day_profit"].values
#     mask_valid = np.isfinite(xs) & np.isfinite(ys)
#     if mask_valid.sum() > 2:
#         z = np.polyfit(xs[mask_valid], ys[mask_valid], 1)
#         xline = np.linspace(xs[mask_valid].min(), xs[mask_valid].max(), 100)
#         ax.plot(xline, np.polyval(z, xline), color="#374151", linewidth=1.2,
#                 linestyle="--", alpha=0.6, label="Linear trend")

#     ax.set_xlabel(xlabel); ax.set_ylabel("Daily Profit (£)")
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_gbp))
#     ax.set_title(title, fontsize=11, pad=8)
#     ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
#     ax.axhline(0, color="#9CA3AF", linewidth=0.8, linestyle="--")

# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
#            fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
# plt.tight_layout()
# plt.savefig(f"{OUT}/figS6_stress_scatter.png"); plt.close(); print("Fig S6 saved.")

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE S7 – Cumulative profit on stress-day subsets (spike vs high-vol vs normal)
# # ═════════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# fig.suptitle("Cumulative Profit — Stress Regime Subsets",
#              fontsize=14, fontweight="bold", y=1.01)

# for ax, (regime, title) in zip(axes, [
#         ("Normal",  "(a) Normal Days"),
#         ("Spike",   "(b) Spike-Event Days"),
#         ("High Vol","(c) High-Volatility Days"),
# ]):
#     subset_regime = dws[dws["stress_regime"] == regime]
#     for method in METHODS:
#         sub = subset_regime[subset_regime["method"]==method].sort_values("date")
#         if sub.empty: continue
#         cum = sub["day_profit"].cumsum().values
#         ax.plot(np.arange(len(cum)), cum,
#                 color=METHOD_COLOR[method], linewidth=1.6,
#                 linestyle=METHOD_LS[method],
#                 label=_short(method).replace("\n"," "))

#     ax.set_xlabel("Day (sorted)")
#     ax.set_ylabel("Cumulative Profit (£)")
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_gbp))
#     ax.set_title(title, fontsize=11, pad=8)
#     ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
#     ax.axhline(0, color="#9CA3AF", linewidth=0.8, linestyle="--")

# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
#            fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
# plt.tight_layout()
# plt.savefig(f"{OUT}/figS7_stress_cumulative.png"); plt.close(); print("Fig S7 saved.")

# # ═════════════════════════════════════════════════════════════════════════════
# # FIGURE S8 – Relative performance: normalised profit vs Normal baseline
# # ═════════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle("Relative Performance Under Stress (Normalised to Normal-Day Baseline)",
#              fontsize=13, fontweight="bold", y=1.01)

# pivot_mean = stress_summary.pivot(index="method", columns="stress_regime",
#                                   values="mean_day_profit").reindex(METHODS)
# pivot_p5   = stress_summary.pivot(index="method", columns="stress_regime",
#                                   values="p5_day_profit").reindex(METHODS)

# stress_regimes_nonnorm = [r for r in regimes_present if r != "Normal"]
# sw2  = 0.22
# off2 = np.linspace(-(len(stress_regimes_nonnorm)-1)/2,
#                     (len(stress_regimes_nonnorm)-1)/2, len(stress_regimes_nonnorm)) * sw2

# for ax, (pivot, ylabel, title) in zip(axes, [
#         (pivot_mean, "Relative Mean Profit (%)", "(a) Mean Daily Profit vs Normal"),
#         (pivot_p5,   "Relative P5 Profit (%)",   "(b) P5 Downside vs Normal"),
# ]):
#     normal_base = pivot["Normal"]
#     for offset, regime in zip(off2, stress_regimes_nonnorm):
#         rel = ((pivot[regime] - normal_base) / normal_base.abs() * 100)
#         colors_bar = [REGIME_COLOR[regime] if v >= 0 else "#374151" for v in rel.values]
#         for xi, (vi, ci) in enumerate(zip(rel.values, colors_bar)):
#             ax.bar(xi + offset, vi, sw2, color=ci, alpha=0.85,
#                    edgecolor="white", linewidth=0.4, zorder=3,
#                    label=regime if xi == 0 else "")

#     ax.axhline(0, color="black", linewidth=1.0, zorder=4)
#     ax.set_xticks(x8); ax.set_xticklabels(xlbls, fontsize=8.5)
#     ax.set_ylabel(ylabel)
#     ax.yaxis.set_major_formatter(mticker.PercentFormatter())
#     ax.set_title(title, fontsize=11, pad=8)
#     ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)

# # unified legend
# legend_handles = [mpatches.Patch(facecolor=REGIME_COLOR[r], label=r)
#                   for r in stress_regimes_nonnorm]
# fig.legend(handles=legend_handles, loc="lower center", ncol=3,
#            frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
# plt.tight_layout()
# plt.savefig(f"{OUT}/figS8_relative_stress.png"); plt.close(); print("Fig S8 saved.")

# # ─────────────────────────────────────────────────────────────────────────────
# # Save summary CSVs
# # ─────────────────────────────────────────────────────────────────────────────
# stress_feat.to_csv(f"{OUT}/stress_features_daily.csv", index=False)
# stress_summary.to_csv(f"{OUT}/stress_regime_summary.csv", index=False)
# dws.to_csv(f"{OUT}/daily_profit_with_stress.csv", index=False)
# print(f"\nAll stress figures + CSVs written to ./{OUT}/")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 0. CONFIG
# ============================================================
START_DATE = "2019-01-01"
OUT = "stress_selected_figures"
os.makedirs(OUT, exist_ok=True)

FILE_MAP = {
    "MILP":         "MILP_forecast_RF_one_year_NEWYORK_2023.csv",
    "MILP(BC)":     "bias_corrected_milp_NEWYORK_2023.csv",
    "MILP(Stoch)":  "bias_milp_stochastic_expected_results_NEWYORK_2023.csv",
    r"MILP(CVaR$_{\beta=0.95,\lambda=0.5}$)": "cvar_milp_results_NEWYORK_2023.csv",
    "HeurTime":     "HeurTime_forecast_RF_one_year_NEWYORK_2023.csv",
    "HeurQuantile": "HeurQuantile_forecast_RF_one_year_NEWYORK_2023.csv",
    "GPT-4":        "gpt4_hourly_arbitrage_new_york_2023_corrected.csv",
    "RL-PPO":       "rlPPO_forecast_RF_one_year_NEWYORK_2023.csv",
}

METHODS = list(FILE_MAP.keys())
STRESS_BASE = "MILP"

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

REGIME_ORDER = ["Normal", "Spike", "High Vol", "Spike + High Vol"]
REGIME_COLOR = {
    "Normal":           "#6B7280",
    "Spike":            "#EF4444",
    "High Vol":         "#F59E0B",
    "Spike + High Vol": "#7C3AED",
}
REGIME_HATCH = {
    "Normal":           "",
    "Spike":            "",
    "High Vol":         "///",
    "Spike + High Vol": "xxx",
}

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

def _fmt_gbp(v, _):
    return f"£{v:,.0f}"

def _short(m):
    return SHORT_LABEL.get(m, m)

# ============================================================
# 1. DATA PREP
# ============================================================
def month_to_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    if m in [3, 4, 5]:
        return "Spring"
    if m in [6, 7, 8]:
        return "Summer"
    return "Autumn"

def prepare_result_df(path, method_name, start_date=START_DATE):
    df = pd.read_csv(path).copy()
    n_rows = len(df)
    n_days = n_rows // 24

    df["day_index"] = df["day"].astype(int) if "day" in df.columns else np.arange(n_rows) // 24
    df["hour"] = np.arange(n_rows) % 24

    dr = pd.date_range(start=start_date, periods=n_days, freq="D")
    df["date"] = df["day_index"].map({i: dr[i] for i in range(n_days)})
    df["is_weekend"] = df["date"].dt.weekday >= 5
    df["regime_week"] = np.where(df["is_weekend"], "Weekend", "Weekday")
    df["season"] = df["date"].dt.month.apply(month_to_season)
    df["method"] = method_name
    return df

def make_daily(hourly_df):
    return hourly_df.groupby(
        ["method", "day_index", "date", "regime_week", "season"], as_index=False
    ).agg(day_profit=("profit_step", "sum"))

print("Loading CSVs ...")
all_hourly, all_daily = [], []
for m, p in FILE_MAP.items():
    h = prepare_result_df(p, m)
    all_hourly.append(h)
    all_daily.append(make_daily(h))
    print(f"  {m}: loaded")

hourly_all = pd.concat(all_hourly, ignore_index=True)
daily_all = pd.concat(all_daily, ignore_index=True)

# ============================================================
# 2. STRESS CLASSIFICATION
# ============================================================
base_h = hourly_all[hourly_all["method"] == STRESS_BASE].copy()

stress_feat = base_h.groupby(["day_index", "date"], as_index=False).agg(
    price_max=("prices_actual", "max"),
    price_min=("prices_actual", "min"),
    price_mean=("prices_actual", "mean"),
    price_std=("prices_actual", "std"),
    price_range=("prices_actual", lambda x: x.max() - x.min()),
    abs_change_mean=("prices_actual", lambda x: np.mean(np.abs(np.diff(x.values))) if len(x) > 1 else 0.0),
    abs_change_max=("prices_actual", lambda x: np.max(np.abs(np.diff(x.values))) if len(x) > 1 else 0.0),
)

thr_spike = stress_feat["price_max"].quantile(0.95)
thr_vol = stress_feat["price_std"].quantile(0.90)

stress_feat["is_spike"] = stress_feat["price_max"] >= thr_spike
stress_feat["is_high_vol"] = stress_feat["price_std"] >= thr_vol
stress_feat["stress_regime"] = "Normal"
stress_feat.loc[stress_feat["is_spike"], "stress_regime"] = "Spike"
stress_feat.loc[stress_feat["is_high_vol"], "stress_regime"] = "High Vol"
stress_feat.loc[
    stress_feat["is_spike"] & stress_feat["is_high_vol"],
    "stress_regime"
] = "Spike + High Vol"

dws = daily_all.merge(
    stress_feat[[
        "day_index", "date", "is_spike", "is_high_vol", "stress_regime",
        "price_max", "price_std", "price_range", "abs_change_mean"
    ]],
    on=["day_index", "date"],
    how="left",
)

stress_summary = dws.groupby(["method", "stress_regime"], as_index=False).agg(
    n_days=("day_profit", "count"),
    mean_day_profit=("day_profit", "mean"),
    std_day_profit=("day_profit", "std"),
    median_day_profit=("day_profit", "median"),
    p5_day_profit=("day_profit", lambda x: np.quantile(x, 0.05)),
    total_profit=("day_profit", "sum"),
)

xlbls = [_short(m) for m in METHODS]
x8 = np.arange(len(METHODS))

print(f"\nSpike threshold = {thr_spike:.2f}")
print(f"High-vol threshold = {thr_vol:.2f}")

# ============================================================
# 3. PLOT 1 — Mean Hourly Price Change
# ============================================================
def _violin_box(ax, data_list, positions, colors, widths=0.55):
    parts = ax.violinplot(
        data_list, positions=positions, widths=widths,
        showmeans=False, showmedians=False, showextrema=False
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor("none")

    bp = ax.boxplot(
        data_list, positions=positions, widths=widths * 0.27, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=0.9),
        capprops=dict(linewidth=0.9),
        flierprops=dict(marker=".", markersize=2, alpha=0.3, color="#6B7280"),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)

reg_order_plot = [r for r in REGIME_ORDER if r in stress_feat["stress_regime"].unique()]

fig, ax = plt.subplots(figsize=(6, 5))
data = [stress_feat[stress_feat["stress_regime"] == r]["abs_change_mean"].values for r in reg_order_plot]
colors = [REGIME_COLOR[r] for r in reg_order_plot]
_violin_box(ax, data, np.arange(len(reg_order_plot)), colors, widths=0.5)

ax.set_xticks(np.arange(len(reg_order_plot)))
ax.set_xticklabels(reg_order_plot, fontsize=11)
ax.set_ylabel("Mean |ΔPrice| (£/MWh)")
ax.set_title("Mean Hourly Price Change", fontsize=14, pad=8)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)

for i, r in enumerate(reg_order_plot):
    n = (stress_feat["stress_regime"] == r).sum()
    ax.text(i, ax.get_ylim()[0], f"n={n}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/plot1_mean_hourly_price_change.png")
plt.show()

# ============================================================
# 4. PLOT 2 — P5 Downside Risk
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))

regimes_present = [r for r in REGIME_ORDER if r in stress_summary["stress_regime"].unique()]
# show the three stressed regimes from your screenshot style
plot_regimes = [r for r in ["Normal", "High Vol", "Spike + High Vol"] if r in regimes_present]

rw = 0.22
roff = np.linspace(-(len(plot_regimes)-1)/2, (len(plot_regimes)-1)/2, len(plot_regimes)) * rw

pivot = stress_summary.pivot(index="method", columns="stress_regime", values="p5_day_profit").reindex(METHODS)[plot_regimes]

for offset, regime in zip(roff, plot_regimes):
    ax.bar(
        x8 + offset,
        pivot[regime],
        rw,
        color=REGIME_COLOR[regime],
        alpha=0.88,
        label=regime,
        edgecolor="white",
        linewidth=0.4,
        hatch=REGIME_HATCH[regime],
        zorder=3,
    )

ax.set_xticks(x8)
ax.set_xticklabels(xlbls, fontsize=10)
ax.set_ylabel("P5 Daily Profit (£)")
ax.set_title("P5 Downside Risk", fontsize=14, pad=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_gbp))
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.axhline(0, color="#9CA3AF", linewidth=0.7, linestyle="--")

ax.legend(title="Stress Regime", ncol=2, loc="lower left", fontsize=10, title_fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUT}/plot2_p5_downside_risk.png")
plt.show()

# ============================================================
# 5. PLOT 3 — Cumulative Profit on Normal Days
# ============================================================
# ============================================================
# COMBINED PLOT — Cumulative Profit on Normal vs High-Volatility Days
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

plot_specs = [
    ("Normal", "Normal Days"),
    ("High Vol", "High-Volatility Days"),
]

for ax, (regime_name, title) in zip(axes, plot_specs):
    subset_regime = dws[dws["stress_regime"] == regime_name]

    for method in METHODS:
        sub = subset_regime[subset_regime["method"] == method].sort_values("date")
        if sub.empty:
            continue

        cum = sub["day_profit"].cumsum().values
        ax.plot(
            np.arange(len(cum)),
            cum,
            color=METHOD_COLOR[method],
            linewidth=1.8,
            linestyle=METHOD_LS[method],
            label=_short(method).replace("\n", " "),
        )

    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Profit (£)")
    ax.set_title(title, fontsize=14, pad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_gbp))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#9CA3AF", linewidth=0.8, linestyle="--")

# --- shared legend ---
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    frameon=False,
    # fontsize=10,
    bbox_to_anchor=(0.5, 0.01),
)

plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig(f"{OUT}/combined_cum_profit_normal_vs_highvol.png", bbox_inches="tight")
plt.show()

print(f"\nSaved 4 selected plots to ./{OUT}/")