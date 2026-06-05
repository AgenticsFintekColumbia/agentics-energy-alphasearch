import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io

# ── Battery parameters ───────────────────────────────────────────────────────
C      = 49.44
eta_c  = 0.95
eta_d  = 0.95
P_max  = 12.36
dt     = 1.0
SOC_0  = 0.5

csv_data = """soc,charge_MW,discharge_MW,import_MW,export_MW,net_MW,objective_cost,run_id
0.5368421052631579,0.0,12.36,14.85163527,0.0,14.85163527,45551.649548329995,0
0.5368421052631579,0.0,0.0,25.61946785,0.0,25.61946785,45551.649548329995,0
0.5368421052631579,0.0,0.0,24.58397957,0.0,24.58397957,45551.649548329995,0
0.7743421052631579,12.36,0.0,36.58109296,0.0,36.58109296,45551.649548329995,0
1.0,11.743711911357337,0.0,35.92142956135734,0.0,35.92142956135734,45551.649548329995,0
1.0,0.0,0.0,24.83470126,0.0,24.83470126,45551.649548329995,0
1.0,0.0,0.0,26.19666667,0.0,26.19666667,45551.649548329995,0
0.736842105263158,0.0,12.359999999999996,15.599220480000003,0.0,15.599220480000003,45551.649548329995,0
0.4736842105263159,0.0,12.359999999999998,22.546100520000003,0.0,22.546100520000003,45551.649548329995,0
0.21052631578947378,0.0,12.359999999999998,22.298405090000003,0.0,22.298405090000003,45551.649548329995,0
0.21052631578947378,0.0,0.0,34.97362482,0.0,34.97362482,45551.649548329995,0
0.21052631578947378,0.0,0.0,33.81940341,0.0,33.81940341,45551.649548329995,0
0.4480263157894738,12.36,0.0,46.54180884,0.0,46.54180884,45551.649548329995,0
0.6855263157894738,12.36,0.0,48.6975,0.0,48.6975,45551.649548329995,0
0.9230263157894738,12.36,0.0,49.78111111,0.0,49.78111111,45551.649548329995,0
0.9230263157894738,0.0,0.0,34.66035584,0.0,34.66035584,45551.649548329995,0
0.9230263157894738,0.0,0.0,35.169247,0.0,35.169247,45551.649548329995,0
0.659736842105263,0.0,12.36,25.443896459999998,0.0,25.443896459999998,45551.649548329995,0
0.3964473684210526,0.0,12.36,26.8327837,0.0,26.8327837,45551.649548329995,0
0.1331578947368421,0.0,12.36,27.12984687,0.0,27.12984687,45551.649548329995,0
0.0,0.0,6.254159999999999,31.677840000000003,0.0,31.677840000000003,45551.649548329995,0
0.0,0.0,0.0,35.10596615,0.0,35.10596615,45551.649548329995,0
0.0,0.0,0.0,32.22280021,0.0,32.22280021,45551.649548329995,0"""

df_raw = pd.read_csv(io.StringIO(csv_data))
soc_csv = df_raw["soc"].values
soc_all = np.concatenate([[SOC_0], soc_csv])  # length 24 here after prepend? actually 24
hours = np.arange(len(soc_all))

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
    "legend.fontsize": 12,
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

# ── Back-calculate c[t] and d[t] from SOC ───────────────────────────────────
charge_MW = np.zeros(24)
discharge_MW = np.zeros(24)

for t in range(23):
    delta = soc_all[t + 1] - soc_all[t]
    if delta >= 0:
        charge_MW[t] = delta * C / (eta_c * dt)
        discharge_MW[t] = 0.0
    else:
        discharge_MW[t] = -delta * C * eta_d / dt
        charge_MW[t] = 0.0

# ── Actual prices (24 hours) ────────────────────────────────────────────────
prices_h1_24 = np.array([
    65.10, 62.12, 59.05, 59.00, 56.63, 56.63, 61.66,
    66.61, 73.00, 73.00, 65.00, 63.82, 62.19, 59.50,
    56.99, 61.92, 65.63, 82.47, 82.79, 83.50, 78.62,
    72.94, 64.53, 60.32
])

# ── Profit ──────────────────────────────────────────────────────────────────
hourly_profit = discharge_MW * prices_h1_24 - charge_MW * prices_h1_24
cumulative_profit = np.cumsum(hourly_profit)

# ── Base Qwen: flat SOC=0.5, zero profit ───────────────────────────────────
soc_base = np.full(24, 0.5)
cum_base = np.zeros(24)

# ── Colors ──────────────────────────────────────────────────────────────────
COLOR_FT = "C0"
COLOR_BASE = "C3"

# ============================================================================
# 1. PLOT: SOC trajectory + actual price
# ============================================================================
fig, ax1 = plt.subplots(figsize=(10.5, 4.8))

ax1.plot(
    hours, soc_all,
    color=COLOR_FT, marker="o", ms=4, lw=1.8,
    label="Fine-tuned Qwen"
)
ax1.plot(
    hours, soc_base,
    color=COLOR_BASE, marker="s", ms=3.5, lw=1.6, ls="--",
    label="Base Qwen"
)

ax1.set_ylabel("SOC")
ax1.set_xlabel("Hour")
ax1.set_ylim(-0.05, 1.12)
ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax1.set_title("Battery SOC trajectory")
ax1.grid(True, alpha=0.3)

ax1b = ax1.twinx()
ax1b.plot(
    hours, prices_h1_24,
    color="black", lw=1.8, label="Actual price"
)
ax1b.set_ylabel("EUR/MWh")

# combined legend outside plot area
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1b.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labs1 + labs2,
    loc="upper left",
    bbox_to_anchor=(1.18, 1.0),   # push legend further right
    borderaxespad=0.0,
    fontsize=11
)

plt.tight_layout(rect=[0, 0, 1.04, 1])   # reserve more space on right

# ax1.legend(
#     lines1 + lines2,
#     labs1 + labs2,
#     loc="upper left",
#     bbox_to_anchor=(1.02, 1.0),
#     borderaxespad=0.0,
#     fontsize=11
# )

# plt.tight_layout(rect=[0, 0, 0.84, 1])
plt.savefig("battery_soc_trajectory.png", dpi=150)
plt.show()

# ============================================================================
# 2. PLOT: cumulative profit
# ============================================================================
fig, ax2 = plt.subplots(figsize=(10.5, 4.8))

ft_total = cumulative_profit[-1]

ax2.plot(
    hours, cumulative_profit,
    color=COLOR_FT, marker="o", ms=4, lw=1.8,
    label=f"Fine-tuned Qwen  ({ft_total:+.2f} EUR)"
)
ax2.plot(
    hours, cum_base,
    color=COLOR_BASE, marker="s", ms=3.5, lw=1.6, ls="--",
    label="Base Qwen  (+0.00 EUR)"
)

ax2.axhline(0, color="gray", linewidth=0.9, linestyle="--")
ax2.set_ylabel("EUR")
ax2.set_xlabel("Hour")
# ax2.set_title("Cumulative profit over the day")
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# keep legend inside, top-left, where it doesn't clash much
ax2.legend(loc="upper left", fontsize=11)

plt.tight_layout()
plt.savefig("battery_cumulative_profit.png", dpi=150)
plt.show()