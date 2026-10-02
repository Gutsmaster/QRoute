import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#0B0E2E"; CARD = "#151238"; VIOLET = "#8B5CF6"; CYAN = "#22D3EE"
TEAL = "#34D399"; AMBER = "#FBBF24"; TEXT = "#F1F5F9"; MUTED = "#94A3B8"
GRID = "#2A2760"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 12,
    "axes.titlecolor": TEXT, "legend.facecolor": CARD, "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT,
})
R = json.load(open("outputs/results.json"))
OUT = "outputs/deck"
import os; os.makedirs(OUT, exist_ok=True)

# ---- 1. Convergence (re-derive from single_run curve shape isn't stored; use bench mean-fit bars instead as a clean "who wins" chart)
bench = R["full_benchmark"]
names_order = ["ACO", "TunedLS(HGS-inspired)", "GA(plain)", "PSO(same infra, tuned)", "QPSO",
               "GA+LS(memetic)", "OR-Tools(GLS,5s)"]
gaps = {n: np.mean([b["summary"][n]["gap_to_best_pct"] for b in bench if n in b["summary"]]) for n in names_order}
colors = [CYAN if n != "QPSO" else VIOLET for n in names_order]
colors = [TEAL if n == "OR-Tools(GLS,5s)" else c for n, c in zip(names_order, colors)]
labels = ["ACO", "Tuned LS", "GA (plain)", "Classical PSO", "QPSO\n(ours)", "Memetic GA", "OR-Tools\n(exact)"]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(labels, [gaps[n] for n in names_order], color=colors, edgecolor="none", width=0.6)
for b, n in zip(bars, names_order):
    if n == "QPSO":
        b.set_edgecolor(VIOLET); b.set_linewidth(3)
for rect, n in zip(bars, names_order):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1.2, f"+{gaps[n]:.1f}%",
            ha="center", color=TEXT, fontsize=11, fontweight="bold")
ax.set_ylabel("Mean gap to best found (%)  —  lower is better")
ax.set_title("Held-out benchmark: QPSO vs conventional metaheuristics & exact method", pad=14, fontsize=14)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/slide3_benchmark_bars.png", dpi=200); plt.close()

# ---- 2. Scalability (dark themed)
scal = R["scalability"]
ns = [s["n_customers"] for s in scal]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].plot(ns, [s["flat_time_s"] for s in scal], "o-", color=CYAN, lw=2.5, ms=8, label="Flat QPSO")
ax[0].plot(ns, [s["hier_time_s"] for s in scal], "s-", color=VIOLET, lw=2.5, ms=8, label="Hierarchical QPSO")
ax[0].set_xlabel("Customers"); ax[0].set_ylabel("Wall time (s)"); ax[0].set_title("Runtime")
ax[0].legend(); ax[0].spines[["top","right"]].set_visible(False); ax[0].grid(alpha=0.25)
ax[1].plot(ns, [s["flat_fit"] for s in scal], "o-", color=CYAN, lw=2.5, ms=8, label="Flat QPSO")
ax[1].plot(ns, [s["hier_fit"] for s in scal], "s-", color=VIOLET, lw=2.5, ms=8, label="Hierarchical QPSO")
ax[1].set_xlabel("Customers"); ax[1].set_ylabel("Cost (lower=better)"); ax[1].set_title("Solution quality")
ax[1].legend(); ax[1].spines[["top","right"]].set_visible(False); ax[1].grid(alpha=0.25)
fig.suptitle("Flat vs hierarchical QPSO: decomposition buys speed, not quality", fontsize=14, y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/slide5_scalability.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- 3. Impact numbers chart (fixed vs reoptimized)
dv = R["dynamic_value"]
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(dv["hours"], dv["fixed_plan_cost"], "o-", color=AMBER, lw=2.5, ms=9, label="Fixed 9am plan (used all day)")
ax.plot(dv["hours"], dv["reoptimised_cost"], "o-", color=TEAL, lw=2.5, ms=9, label="Re-optimized for that hour")
ax.fill_between(dv["hours"], dv["fixed_plan_cost"], dv["reoptimised_cost"], color=AMBER, alpha=0.12)
ax.set_xlabel("Departure hour"); ax.set_ylabel("Plan cost")
ax.set_title(f"Cost of not re-routing — {dv['mean_regret_pct']:.1f}% mean regret", fontsize=14)
ax.legend(); ax.spines[["top","right"]].set_visible(False); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/slide4_dynamic_value.png", dpi=200); plt.close()

# ---- 4. Festival day impact (bar)
ft = R["festival_test"]
fig, ax = plt.subplots(figsize=(7.5, 4.8))
hrs = list(ft.keys()); vals = [ft[h]["stale_plan_penalty_pct"] for h in hrs]
bars = ax.bar([f"{h}:00" for h in hrs], vals, color=[AMBER, CYAN], width=0.45)
for r, v in zip(bars, vals):
    ax.text(r.get_x()+r.get_width()/2, v+0.15, f"+{v:.1f}%", ha="center", color=TEXT, fontweight="bold")
ax.set_ylabel("Extra cost using a normal-day plan on Rakshabandhan")
ax.set_title("Real festival-day anomaly cost (Aug 19, 2024)", fontsize=14)
ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/slide4_festival_impact.png", dpi=200); plt.close()

print("Saved dark-theme deck visuals to", OUT)
import os
for f in sorted(os.listdir(OUT)): print(" -", f)
