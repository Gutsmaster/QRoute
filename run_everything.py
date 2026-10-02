"""
End-to-end driver:  python3 run_everything.py   -> outputs/
Sections: demo instance (convergence) | festival-day anomaly test | disruption
recovery | hierarchical scalability | Pareto (4 metrics) | MAP-Elites archive +
road-closure fallback | fixed-plan vs re-optimised across the day |
HELD-OUT benchmark (Wilcoxon, Dolan-More, regime map, ablation).
"""
import sys, os, json, time, shutil, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src"); sys.path.insert(0, "benchmarks")
from data_loader import load_day
from calibration import build_edge_table
from graph_builder import build_graph, largest_component_subgraph
from vrp_instance import make_synthetic_instance, VRPInstance
from qpso_core import QPSO, evaluate_routes, fitness_weighted
from local_search import split_optimal, local_search_pass
from quantum_extras import (quantum_walk_centrality, hierarchical_qpso, pareto_front_sweep,
                             build_map_elites_archive, route_stability)
from baselines import GeneticAlgorithm, AntColony, TunedLocalSearch
from solomon_synthetic import generate_sweep
import run_full_benchmark as RB

QPSO.STACK_DEFAULTS.update(writeback_mode="sorted_reassign", ls_extended=True)   # confirmed stack: write-back + extended inter-route search
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
GJ = "data/new_delhi_traffic_dataset/probe_counts/geojson/new_delhi__%s_to_%s_.geojson"
BBOX = (77.205, 28.625, 77.225, 28.645)
N = 24
QUICK = os.environ.get("QUICK") == "1"
NP, NI = (6, 6) if QUICK else (24, 80)
SEEDS = (10,) if QUICK else (10, 11, 12, 13)
OT = 1 if QUICK else 5
R = {}; log = []
def L(m):
    print(m, flush=True); log.append(m)
def save():
    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    json.dump(clean(R), open(f"{OUT}/results.json", "w"), indent=1)
    open(f"{OUT}/log.txt", "w").write("\n".join(log))
def unpack(inst, hour):
    return dict(demands=[inst.demands[c] for c in inst.customers],
                tw=[inst.time_windows[c] for c in inst.customers],
                T=inst.time_mat[hour], D=inst.dist_mat[hour], E=inst.emis_mat[hour])
def build_day(day):
    d = build_edge_table(load_day(GJ % (day, day)))
    G = build_graph(d)
    return G, largest_component_subgraph(G, BBOX, strongly_connected=True)
t_start = time.time()

# ------------------------------------------------------------------ data
L("=== Loading real Delhi NCR data (Aug 12, 2024: normal weekday) ===")
G_full, Gs = build_day("2024-08-12")
R["graph_stats"] = {"full_nodes": G_full.number_of_nodes(), "full_edges": G_full.number_of_edges(),
                    "demo_nodes": Gs.number_of_nodes(), "demo_edges": Gs.number_of_edges()}
L(f"Full graph {R['graph_stats']['full_nodes']} nodes / {R['graph_stats']['full_edges']} edges; "
  f"demo subgraph {Gs.number_of_nodes()} / {Gs.number_of_edges()}")
inst = make_synthetic_instance(Gs, n_customers=N, capacity=100, seed=0)
P = unpack(inst, 9)
args = (N, P["demands"], 100, P["tw"], P["T"], P["D"], P["E"])

# ------------------------------------------------------------------ 1. convergence (illustration on the demo instance)
L("=== 1. Convergence on the demo instance (9am) ===")
qc = quantum_walk_centrality(Gs, [inst.depot] + inst.customers)
bias = qc[1:] / qc[1:].max()
runs = {
    "QPSO": QPSO(*args, n_particles=NP, n_iter=NI, seed=1).run(),
    "QPSO + quantum-walk init bias": QPSO(*args, n_particles=NP, n_iter=NI, seed=1, init_bias=bias).run(),
    "PSO (same infra, tuned)": QPSO(*args, n_particles=NP, n_iter=NI, seed=1, update_rule="classical").run(),
    "GA + local search": GeneticAlgorithm(*args, pop_size=NP, n_gen=NI, seed=1, **RB.GA_MEMETIC_KW).run(),
    "GA (plain)": GeneticAlgorithm(*args, pop_size=NP + 10, n_gen=NI, seed=1).run(),
    "ACO": AntColony(*args, n_ants=NP, n_iter=max(2, int(NI * 0.8)), seed=1).run(),
}
ls_res = TunedLocalSearch(*args, n_restarts=15, seed=1).run()
plt.figure(figsize=(7.5, 4.6))
for k, v in runs.items():
    plt.plot(v["history"], label=k, lw=2.2 if k == "QPSO" else 1.3)
plt.axhline(ls_res["gbest_fit"], color="gray", ls="--", label="Tuned LS (final)")
plt.xlabel("Iteration"); plt.ylabel("Best weighted cost"); plt.title("Convergence — real Delhi instance, 9am departure")
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT}/convergence.png", dpi=130); plt.close()
R["single_run"] = {k: v["gbest_fit"] for k, v in runs.items()}; R["single_run"]["TunedLS"] = ls_res["gbest_fit"]
R["quantum_walk_note"] = "quantum-walk init bias evaluated on the demo instance only; not part of the held-out claim"
L("single-run (seed 1): " + ", ".join(f"{k}={v:.0f}" for k, v in R["single_run"].items()))
best_routes = runs["QPSO"]["gbest_routes"]; save()

# ------------------------------------------------------------------ 2. festival-day anomaly test (REAL anomaly data)
L("=== 2. Festival-day test: plan built on Aug 12 evaluated on Rakshabandhan (Aug 19) ===")
fest = {}
try:
    _, Gf = build_day("2024-08-19")
    inst_f = VRPInstance(Gf, inst.depot, inst.customers, inst.demands, inst.capacity, inst.time_windows)
    for hour in (9, 18):
        Pn, Pf = unpack(inst, hour), unpack(inst_f, hour)
        an = (N, Pn["demands"], 100, Pn["tw"], Pn["T"], Pn["D"], Pn["E"])
        af = (N, Pf["demands"], 100, Pf["tw"], Pf["T"], Pf["D"], Pf["E"])
        plan_n = QPSO(*an, n_particles=NP, n_iter=NI, seed=1).run()
        plan_f = QPSO(*af, n_particles=NP, n_iter=NI, seed=1).run()
        stale = fitness_weighted(evaluate_routes(plan_n["gbest_routes"], Pf["T"], Pf["D"], Pf["E"], Pf["tw"], Pf["demands"], 100))
        fest[hour] = {"normal_day_plan_on_normal_day": plan_n["gbest_fit"], "normal_day_plan_on_festival_day": float(stale),
                      "reoptimised_on_festival_day": plan_f["gbest_fit"],
                      "stale_plan_penalty_pct": 100 * (stale - plan_f["gbest_fit"]) / plan_f["gbest_fit"],
                      "festival_vs_normal_cost_pct": 100 * (plan_f["gbest_fit"] - plan_n["gbest_fit"]) / plan_n["gbest_fit"]}
        L(f"  hour {hour}: stale-plan penalty on festival day = {fest[hour]['stale_plan_penalty_pct']:.1f}% ; "
          f"festival-day optimum vs normal-day optimum = {fest[hour]['festival_vs_normal_cost_pct']:+.1f}%")
except Exception as e:
    L(f"  festival test skipped: {e}")
R["festival_test"] = fest; save()

# ------------------------------------------------------------------ 3. disruption recovery (5 seeds)
L("=== 3. Disruption recovery (5 seeds): 25% of links slow x3 at iteration 40 ===")
rng = np.random.default_rng(42)
Td = P["T"] * (1 + 2 * (rng.random(P["T"].shape) < 0.25)); np.fill_diagonal(Td, 0)
H_on, H_off = [], []
for s in range(5):
    H_on.append(QPSO(*args, n_particles=NP, n_iter=NI, seed=s, disruption_boost=True).run(40, Td)["history"])
    H_off.append(QPSO(*args, n_particles=NP, n_iter=NI, seed=s, disruption_boost=False).run(40, Td)["history"])
H_on, H_off = np.array(H_on), np.array(H_off)
plt.figure(figsize=(7.5, 4.6))
for H, lab in ((H_on, "with reverse-annealing recovery"), (H_off, "no recovery (beta keeps decaying)")):
    plt.plot(H.mean(0), label=lab); plt.fill_between(range(H.shape[1]), H.min(0), H.max(0), alpha=0.15)
plt.axvline(40, color="red", ls=":", label="disruption"); plt.xlabel("Iteration"); plt.ylabel("Best cost (on disrupted network)")
plt.title("Recovery after disruption (mean and min-max over 5 seeds)"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{OUT}/recovery.png", dpi=130); plt.close()
R["recovery_demo"] = {"with_recovery_mean_final": float(H_on[:, -1].mean()), "without_recovery_mean_final": float(H_off[:, -1].mean()),
                      "with_recovery_wins": int((H_on[:, -1] < H_off[:, -1]).sum()), "n_seeds": 5}
L(f"  final mean cost: with recovery {H_on[:, -1].mean():.0f} vs without {H_off[:, -1].mean():.0f} "
  f"(recovery better in {R['recovery_demo']['with_recovery_wins']}/5 seeds)"); save()

# ------------------------------------------------------------------ 4. hierarchical scalability
L("=== 4. Scalability: flat vs hierarchical QPSO ===")
scal = []
for n_c in ((12, 24) if QUICK else (12, 24, 48, 72)):
    i2 = make_synthetic_instance(Gs, n_customers=n_c, capacity=100, seed=0); p2 = unpack(i2, 9)
    t0 = time.time()
    flat = QPSO(n_c, p2["demands"], 100, p2["tw"], p2["T"], p2["D"], p2["E"], n_particles=NP, n_iter=NI, seed=5).run()
    tf = time.time() - t0
    xy = np.array([[Gs.nodes[c]["lon"], Gs.nodes[c]["lat"]] for c in i2.customers])
    t0 = time.time()
    h = hierarchical_qpso(xy, p2["demands"], 100, p2["tw"], p2["T"], p2["D"], p2["E"], n_zones=max(2, n_c // 10),
                          n_particles=NP, n_iter=NI, seed=5)
    th = time.time() - t0
    scal.append({"n_customers": n_c, "flat_time_s": tf, "flat_fit": float(flat["gbest_fit"]),
                 "hier_time_s": th, "hier_fit": float(fitness_weighted(h["metrics"])), "zones": h["n_zones"]})
    L(f"  n={n_c}: flat {tf:.1f}s fit {scal[-1]['flat_fit']:.0f} | hierarchical({h['n_zones']} zones) {th:.1f}s fit {scal[-1]['hier_fit']:.0f}")
R["scalability"] = scal
fig, ax = plt.subplots(1, 2, figsize=(10, 4)); ns = [r["n_customers"] for r in scal]
ax[0].plot(ns, [r["flat_time_s"] for r in scal], "o-", label="flat"); ax[0].plot(ns, [r["hier_time_s"] for r in scal], "s-", label="hierarchical")
ax[0].set_xlabel("customers"); ax[0].set_ylabel("wall time (s)"); ax[0].legend(); ax[0].set_title("Runtime")
ax[1].plot(ns, [r["flat_fit"] for r in scal], "o-", label="flat"); ax[1].plot(ns, [r["hier_fit"] for r in scal], "s-", label="hierarchical")
ax[1].set_xlabel("customers"); ax[1].set_ylabel("cost (lower=better)"); ax[1].legend(); ax[1].set_title("Solution quality")
plt.tight_layout(); plt.savefig(f"{OUT}/scalability.png", dpi=130); plt.close(); save()

# ------------------------------------------------------------------ 5. Pareto (time, distance, emissions, stability)
L("=== 5. Multi-objective Pareto sweep ===")
front, cands = pareto_front_sweep(*args, n_weight_samples=5 if QUICK else 12, n_particles=max(4, NP - 4), n_iter=max(4, NI - 20) if not QUICK else 4, seed=7, time_mat_by_hour=inst.time_mat)
L(f"  {len(front)} non-dominated of {len(cands)} candidates (4 metrics)")
fig = plt.figure(figsize=(12, 5)); ax = fig.add_subplot(121, projection="3d")
ax.scatter([c["time_s"] for c in cands], [c["dist_m"] for c in cands], [c["emis_g"] for c in cands], c="lightgray", label="dominated")
ax.scatter([c["time_s"] for c in front], [c["dist_m"] for c in front], [c["emis_g"] for c in front], c="red", s=55, label="Pareto front")
ax.set_xlabel("time (s)"); ax.set_ylabel("distance (m)"); ax.set_zlabel("emissions (g)"); ax.legend(); ax.set_title("Time / distance / emissions")
ax2 = fig.add_subplot(122)
ax2.scatter([c["time_s"] for c in cands], [c["stability_cv"] for c in cands], c="lightgray")
ax2.scatter([c["time_s"] for c in front], [c["stability_cv"] for c in front], c="red")
ax2.set_xlabel("time at 9am (s)"); ax2.set_ylabel("stability CV over 6am-9pm (lower = steadier)"); ax2.set_title("Fastest vs most stable")
plt.tight_layout(); plt.savefig(f"{OUT}/pareto.png", dpi=130); plt.close()
R["pareto"] = {"front_size": len(front), "n_candidates": len(cands),
               "front": [{k: v for k, v in c.items() if k != "routes"} for c in front]}; save()

# ------------------------------------------------------------------ 6. MAP-Elites archive + road-closure fallback
L("=== 6. Diverse plan archive and road-closure fallback ===")
qa = QPSO(*args, n_particles=NP, n_iter=NI, seed=3); ra = qa.run()
cells = build_map_elites_archive(qa.archive, n_route_bins=5, n_emis_bins=4)
grid = np.full((5, 4), np.nan)
for (nr, eb), (fit, keys, met) in cells.items(): grid[nr - 1, eb] = fit
plt.figure(figsize=(6, 4.2)); plt.imshow(grid, cmap="viridis_r", aspect="auto")
plt.colorbar(label="best cost in cell"); plt.xlabel("emissions bucket"); plt.ylabel("number of routes - 1")
plt.title("Plan archive: best plan per behaviour cell"); plt.tight_layout(); plt.savefig(f"{OUT}/archive.png", dpi=130); plt.close()
def plan_from_keys(keys):
    rts = split_optimal(list(np.argsort(keys)), P["demands"], 100, P["T"], P["D"], P["E"], P["tw"], (1.0, 0.02, 0.01, 300.0))
    return local_search_pass(rts, P["demands"], 100, P["T"], P["D"], P["E"], P["tw"], (1.0, 0.02, 0.01, 300.0))
inc = ra["gbest_routes"]
Tc = P["T"].copy(); longest = max(inc, key=len)
for a, b in zip([0] + [c + 1 for c in longest[:-1]], [c + 1 for c in longest]): Tc[a, b] *= 8   # close/jam the incumbent's legs
ev = lambda rts: fitness_weighted(evaluate_routes(rts, Tc, P["D"], P["E"], P["tw"], P["demands"], 100))
inc_cost = ev(inc)
archive_costs = [ev(plan_from_keys(k)) for (_, k, _) in cells.values()]
t0 = time.time(); reopt = QPSO(N, P["demands"], 100, P["tw"], Tc, P["D"], P["E"], n_particles=NP, n_iter=NI, seed=3).run(); t_reopt = time.time() - t0
R["archive_fallback"] = {"n_cells": len(cells), "incumbent_after_closure": float(inc_cost),
                         "best_archive_plan_after_closure": float(min(archive_costs)), "reoptimised_from_scratch": float(reopt["gbest_fit"]),
                         "reoptimise_time_s": t_reopt}
L(f"  {len(cells)} archive cells | after closure: incumbent {inc_cost:.0f}, best archived plan {min(archive_costs):.0f} "
  f"(instant), full re-optimisation {reopt['gbest_fit']:.0f} ({t_reopt:.1f}s)"); save()

# ------------------------------------------------------------------ 7. fixed 9am plan vs re-optimising each hour
L("=== 7. Fixed 9am plan vs re-optimising at each departure hour ===")
hrs = [6, 9, 12, 15, 18, 21]; fixed, reopt_h = [], []
for h in hrs:
    Ph = unpack(inst, h)
    fixed.append(float(fitness_weighted(evaluate_routes(best_routes, Ph["T"], Ph["D"], Ph["E"], Ph["tw"], Ph["demands"], 100))))
    reopt_h.append(float(QPSO(N, Ph["demands"], 100, Ph["tw"], Ph["T"], Ph["D"], Ph["E"], n_particles=NP, n_iter=NI, seed=2).run()["gbest_fit"]))
stab = route_stability(best_routes, inst.time_mat, list(range(6, 22)))
plt.figure(figsize=(7.5, 4.4)); plt.plot(hrs, fixed, "o-", label="fixed 9am plan"); plt.plot(hrs, reopt_h, "s-", label="re-optimised at that hour")
plt.xlabel("departure hour"); plt.ylabel("plan cost"); plt.title("Cost of not re-routing"); plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/dynamic_value.png", dpi=130); plt.close()
R["dynamic_value"] = {"hours": hrs, "fixed_plan_cost": fixed, "reoptimised_cost": reopt_h,
                      "mean_regret_pct": float(100 * np.mean([(f - r) / r for f, r in zip(fixed, reopt_h)]))}
R["route_stability"] = {"mean": float(stab["mean"]), "std": float(stab["std"]), "cv": float(stab["cv"])}
L("  mean regret of never re-routing: %.1f%%" % R["dynamic_value"]["mean_regret_pct"]); save()

# ------------------------------------------------------------------ 8. HELD-OUT benchmark
L("=== 8. HELD-OUT benchmark (instances + seeds different from tuning) ===")
bench = []
for name, seed, hour in (("Delhi-9am-s7", 7, 9), ("Delhi-6pm-s11", 11, 18)):
    ii = make_synthetic_instance(Gs, n_customers=N, capacity=100, seed=seed); pp = unpack(ii, hour)
    r = RB.benchmark_instance(name, N, pp["demands"], 100, pp["tw"], pp["T"], pp["D"], pp["E"], seeds=SEEDS, n_particles=NP, n_iter=NI, ortools_time_limit=OT)
    bench.append(r); L(f"  {name}: " + ", ".join(f"{a}={v['mean_fit']:.0f}" for a, v in r["summary"].items())); save()
for s in generate_sweep(n_per_kind=1 if QUICK else 2, n_customers=N, seed_base=300):
    r = RB.benchmark_instance(s["name"], s["n"], s["demands"], s["capacity"], s["time_windows_s"], s["time_mat"],
                              s["dist_mat"], s["emis_mat"], seeds=SEEDS, n_particles=NP, n_iter=NI, ortools_time_limit=OT)
    bench.append(r); L(f"  {s['name']}: " + ", ".join(f"{a}={v['mean_fit']:.0f}" for a, v in r["summary"].items())); save()
R["full_benchmark"] = bench
names = [a for a in bench[0]["summary"] if all(a in b["summary"] for b in bench)]
R["wilcoxon"] = {a: RB.wilcoxon_qpso_vs(bench, a) for a in RB.ALGO_ORDER if a != "QPSO"}
for a, w in R["wilcoxon"].items():
    if w: L(f"  QPSO vs {a}: mean improvement {w['qpso_mean_improvement_pct']:+.1f}%  wins {w['qpso_lower_pct']:.0f}% of {w['n_pairs']}  p={w['p_value']:.4f}")
R["ablation_quantum_term"] = RB.ablation_quantum_term(bench)
ab = R["ablation_quantum_term"]
L(f"  ABLATION (identical infrastructure, only update rule differs): QPSO better in {ab['n_qpso_better']}/{ab['n_pairs']}, "
  f"mean {ab['pct_improvement_over_classical_pso']:+.2f}%, p={ab['p_value']:.4f}")
dm = RB.dolan_more_profile(bench, names); R["dolan_more"] = dm
plt.figure(figsize=(7.5, 4.6))
for a in names: plt.plot(dm["taus"], dm["profile"][a], label=a, lw=2.2 if a == "QPSO" else 1.2)
plt.xlabel("performance ratio tau (x best)"); plt.ylabel("fraction of instances"); plt.title("Dolan-More profile (held-out set)")
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT}/dolan_more.png", dpi=130); plt.close()
reg = np.array([[b["summary"][a]["gap_to_best_pct"] for a in names] for b in bench])
plt.figure(figsize=(9, 5)); im = plt.imshow(reg, cmap="RdYlGn_r", aspect="auto"); plt.colorbar(im, label="gap to best (%)")
plt.xticks(range(len(names)), names, rotation=30, ha="right", fontsize=7); plt.yticks(range(len(bench)), [b["instance"] for b in bench], fontsize=8)
for i in range(reg.shape[0]):
    for j in range(reg.shape[1]): plt.text(j, i, f"{reg[i, j]:.0f}", ha="center", va="center", fontsize=6)
plt.title("Regime map — gap to best found (%), held-out instances (0 = won)"); plt.tight_layout(); plt.savefig(f"{OUT}/regime_map.png", dpi=130); plt.close()
with open(f"{OUT}/benchmark_table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["instance"] + names)
    for b in bench: w.writerow([b["instance"]] + [f"{b['summary'][a]['mean_fit']:.0f} (+-{b['summary'][a]['std_fit']:.0f})" for a in names])
    w.writerow(["MEAN GAP TO BEST %"] + [f"{np.mean([b['summary'][a]['gap_to_best_pct'] for b in bench]):.1f}" for a in names])
    w.writerow(["MEAN TIME PER RUN (s)"] + [f"{np.mean([b['summary'][a]['mean_time_s'] for b in bench]):.1f}" for a in names])
os.makedirs(f"{OUT}/tuning_logs", exist_ok=True)
for f in glob.glob("/tmp/tune*.log"): shutil.copy(f, f"{OUT}/tuning_logs/")
R["wall_time_s"] = time.time() - t_start; save()
L(f"=== DONE in {R['wall_time_s']:.0f}s ===")
