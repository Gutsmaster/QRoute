"""
PRE-REGISTERED (written before running).
PHASE 1  Harden OR-Tools: choose its configuration on the TUNING instances (Delhi seed0/9am, Solomon-style C/R/RC seed 101), 5 s each:
         5 first-solution strategies x 3 metaheuristics at exact (x100) integer precision, vs the ORIGINAL config. Pick min mean normalised cost.
PHASE 2  Anytime comparison on 8 NEW instances (Delhi Aug-12 3pm seed87; Delhi Aug-19 festival 6pm seed81; Solomon-style seed_base 900), n=24.
         QPSO+WB at 3 iteration levels (25/80/250); memetic GA at matched-time generation counts (60/190/600); OR-Tools (hardened AND original)
         given time limit = QPSO's measured wall-clock at that level. Algorithm seeds 50-52.
         CRITERION (fixed in advance): at the largest level QPSO+WB mean cost < hardened OR-Tools at matched wall-clock on >= 5/8 instances.
Reported as-is. Nothing tuned on the Phase-2 data.
"""
import sys, time, json, math
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "benchmarks")
from data_loader import load_day
from calibration import build_edge_table
from graph_builder import build_graph, largest_component_subgraph
from vrp_instance import make_synthetic_instance
from solomon_synthetic import generate_instance, generate_sweep
from qpso_core import QPSO, evaluate_routes, fitness_weighted
from baselines import GeneticAlgorithm, solve_ortools_cvrptw

BB = (77.205, 28.625, 77.225, 28.645); WB = dict(writeback_mode="sorted_reassign")
GJ = "data/new_delhi_traffic_dataset/probe_counts/geojson/new_delhi__%s_to_%s_.geojson"
def graph(day): return largest_component_subgraph(build_graph(build_edge_table(load_day(GJ % (day, day)))), BB, strongly_connected=True)
Gs, Gf = graph("2024-08-12"), graph("2024-08-19")
def unpack(i, h): return dict(d=[i.demands[c] for c in i.customers], tw=[i.time_windows[c] for c in i.customers], T=i.time_mat[h], D=i.dist_mat[h], E=i.emis_mat[h])
def fit(routes, I): return float(fitness_weighted(evaluate_routes(routes, I["T"], I["D"], I["E"], I["tw"], I["d"], 100)))
def orc(I, secs, cfg):
    r = solve_ortools_cvrptw(24, I["d"], 100, I["tw"], I["T"], I["D"], I["E"], time_limit_s=secs, **cfg)
    return fit(r, I) if r else float("inf")
def sy(s): return dict(d=s["demands"], tw=s["time_windows_s"], T=s["time_mat"], D=s["dist_mat"], E=s["emis_mat"])

# ---------------- PHASE 1
TUNE = {"Delhi(t)": unpack(make_synthetic_instance(Gs, n_customers=24, capacity=100, seed=0), 9)}
for k in ("C", "R", "RC"): TUNE[k + "1(t)"] = sy(generate_instance(k, n_customers=24, seed=101))
CFG = {"ORIG": dict(first_solution="PATH_CHEAPEST_ARC", metaheuristic="GUIDED_LOCAL_SEARCH", scale=1, precise=False)}
for f in ("PATH_CHEAPEST_ARC", "SAVINGS", "PARALLEL_CHEAPEST_INSERTION", "LOCAL_CHEAPEST_INSERTION", "AUTOMATIC"):
    for m in ("GUIDED_LOCAL_SEARCH", "SIMULATED_ANNEALING", "TABU_SEARCH"):
        CFG[f + "|" + m] = dict(first_solution=f, metaheuristic=m, scale=100, precise=True)
cost = {c: {n: orc(I, 5, cfg) for n, I in TUNE.items()} for c, cfg in CFG.items()}
best_per = {n: min(cost[c][n] for c in CFG) for n in TUNE}
norm = {c: np.mean([cost[c][n] / best_per[n] for n in TUNE]) for c in CFG}
print("PHASE 1 - OR-Tools configs on TUNING instances, 5 s (mean cost / best-for-instance; 1.000 = best everywhere)")
for c in sorted(CFG, key=lambda c: norm[c]): print("  %-52s %.4f   %s" % (c, norm[c], "  ".join("%.0f" % cost[c][n] for n in TUNE)))
HARD = CFG[min(CFG, key=lambda c: norm[c])]; print("CHOSEN:", min(CFG, key=lambda c: norm[c]), HARD, flush=True)
json.dump({"chosen": HARD, "norm": norm}, open("/tmp/or_cfg.json", "w"))

# ---------------- PHASE 2
INST = {"Delhi-Aug12-3pm-s87": unpack(make_synthetic_instance(Gs, n_customers=24, capacity=100, seed=87), 15),
        "Delhi-Aug19fest-6pm-s81": unpack(make_synthetic_instance(Gf, n_customers=24, capacity=100, seed=81), 18)}
for s in generate_sweep(n_per_kind=2, n_customers=24, seed_base=900): INST[s["name"]] = sy(s)
LQ, LG, SEEDS = (25, 80, 250), (60, 190, 600), (50, 51, 52); P2 = {}
for name, I in INST.items():
    a = (24, I["d"], 100, I["tw"], I["T"], I["D"], I["E"]); P2[name] = []
    for lq, lg in zip(LQ, LG):
        tq, cq, tg, cg = [], [], [], []
        for sd in SEEDS:
            t = time.time(); r = QPSO(*a, n_particles=24, n_iter=lq, seed=sd, **WB).run(); tq.append(time.time() - t); cq.append(r["gbest_fit"])
            t = time.time(); g = GeneticAlgorithm(*a, pop_size=24, n_gen=lg, seed=sd, local_search=True, lamarck=True).run(); tg.append(time.time() - t); cg.append(g["gbest_fit"])
        T_or = max(1.0, round(float(np.mean(tq)), 1))
        P2[name].append(dict(lq=lq, q=float(np.mean(cq)), tq=float(np.mean(tq)), g=float(np.mean(cg)), tg=float(np.mean(tg)), T_or=T_or,
                             or_hard=orc(I, T_or, HARD), or_orig=orc(I, T_or, CFG["ORIG"])))
    print("done", name, flush=True); json.dump(P2, open("/tmp/anytime.json", "w"))

print("\nPHASE 2 - anytime comparison, n=24 (mean cost; lower = better). OR-Tools given the SAME wall-clock as QPSO at each level.")
for li in range(3):
    print("\n level %d: QPSO %d iters  | GA %d gens" % (li + 1, LQ[li], LG[li]))
    print("  %-26s %9s %7s | %9s %7s | %9s %9s %6s" % ("instance", "QPSO+WB", "t(s)", "GA+LS", "t(s)", "OR-hard", "OR-orig", "T_or"))
    for n, L in P2.items():
        r = L[li]; print("  %-26s %9.0f %7.1f | %9.0f %7.1f | %9.0f %9.0f %6.1f" % (n, r["q"], r["tq"], r["g"], r["tg"], r["or_hard"], r["or_orig"], r["T_or"]))
    qw = sum(L[li]["q"] < L[li]["or_hard"] for L in P2.values()); gw = sum(L[li]["g"] < L[li]["or_hard"] for L in P2.values())
    gap = np.mean([100 * (L[li]["q"] - L[li]["or_hard"]) / L[li]["or_hard"] for L in P2.values()])
    gapg = np.mean([100 * (L[li]["g"] - L[li]["or_hard"]) / L[li]["or_hard"] for L in P2.values()])
    gapd = np.mean([100 * (L[li]["q"] - L[li]["or_hard"]) / L[li]["or_hard"] for n, L in P2.items() if n.startswith("Delhi")])
    print("  QPSO < OR-hard on %d/8 | GA < OR-hard on %d/8 | mean gap QPSO %+.1f%% (Delhi only %+.1f%%), GA %+.1f%%" % (qw, gw, gap, gapd, gapg))
qw3 = sum(L[2]["q"] < L[2]["or_hard"] for L in P2.values())
print("\nPRE-REGISTERED CRITERION (QPSO < hardened OR-Tools at matched time, largest level, >= 5/8): %s (%d/8)" % ("MET" if qw3 >= 5 else "NOT MET", qw3))
