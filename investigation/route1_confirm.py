"""
PRE-REGISTERED CONFIRMATION of Route 1 (written before running). Resumable.
Change: extended inter-route local search (ls_extended), shared infrastructure, applied identically to QPSO, classical PSO and the memetic GA.
NEW data: Delhi Aug-12 9am seed111; Delhi Aug-19 (festival) noon seed117; Solomon-style seed_base 1100 (6). n=24. Algorithm seeds 60-65 -> 48 pairs.
Hypotheses (two-sided paired Wilcoxon, Holm over 4):
  H1 QPSO+WB+EXT vs QPSO+WB    H2 PSO+WB+EXT vs PSO+WB    H3 QPSO+WB+EXT vs PSO+WB+EXT    H4 QPSO+WB+EXT vs GA+LS+EXT
OR-Tools (hardened config chosen on tuning data; and the original config) is given a time limit equal to QPSO+EXT's measured wall-clock, plus 10 s.
OR comparison is descriptive: instance-mean QPSO+EXT vs OR, wins out of 8 (OR is a single run per instance).
"""
import sys, os, time, json
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, "src"); sys.path.insert(0, "benchmarks")
from data_loader import load_day
from calibration import build_edge_table
from graph_builder import build_graph, largest_component_subgraph
from vrp_instance import make_synthetic_instance
from solomon_synthetic import generate_sweep
from qpso_core import QPSO, evaluate_routes, fitness_weighted
from baselines import GeneticAlgorithm, solve_ortools_cvrptw
BB = (77.205, 28.625, 77.225, 28.645); WB = dict(writeback_mode="sorted_reassign")
GJ = "data/new_delhi_traffic_dataset/probe_counts/geojson/new_delhi__%s_to_%s_.geojson"
def graph(day): return largest_component_subgraph(build_graph(build_edge_table(load_day(GJ % (day, day)))), BB, strongly_connected=True)
Gs, Gf = graph("2024-08-12"), graph("2024-08-19")
def unpack(i, h): return dict(d=[i.demands[c] for c in i.customers], tw=[i.time_windows[c] for c in i.customers], T=i.time_mat[h], D=i.dist_mat[h], E=i.emis_mat[h])
INST = {"Delhi-Aug12-9am-s111": unpack(make_synthetic_instance(Gs, n_customers=24, capacity=100, seed=111), 9),
        "Delhi-Aug19fest-noon-s117": unpack(make_synthetic_instance(Gf, n_customers=24, capacity=100, seed=117), 12)}
for s in generate_sweep(n_per_kind=2, n_customers=24, seed_base=1100): INST[s["name"]] = dict(d=s["demands"], tw=s["time_windows_s"], T=s["time_mat"], D=s["dist_mat"], E=s["emis_mat"])
HARD = dict(first_solution="LOCAL_CHEAPEST_INSERTION", metaheuristic="GUIDED_LOCAL_SEARCH", scale=100, precise=True)
ORIG = dict(first_solution="PATH_CHEAPEST_ARC", metaheuristic="GUIDED_LOCAL_SEARCH", scale=1, precise=False)
def fit(r, I): return float(fitness_weighted(evaluate_routes(r, I["T"], I["D"], I["E"], I["tw"], I["d"], 100)))
def orc(I, secs, cfg):
    r = solve_ortools_cvrptw(24, I["d"], 100, I["tw"], I["T"], I["D"], I["E"], time_limit_s=secs, **cfg); return fit(r, I) if r else float("inf")
SEEDS = tuple(range(60, 66))
def algos(a, sd):
    return {"Q+WB": lambda: QPSO(*a, n_particles=24, n_iter=80, seed=sd, **WB),
            "Q+WB+EXT": lambda: QPSO(*a, n_particles=24, n_iter=80, seed=sd, ls_extended=True, **WB),
            "P+WB": lambda: QPSO(*a, n_particles=24, n_iter=80, seed=sd, update_rule="classical", **WB),
            "P+WB+EXT": lambda: QPSO(*a, n_particles=24, n_iter=80, seed=sd, update_rule="classical", ls_extended=True, **WB),
            "GA+LS": lambda: GeneticAlgorithm(*a, pop_size=24, n_gen=80, seed=sd, local_search=True, lamarck=True),
            "GA+LS+EXT": lambda: GeneticAlgorithm(*a, pop_size=24, n_gen=80, seed=sd, local_search=True, lamarck=True, ls_extended=True)}
NAMES = list(algos((24, [], 100, [], None, None, None), 0).keys())
SAVE = "/tmp/route1_confirm.json"; R = {n: {} for n in NAMES}; TM = {n: {} for n in NAMES}; ORR = {}
if os.path.exists(SAVE):
    d = json.load(open(SAVE)); R, TM, ORR = d["R"], d["TM"], d["OR"]; print("resumed; complete:", list(ORR), flush=True)
for name, I in INST.items():
    if name in ORR: continue
    a = (24, I["d"], 100, I["tw"], I["T"], I["D"], I["E"])
    for n in NAMES: R[n][name] = []; TM[n][name] = []
    for sd in SEEDS:
        for n, f in algos(a, sd).items():
            t = time.time(); R[n][name].append(f().run()["gbest_fit"]); TM[n][name].append(time.time() - t)
    T = max(1.0, round(float(np.mean(TM["Q+WB+EXT"][name])), 1))
    ORR[name] = dict(T=T, hard=orc(I, T, HARD), orig=orc(I, T, ORIG), hard10=orc(I, 10, HARD))
    print("done", name, flush=True); json.dump({"R": R, "TM": TM, "OR": ORR}, open(SAVE, "w"))
inames = list(INST)
print("\nMEAN COST per NEW instance (lower = better); seconds = mean per run")
print("%-11s" % "algorithm" + "".join("%13s" % n[:12] for n in inames) + "   sec")
for n in NAMES: print("%-11s" % n + "".join("%13.0f" % np.mean(R[n][i]) for i in inames) + "  %4.1f" % np.mean([t for i in inames for t in TM[n][i]]))
for lab, k in (("OR hard@T", "hard"), ("OR orig@T", "orig"), ("OR hard@10s", "hard10")): print("%-11s" % lab + "".join("%13.0f" % ORR[i][k] for i in inames))
def flat(n): return np.array([R[n][i][s] for i in inames for s in range(len(SEEDS))])
def cmp(a, b):
    x, y = flat(a), flat(b); return (x < y).sum(), len(x), 100 * np.mean((y - x) / y), wilcoxon(x, y)[1]
H = [("H1 QPSO+EXT vs QPSO", "Q+WB+EXT", "Q+WB"), ("H2 PSO+EXT vs PSO", "P+WB+EXT", "P+WB"), ("H3 QPSO+EXT vs PSO+EXT", "Q+WB+EXT", "P+WB+EXT"), ("H4 QPSO+EXT vs GA+EXT", "Q+WB+EXT", "GA+LS+EXT")]
out = [cmp(a, b) for _, a, b in H]; ps = np.array([o[3] for o in out]); o = np.argsort(ps); adj = np.empty(4); run = 0
for r, i in enumerate(o): run = max(run, (4 - r) * ps[i]); adj[i] = min(1, run)
print("\nPRE-REGISTERED TESTS (48 pairs; positive = first better)")
for (lab, _, _), (w, n, rel, p), ap in zip(H, out, adj): print("  %-26s wins %2d/%d  mean %+6.2f%%  p=%.4f  Holm-p=%.4f" % (lab, w, n, rel, p, ap))
print("\nvs OR-Tools (instance means; OR single run):")
for lab, n in (("QPSO+EXT", "Q+WB+EXT"), ("GA+EXT", "GA+LS+EXT"), ("PSO+EXT", "P+WB+EXT"), ("QPSO (no EXT)", "Q+WB")):
    m = np.array([np.mean(R[n][i]) for i in inames]); h = np.array([ORR[i]["hard"] for i in inames]); og = np.array([ORR[i]["orig"] for i in inames]); b = np.minimum(h, og); h10 = np.array([ORR[i]["hard10"] for i in inames])
    print("  %-14s < OR-hard@T on %d/8 (mean gap %+.1f%%) | < best(hard,orig)@T on %d/8 (%+.1f%%) | vs OR-hard@10s %+.1f%% (%d/8) | Delhi-only gap vs hard@T %+.1f%%" % (
          lab, (m < h).sum(), np.mean(100 * (m - h) / h), (m < b).sum(), np.mean(100 * (m - b) / b), np.mean(100 * (m - h10) / h10), (m < h10).sum(), np.mean((100 * (m - h) / h)[:2])))
