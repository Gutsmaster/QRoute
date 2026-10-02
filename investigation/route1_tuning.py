"""
ROUTE 1 SCREEN (TUNING set only; nothing here touches held-out data).
Shared-infrastructure change under test: extended inter-route local search (segment relocation to any route, swap, 2-opt*, granular),
triggered only for candidates within 2% of the best-so-far, identical rule for QPSO and the memetic GA.
Equal-ITERATION comparison first (wall-clock reported). Holm over the 2 paired EXT-vs-base tests; QPSO+EXT vs GA+EXT descriptive.
OR-Tools reference = hardened config at 5 s from the hardening log (Delhi 8579, C1 45943, R1 73655, RC1 64080).
"""
import sys, time, json
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, "src"); sys.path.insert(0, "benchmarks")
from data_loader import load_day
from calibration import build_edge_table
from graph_builder import build_graph, largest_component_subgraph
from vrp_instance import make_synthetic_instance
from solomon_synthetic import generate_instance
from qpso_core import QPSO
from baselines import GeneticAlgorithm
df = build_edge_table(load_day("data/new_delhi_traffic_dataset/probe_counts/geojson/new_delhi__2024-08-12_to_2024-08-12_.geojson"))
G = build_graph(df); Gs = largest_component_subgraph(G, (77.205, 28.625, 77.225, 28.645), strongly_connected=True)
i0 = make_synthetic_instance(Gs, n_customers=24, capacity=100, seed=0)
INST = {"Delhi": dict(d=[i0.demands[c] for c in i0.customers], tw=[i0.time_windows[c] for c in i0.customers], T=i0.time_mat[9], D=i0.dist_mat[9], E=i0.emis_mat[9])}
for k in ("C", "R", "RC"):
    s = generate_instance(k, n_customers=24, seed=101); INST[k + "1"] = dict(d=s["demands"], tw=s["time_windows_s"], T=s["time_mat"], D=s["dist_mat"], E=s["emis_mat"])
OR = {"Delhi": 8579, "C1": 45943, "R1": 73655, "RC1": 64080}
WB = dict(writeback_mode="sorted_reassign"); SEEDS = tuple(range(6))
V = {"Q+WB":      lambda a, sd: QPSO(*a, n_particles=24, n_iter=80, seed=sd, **WB),
     "Q+WB+EXT":  lambda a, sd: QPSO(*a, n_particles=24, n_iter=80, seed=sd, ls_extended=True, **WB),
     "GA+LS":     lambda a, sd: GeneticAlgorithm(*a, pop_size=24, n_gen=80, seed=sd, local_search=True, lamarck=True),
     "GA+LS+EXT": lambda a, sd: GeneticAlgorithm(*a, pop_size=24, n_gen=80, seed=sd, local_search=True, lamarck=True, ls_extended=True)}
R = {v: {k: [] for k in INST} for v in V}; TM = {v: [] for v in V}
for name, I in INST.items():
    a = (24, I["d"], 100, I["tw"], I["T"], I["D"], I["E"])
    for sd in SEEDS:
        for v, f in V.items():
            t = time.time(); R[v][name].append(f(a, sd).run()["gbest_fit"]); TM[v].append(time.time() - t)
    print("done", name, flush=True); json.dump({"R": R, "TM": TM}, open("/tmp/route1.json", "w"))
names = list(INST)
print("\nMEAN COST per tuning instance (lower = better) | mean seconds per run | gap to hardened OR-Tools@5s")
print("%-12s" % "variant" + "".join("%9s" % n for n in names) + "   sec   gapOR%")
for v in V:
    row = [np.mean(R[v][n]) for n in names]
    print("%-12s" % v + "".join("%9.0f" % x for x in row) + "  %4.1f  %+6.1f" % (np.mean(TM[v]), np.mean([100 * (r - OR[n]) / OR[n] for r, n in zip(row, names)])))
print("%-12s" % "OR hard 5s" + "".join("%9.0f" % OR[n] for n in names))
def flat(v): return np.array([R[v][n][i] for n in names for i in range(len(SEEDS))])
def cmp(a, b):
    x, y = flat(a), flat(b); return (x < y).sum(), len(x), 100 * np.mean((y - x) / y), wilcoxon(x, y)[1]
tests = [("Q+WB+EXT vs Q+WB", "Q+WB+EXT", "Q+WB"), ("GA+LS+EXT vs GA+LS", "GA+LS+EXT", "GA+LS")]
out = [cmp(a, b) for _, a, b in tests]; ps = np.array([o[3] for o in out]); o = np.argsort(ps); adj = np.empty(2); run = 0
for r, i in enumerate(o): run = max(run, (2 - r) * ps[i]); adj[i] = min(1, run)
print("\nEXT vs base (24 pairs; positive = EXT better):")
for (lab, _, _), (w, n, rel, p), ap in zip(tests, out, adj): print("  %-22s wins %2d/%d  mean %+6.2f%%  p=%.3f  Holm-p=%.3f" % (lab, w, n, rel, p, ap))
w, n, rel, p = cmp("Q+WB+EXT", "GA+LS+EXT"); print("descriptive: Q+WB+EXT vs GA+LS+EXT wins %d/%d mean %+.2f%% p=%.3f" % (w, n, rel, p))
w, n, rel, p = cmp("Q+WB", "GA+LS"); print("descriptive: Q+WB     vs GA+LS     wins %d/%d mean %+.2f%% p=%.3f" % (w, n, rel, p))
