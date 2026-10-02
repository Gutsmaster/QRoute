import sys, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "src"); sys.path.insert(0, "benchmarks")
from data_loader import load_day
from calibration import build_edge_table
from graph_builder import build_graph, largest_component_subgraph
from vrp_instance import make_synthetic_instance
from solomon_synthetic import generate_instance
from qpso_core import QPSO

df = build_edge_table(load_day("data/new_delhi_traffic_dataset/probe_counts/geojson/new_delhi__2024-08-12_to_2024-08-12_.geojson"))
G = build_graph(df); Gs = largest_component_subgraph(G, (77.205, 28.625, 77.225, 28.645), strongly_connected=True)
inst = make_synthetic_instance(Gs, n_customers=24, capacity=100, seed=0)   # TUNING instance
INST = {"Delhi(tune)": dict(d=[inst.demands[c] for c in inst.customers], tw=[inst.time_windows[c] for c in inst.customers],
                            T=inst.time_mat[9], D=inst.dist_mat[9], E=inst.emis_mat[9])}
s = generate_instance("R", n_customers=24, seed=101)                       # TUNING instance
INST["R1(tune)"] = dict(d=s["demands"], tw=s["time_windows_s"], T=s["time_mat"], D=s["dist_mat"], E=s["emis_mat"])

def run(I, seed, **kw):
    q = QPSO(24, I["d"], 100, I["tw"], I["T"], I["D"], I["E"], n_particles=24, n_iter=80, seed=seed, diag=True, **kw)
    r = q.run(); return r["gbest_fit"], q.diag

configs = [("quantum  tunnel=0  ", dict(update_rule="quantum", tunnel_prob=0.0)),
           ("classical tunnel=0  ", dict(update_rule="classical", tunnel_prob=0.0)),
           ("quantum  tunnel=.5 ", dict(update_rule="quantum", tunnel_prob=0.5)),
           ("classical tunnel=.5 ", dict(update_rule="classical", tunnel_prob=0.5))]
print("Aggregated over 2 tuning instances x 2 seeds (24 particles x 80 iters)\n")
print("%-21s %8s | %9s %10s | %9s %9s %9s | %9s | %8s %8s %8s" % (
      "config", "fit", "LS_impr%", "desyncKT", "nat_chg%", "nat_KT", "rho(d,KT)", "total_KT", "mbStd@e", "mbStd@l", "mbDev"))
for name, kw in configs:
    agg = {}; fits = []
    for iname, I in INST.items():
        for seed in (0, 1):
            f, dg = run(I, seed, **kw); fits.append(f)
            for k, v in dg.items(): agg.setdefault(k, []).extend(v)
    a = {k: np.array(v) for k, v in agg.items()}
    rho = spearmanr(a["native_dist"], a["native_kendall"])[0]
    ms = a.get("mbest_std")
    mb_e = ms[:240].mean() if ms is not None else float("nan")      # first ~10 iterations worth of entries
    mb_l = ms[-240:].mean() if ms is not None else float("nan")
    print("%-21s %8.0f | %8.1f%% %10.3f | %8.1f%% %9.4f %9.3f | %9.4f | %8.3f %8.3f %8.3f" % (
          name, np.mean(fits), 100 * a["ls_improved"].mean(), a["desync_kendall"].mean(),
          100 * a["native_changed"].mean(), a["native_kendall"].mean(), rho, a["total_kendall"].mean(),
          mb_e, mb_l, a["mbest_absdev"].mean() if "mbest_absdev" in a else float("nan")))
print("\nLegend: LS_impr% = share of particle-evals where local search improved the decoded plan;")
print("desyncKT = normalised Kendall distance between the keys' own permutation (what pbest stores) and the LS-improved plan that earned the stored cost;")
print("nat_chg% = share of native (pre-tunnel) updates that change the decoded permutation at all; nat_KT = mean Kendall change of native update;")
print("rho = Spearman corr between key-space distance moved and Kendall change; total_KT = mean Kendall change incl. tunneling;")
print("mbStd = std of mbest across customers (0 => flat vector), early/late; mbDev = mean |mbest - x|.")
