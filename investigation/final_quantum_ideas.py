"""
PRE-REGISTERED SCREEN (written before running; TUNING set only; resumable). Everything on the confirmed stack (write-back + extended local search).
Instances: Delhi seed0/9am + Solomon-style C1,R1,RC1 (seed 101); algorithm seeds 0-5 -> 24 pairs per comparison.
F1 re-test of earlier-rejected ideas on the corrected substrate: scalar-u, elite-mBest(15%), both, Borda-sorted, Borda-mean, rank-jump.
F2 beta schedule sweep (current 0.45->0.1 was chosen on the pre-fix substrate).
F3 TUNNELING KERNEL (shared infra -> applied to BOTH QPSO and classical PSO): relocation partner chosen uniformly / from CTQW |U_ij|^2 (t=1, fixed a priori) /
   from a classical heat kernel whose mean row entropy is matched to the CTQW kernel (control for 'quantum' attribution).
F4 permutation-boundary-aware jump scale (jump multiplied by nearest-key-gap weight, clipped).
DECISION RULE (fixed in advance): a variant is a CANDIDATE only if mean gain >= +1% AND raw p < 0.05 vs its own baseline; candidates (max 2) go to a fresh-instance
confirmation. Holm applied within each family. Nothing is selected on held-out data.
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "benchmarks")
from data_loader import load_day
from calibration import build_edge_table
from graph_builder import build_graph, largest_component_subgraph
from vrp_instance import make_synthetic_instance
from solomon_synthetic import generate_instance
from qpso_core import QPSO
df = build_edge_table(load_day("data/new_delhi_traffic_dataset/probe_counts/geojson/new_delhi__2024-08-12_to_2024-08-12_.geojson"))
G = build_graph(df); Gs = largest_component_subgraph(G, (77.205, 28.625, 77.225, 28.645), strongly_connected=True)
i0 = make_synthetic_instance(Gs, n_customers=24, capacity=100, seed=0)
INST = {"Delhi": dict(d=[i0.demands[c] for c in i0.customers], tw=[i0.time_windows[c] for c in i0.customers], T=i0.time_mat[9], D=i0.dist_mat[9], E=i0.emis_mat[9])}
for k in ("C", "R", "RC"):
    s = generate_instance(k, n_customers=24, seed=101); INST[k + "1"] = dict(d=s["demands"], tw=s["time_windows_s"], T=s["time_mat"], D=s["dist_mat"], E=s["emis_mat"])
STACK = dict(writeback_mode="sorted_reassign", ls_extended=True); SEEDS = tuple(range(6))
V = {"base": {},
     "scalar_u": dict(scalar_u=True), "elite15": dict(elite_frac=0.15), "scalar+elite": dict(scalar_u=True, elite_frac=0.15),
     "borda_sorted": dict(mbest_mode="borda_sorted"), "borda_mean": dict(mbest_mode="borda_mean"), "rank_jump": dict(rank_jump=True),
     "Q reloc_uniform": dict(tunnel_mode="reloc_uniform"), "Q reloc_ctqw": dict(tunnel_mode="reloc_ctqw"), "Q reloc_heat": dict(tunnel_mode="reloc_heat"),
     "P base": dict(update_rule="classical"), "P reloc_uniform": dict(update_rule="classical", tunnel_mode="reloc_uniform"),
     "P reloc_ctqw": dict(update_rule="classical", tunnel_mode="reloc_ctqw"), "P reloc_heat": dict(update_rule="classical", tunnel_mode="reloc_heat"),
     "gap(0.5,3)": dict(jump_gap_scale=True, gap_clip=(0.5, 3.0)), "gap(0.25,4)": dict(jump_gap_scale=True, gap_clip=(0.25, 4.0)),
     "beta0.9-0.3": dict(beta_start=0.9, beta_end=0.3), "beta0.7-0.2": dict(beta_start=0.7, beta_end=0.2), "beta0.6-0.15": dict(beta_start=0.6, beta_end=0.15),
     "beta0.3-0.05": dict(beta_start=0.3, beta_end=0.05), "beta0.2-0.02": dict(beta_start=0.2, beta_end=0.02), "beta1.0-0.5": dict(beta_start=1.0, beta_end=0.5)}
SAVE = "/tmp/ideas.json"; R = json.load(open(SAVE)) if os.path.exists(SAVE) else {}
for v, kw in V.items():                       # priority order: F1, F3, F4, F2
    R.setdefault(v, {})
    for name, I in INST.items():
        if name in R[v]: continue
        a = (24, I["d"], 100, I["tw"], I["T"], I["D"], I["E"])
        R[v][name] = [QPSO(*a, n_particles=24, n_iter=80, seed=sd, **STACK, **kw).run()["gbest_fit"] for sd in SEEDS]
        json.dump(R, open(SAVE, "w"))
    print("variant done:", v, flush=True)
print("ALL VARIANTS COMPLETE", flush=True)
