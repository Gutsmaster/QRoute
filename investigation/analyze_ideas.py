import json, sys, numpy as np
from scipy.stats import wilcoxon
R = json.load(open("/tmp/ideas.json")); INST = ["Delhi", "C1", "R1", "RC1"]
def ok(v): return v in R and all(i in R[v] and len(R[v][i]) == 6 for i in INST)
def flat(v): return np.array([x for i in INST for x in R[v][i]])
def cmp(a, b):
    x, y = flat(a), flat(b); rel = 100 * (y - x) / y; nz = np.abs(x - y) > 1e-6
    p = wilcoxon(x, y)[1] if nz.sum() > 5 else 1.0
    return int((x < y - 1e-6).sum()), int((x > y + 1e-6).sum()), int((~nz).sum()), rel.mean(), p
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj
def family(title, pairs):
    pairs = [(a, b) for a, b in pairs if ok(a) and ok(b)]
    if not pairs: print(title, "- no complete variants yet"); return
    out = [cmp(a, b) for a, b in pairs]; adj = holm(np.array([o[4] for o in out]))
    print("\n" + title)
    for (a, b), (w, l, t, rel, p), ap in zip(pairs, out, adj):
        flag = "  <-- CANDIDATE" if rel >= 1.0 and p < 0.05 else ""
        print("  %-18s vs %-16s wins %2d losses %2d ties %2d  mean %+6.2f%%  p=%.3f  Holm=%.3f%s" % (a, b, w, l, t, rel, p, ap, flag))
family("F1  earlier-rejected ideas, re-tested (vs Q base; positive = better)", [(v, "base") for v in ("scalar_u", "elite15", "scalar+elite", "borda_sorted", "borda_mean", "rank_jump")])
family("F2  beta schedules (vs Q base, base = 0.45->0.1)", [(v, "base") for v in ("beta0.9-0.3", "beta0.7-0.2", "beta0.6-0.15", "beta0.3-0.05", "beta0.2-0.02", "beta1.0-0.5")])
family("F3a tunneling kernel vs swap tunneling (shared infra)", [("Q reloc_uniform", "base"), ("Q reloc_ctqw", "base"), ("Q reloc_heat", "base"),
        ("P reloc_uniform", "P base"), ("P reloc_ctqw", "P base"), ("P reloc_heat", "P base")])
family("F3b attribution controls: CTQW vs heat(matched entropy) vs uniform, same move shape", [("Q reloc_ctqw", "Q reloc_heat"), ("Q reloc_ctqw", "Q reloc_uniform"),
        ("P reloc_ctqw", "P reloc_heat"), ("P reloc_ctqw", "P reloc_uniform")])
family("F3c quantum vs classical update under each tunneling mode", [("base", "P base"), ("Q reloc_uniform", "P reloc_uniform"), ("Q reloc_ctqw", "P reloc_ctqw"), ("Q reloc_heat", "P reloc_heat")])
family("F4  permutation-boundary-aware jump (vs Q base)", [("gap(0.5,3)", "base"), ("gap(0.25,4)", "base")])
print("\ncomplete variants: %d/%d" % (sum(ok(v) for v in R), 22))
