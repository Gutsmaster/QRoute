"""Run:  ./run_dashboard.sh   (or: streamlit run dashboard/app.py)"""
import streamlit as st, json, os, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.path.dirname(HERE), "outputs")
st.set_page_config(page_title="QRoute: Quantum-Inspired Delhi Traffic Router", layout="wide")
R = json.load(open(os.path.join(OUT, "results.json")))
img = lambda n: st.image(os.path.join(OUT, n))
W, AB, FT, AF, DV, RC, bench = R["wilcoxon"], R["ablation_quantum_term"], R["festival_test"], R["archive_fallback"], R["dynamic_value"], R["recovery_demo"], R["full_benchmark"]
OR = "OR-Tools(GLS,5s)"
gaps = [100 * (b["summary"]["QPSO"]["mean_fit"] / b["summary"][OR]["mean_fit"] - 1) for b in bench]
sig = lambda p: "statistically significant" if p < 0.05 else "NOT statistically significant"
GA, PS = W["GA+LS(memetic)"], W["PSO(same infra, tuned)"]
sc = R["scalability"][-1]

st.title("🚦 QRoute: Quantum-Inspired Intelligent Traffic Route Optimization")
st.caption("SIH26137 · QPSO-based dynamic VRP on real Delhi NCR 2024 probe data (Aug 11-30, incl. Rakshabandhan)")
tabs = st.tabs(["Overview", "Live Route Map", "Real-anomaly tests", "Benchmarking", "Convergence & Pareto", "Scalability", "Honesty & protocol"])

with tabs[0]:
    c = st.columns(4)
    c[0].metric("Road segments (dataset)", f"{R['graph_stats']['full_edges']:,}")
    c[1].metric("QPSO vs plain GA", f"{W['GA(plain)']['qpso_mean_improvement_pct']:+.0f}%", "p<0.0001")
    c[2].metric("QPSO vs ACO / tuned LS", f"{W['ACO']['qpso_mean_improvement_pct']:+.0f}% / {W['TunedLS(HGS-inspired)']['qpso_mean_improvement_pct']:+.0f}%")
    c[3].metric("Mean gap to OR-Tools", f"{np.mean(gaps):+.1f}%", f"Delhi instances: {np.mean(gaps[:2]):+.1f}%", delta_color="off")
    st.markdown(f"""
**Held-out results** ({AB['n_pairs']} paired runs, 8 held-out instances (reused once to confirm infrastructure changes), shared local-search stack for all swarm/evolutionary methods):
- QPSO beats plain GA, ACO and tuned local search decisively (Wilcoxon p<0.0001, 100% of runs).
- QPSO vs classical PSO with **identical infrastructure**: {AB['pct_improvement_over_classical_pso']:+.2f}% mean, p={AB['p_value']:.2f} — **{sig(AB['p_value'])}** here (independent confirmation runs measured +0.5% to +2.5%).
- QPSO vs a **memetic GA** (same local search): {GA['qpso_mean_improvement_pct']:+.2f}%, p={GA['p_value']:.2f} — **{sig(GA['p_value'])}**; the two are indistinguishable.
- OR-Tools (best of two configurations, 5 s) remains the reference: QPSO's mean gap is {np.mean(gaps):+.1f}% (range {min(gaps):+.1f}% to {max(gaps):+.1f}%).
""")

with tabs[1]:
    st.subheader("Real Delhi roads — QPSO plan (red) vs OR-Tools plan (green)")
    p = os.path.join(OUT, "map.html")
    if os.path.exists(p): st.iframe(open(p).read(), height=650)

with tabs[2]:
    st.subheader("What the real data shows (these numbers move with optimiser strength; treat as ranges)")
    st.markdown(f"""
- **Rakshabandhan (Aug 19) vs a normal weekday (Aug 12):** a normal-day plan costs **{FT['9']['stale_plan_penalty_pct']:.1f}%** more at 9am and **{FT['18']['stale_plan_penalty_pct']:.1f}%** more at 6pm than re-optimising on the festival day. The best 9am festival-day plan costs **{FT['9']['festival_vs_normal_cost_pct']:+.0f}%** versus the normal day.
- **Never re-routing:** a fixed 9am plan carries a mean regret of **{DV['mean_regret_pct']:.1f}%** across departure hours 6am-9pm.
- **Diverse-plan archive:** after a simulated road closure the incumbent costs {AF['incumbent_after_closure']:.0f}; the best archived alternative costs **{AF['best_archive_plan_after_closure']:.0f} instantly**, versus {AF['reoptimised_from_scratch']:.0f} after a {AF['reoptimise_time_s']:.1f}s full re-optimisation.
""")
    a, b = st.columns(2)
    with a: img("dynamic_value.png")
    with b: img("archive.png")
    st.subheader("Reverse-annealing recovery — NOT supported by the data")
    st.write(f"With recovery: {RC['with_recovery_mean_final']:.0f} · without: {RC['without_recovery_mean_final']:.0f} · recovery better in {RC['with_recovery_wins']}/{RC['n_seeds']} seeds. No benefit; do not claim one.")
    img("recovery.png")

with tabs[3]:
    st.subheader("Held-out benchmark (mean cost, lower is better)")
    st.dataframe(pd.read_csv(os.path.join(OUT, "benchmark_table.csv")), width='stretch')
    a, b = st.columns(2)
    with a: img("regime_map.png")
    with b: img("dolan_more.png")
    st.subheader("Wilcoxon signed-rank: QPSO vs each opponent (paired by instance and seed)")
    st.dataframe(pd.DataFrame(W).T[["qpso_mean_improvement_pct", "qpso_lower_pct", "n_pairs", "p_value"]])
    st.caption("Many pairs are exact ties (all strong methods reach the same plan on several instances), so 'wins %' understates parity.")

with tabs[4]:
    a, b = st.columns(2)
    with a: st.subheader("Convergence (demo instance)"); img("convergence.png")
    with b: st.subheader("Pareto: time / distance / emissions / stability"); img("pareto.png")
    st.caption(f"{R['pareto']['front_size']} non-dominated of {R['pareto']['n_candidates']} candidates. Stability is measured, not directly optimised.")

with tabs[5]:
    img("scalability.png"); st.dataframe(pd.DataFrame(R["scalability"]))
    st.caption(f"With the shared stack, flat QPSO is stronger on quality than hierarchical decomposition (n={sc['n_customers']}: flat {sc['flat_fit']:.0f} in {sc['flat_time_s']:.1f}s vs hierarchical {sc['hier_fit']:.0f} in {sc['hier_time_s']:.1f}s). Hierarchical buys speed, not quality.")

with tabs[6]:
    st.markdown(f"""
**Protocol.** Settings were tuned on a *tuning set*; benchmark numbers come from a *held-out* set (different instances and seeds). Two shared-infrastructure changes (write-back of local-search gains into particle keys; extended inter-route local search) were each confirmed on separate fresh instances with pre-registered hypotheses. Logs: `outputs/tuning_logs/`.

**Real:** road geometry, hourly probe counts, festival/monsoon window. **Derived:** congestion/time via a BPR model fitted to the dataset's rush-hour statistics. **Synthetic:** demand and time windows; Solomon-*style* instances (not the official files). **Approximations:** COPERT-*inspired* emissions; "HGS-inspired" baseline is NN + 2-opt, not real HGS-CVRP; single-hour traffic snapshot per run. **OR-Tools** = better of a hardened and the original configuration, 5 s, single run per instance.

**Not supported — do not claim:** the quantum update is significantly better than classical PSO ({sig(AB['p_value'])} here); QPSO beats a memetic GA or OR-Tools outright; reverse-annealing recovery helps; hierarchical decomposition improves quality over flat QPSO.
""")
