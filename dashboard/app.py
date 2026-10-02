"""Run:  ./run_dashboard.sh   (or: streamlit run dashboard/app.py)"""
import streamlit as st, json, os, pandas as pd
import streamlit.components.v1 as components
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.path.dirname(HERE), "outputs")
st.set_page_config(page_title="Quantum-Inspired Delhi Traffic Router", layout="wide")
R = json.load(open(os.path.join(OUT, "results.json")))
img = lambda n: st.image(os.path.join(OUT, n))
W, AB, FT = R["wilcoxon"], R["ablation_quantum_term"], R["festival_test"]
AF, DV, RC = R["archive_fallback"], R["dynamic_value"], R["recovery_demo"]
bench = R["full_benchmark"]
gaps = [100 * (b["summary"]["QPSO"]["mean_fit"] / b["summary"]["OR-Tools(GLS,5s)"]["mean_fit"] - 1) for b in bench]

st.title("🚦 Quantum-Inspired Intelligent Traffic Route Optimization")
st.caption("SIH26137 · QPSO-based dynamic VRP on real Delhi NCR 2024 probe data (Aug 11-30, incl. Rakshabandhan)")

tabs = st.tabs(["Overview", "Live Route Map", "Real-anomaly tests", "Benchmarking", "Convergence & Pareto", "Scalability", "Honesty & protocol"])

with tabs[0]:
    c = st.columns(4)
    c[0].metric("Road segments (dataset)", f"{R['graph_stats']['full_edges']:,}")
    c[1].metric("QPSO vs first-pass QPSO", f"{W['QPSO-Basic(first pass)']['qpso_mean_improvement_pct']:+.0f}%", "p<0.0001")
    c[2].metric("QPSO vs plain GA / TunedLS", f"{W['GA(plain)']['qpso_mean_improvement_pct']:+.0f}% / {W['TunedLS(HGS-inspired)']['qpso_mean_improvement_pct']:+.0f}%")
    c[3].metric("Mean gap to OR-Tools (5s)", f"{sum(gaps)/len(gaps):.1f}%", f"Delhi instances: {sum(gaps[:2])/2:.1f}%", delta_color="off")
    st.markdown(f"""
**Held-out results** ({AB['n_pairs']} paired runs, 8 instances never used for tuning):
- QPSO beats plain GA, ACO and tuned local search significantly (Wilcoxon p<0.0001).
- QPSO vs classical PSO with **identical infrastructure**: better in {AB['n_qpso_better']}/{AB['n_pairs']} runs, mean **{AB['pct_improvement_over_classical_pso']:+.1f}%**, **p={AB['p_value']:.2f}** (directional, not statistically significant).
- A **memetic GA** (same local search) is significantly better than QPSO: {W['GA+LS(memetic)']['qpso_mean_improvement_pct']:+.1f}%, p={W['GA+LS(memetic)']['p_value']:.3f}. OR-Tools is best overall.
""")

with tabs[1]:
    st.subheader("Real Delhi roads — QPSO plan (red) vs OR-Tools plan (green)")
    p = os.path.join(OUT, "map.html")
    if os.path.exists(p): components.html(open(p, encoding="utf-8").read(), height=650)

with tabs[2]:
    st.subheader("What the real data shows")
    st.markdown(f"""
- **Rakshabandhan (Aug 19) vs a normal weekday (Aug 12):** a plan optimised for the normal day costs **{FT['9']['stale_plan_penalty_pct']:.1f}%** more at 9am and **{FT['18']['stale_plan_penalty_pct']:.1f}%** more at 6pm than re-optimising on the festival day. At 9am the best achievable festival-day plan costs **{FT['9']['festival_vs_normal_cost_pct']:+.0f}%** versus the normal day.
- **Never re-routing:** a fixed 9am plan carries a mean regret of **{DV['mean_regret_pct']:.1f}%** across departure hours 6am-9pm.
- **Diverse-plan archive:** after a simulated road closure the incumbent plan costs {AF['incumbent_after_closure']:.0f}; the best archived alternative costs **{AF['best_archive_plan_after_closure']:.0f} instantly**, versus {AF['reoptimised_from_scratch']:.0f} after a {AF['reoptimise_time_s']:.1f}s full re-optimisation.
""")
    a, b = st.columns(2)
    with a: img("dynamic_value.png")
    with b: img("archive.png")
    st.subheader("Reverse-annealing recovery — NOT supported by the data")
    st.write(f"With recovery: {RC['with_recovery_mean_final']:.0f} · without: {RC['without_recovery_mean_final']:.0f} · recovery better in {RC['with_recovery_wins']}/{RC['n_seeds']} seeds. No measurable benefit; do not claim one.")
    img("recovery.png")

with tabs[3]:
    st.subheader("Held-out benchmark (mean cost, lower is better)")
    st.dataframe(pd.read_csv(os.path.join(OUT, "benchmark_table.csv")), width='stretch')
    a, b = st.columns(2)
    with a: img("regime_map.png")
    with b: img("dolan_more.png")
    st.subheader("Wilcoxon signed-rank: QPSO vs each opponent (paired by instance and seed)")
    st.dataframe(pd.DataFrame(W).T[["qpso_mean_improvement_pct", "qpso_lower_pct", "n_pairs", "p_value"]])

with tabs[4]:
    a, b = st.columns(2)
    with a: st.subheader("Convergence (demo instance)"); img("convergence.png")
    with b: st.subheader("Pareto: time / distance / emissions / stability"); img("pareto.png")
    st.caption(f"{R['pareto']['front_size']} non-dominated of {R['pareto']['n_candidates']} candidates. Stability is measured, not directly optimised.")

with tabs[5]:
    img("scalability.png"); st.dataframe(pd.DataFrame(R["scalability"]))

with tabs[6]:
    st.markdown("""
**Protocol.** QPSO, classical-PSO and memetic-GA settings were tuned on a *tuning set* (Delhi seed-0 + Solomon-style seed-101 instances, seeds 0-3; logs in `outputs/tuning_logs/`). All benchmark numbers come from a *held-out* set (different instances and seeds).

**Real:** road geometry, hourly probe counts, festival/monsoon window. **Derived:** congestion/time via a BPR model fitted to the dataset's own rush-hour statistics. **Synthetic:** customer demand and time windows; Solomon-*style* instances (not the official files). **Approximations:** emissions are a COPERT-*inspired* curve; "HGS-inspired" baseline is NN + 2-opt, not real HGS-CVRP; each run uses a single-hour traffic snapshot.

**Not supported / weak:** the quantum update rule is not significantly better than classical PSO (p≈0.10); a memetic GA beats QPSO by ~3%; OR-Tools beats QPSO; reverse-annealing recovery showed no benefit; the Pareto front is small because time, distance and emissions are strongly correlated.
""")
