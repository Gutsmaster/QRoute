# QPSO push investigation (post-benchmark) — all on TUNING set only, held-out untouched

Goal: try to make QPSO beat Memetic GA / close gap to OR-Tools further, per explicit
request, without touching the frozen held-out benchmark or its protocol.

Diagnostic finding: without tunneling, delta-well update produces LESS permutation
diversity (Kendall-dist 0.060) than classical PSO's own dynamics (0.086), despite
similar continuous-space distance (~0.25-0.44) -- confirms "diversity in the wrong
space." Tunneling closes this asymmetrically: it helps quantum much more (9012->8635)
than classical (8943->8923).

Five structurally distinct hypotheses tested, all on tuning set (Delhi seed-0 +
Solomon-style seed-101, algorithm seeds 0-3, NP=24, NI=80):

1. Gemini's scalar-u:        WORSE  (+3.7% vs GA, baseline +2.0%)
2. Gemini's elite-mBest:     WORSE  (+4.0%)
3. Both combined:            WORSE  (+3.9%)
4. Rank-space jump (own hypothesis, motivated by diagnostic): NEUTRAL/WORSE
   (+3.0% with tunneling, +6.5% vs +6.8% baseline without tunneling)
5. Rank-jump x scalar-u x elite combos: ALL WORSE (+2.6% to +3.5%)
6. Stochastic multi-elite attractor (per-particle random elite instead of
   shared gbest): NEUTRAL AT BEST (+2.3% at k=.25, worse elsewhere); same
   null result when applied to classical PSO too
7. Local-search frequency/elite-targeting (every-2nd/3rd iter, top 30-50%
   only): WORSE in every configuration for BOTH quantum and classical --
   full local search on every particle every iteration remains optimal
8. Diversity-triggered beta boost (vs. Gemini's rejected stagnation-triggered
   version): INCONCLUSIVE -- never triggered at floor 0.08/0.15/0.25 because
   tunnel_prob=0.5 already keeps the cheap diversity proxy above threshold
   throughout the run. Would need a much stricter floor or a different
   proxy to actually test the hypothesis; not pursued further given time.

CONCLUSION: the current baseline configuration (vector-u, full-swarm mean
mBest, global gbest attractor, raw-key delta-well jump, tunnel_prob=0.5,
full local search every iteration on every particle) remains the best
QPSO configuration found across ~20 tested variants. No tested mechanism
beat it. This is reported as an honest negative result, not a proof that
no better configuration exists -- see recommendations in the main writeup
for what was NOT tried (efficiency/vectorization for larger fair-budget
comparison; genuine hybrid recombination, explicitly deferred by design).

Verified: QPSO()'s default behavior is UNCHANGED (bit-identical, seed=1,
Delhi instance: 8675.2198...) despite all new optional parameters added
during this investigation -- the frozen held-out benchmark numbers in
results.json/README remain valid.

---

# Bounded hybrid attempt (post-"ceiling" walkback, per explicit go-ahead)

Per user instruction, tested a genuine quantum-inspired hybrid: sparse
crossover supplement to the QPSO swarm (crossover replaces only the
worst `crossover_frac` fraction of particles each iteration, bred from
randomly paired ELITE pbest particles; the majority of the swarm keeps
doing normal QPSO dynamics untouched -- this is a hybrid, not a
GA-replacement). Two operators tested:
  - "order": standard GA order-crossover (OX) on the decoded giant tour
  - "quantum_blend": continuous interpolation child = cos^2(theta)*a +
    sin^2(theta)*b, theta ~ U(0, pi/2) per dimension -- a qubit-rotation-
    style blend, staying in continuous key space (mechanistically
    distinct from GA's discrete splicing)

ROUND 1 (4 seeds, coarse sweep): order-crossover made things worse at
every fraction tested (+4.1%, +3.1% gap vs GA, baseline +2.0%).
quantum_blend at frac=.15 showed a small apparent improvement (+1.8%
vs baseline's +2.0%) -- the first "win" found in the entire investigation.

ROUND 2 (8 seeds, refinement + direct paired significance test): the
apparent win did NOT reproduce. Baseline gap moved to +3.1% under the
wider seed set (confirming substantial seed variance), qblend frac=.10
showed +2.2% (nominally best of the crossover variants tested), but the
direct paired Wilcoxon test on the SAME 32 instance x seed pairs (qblend
frac=.15 vs baseline) gave: better in 17/32 (53%), mean improvement only
123 (negligible relative to fitness values of 50,000-80,000), p=0.69.
NOT statistically significant -- indistinguishable from noise.

CONCLUSION: per the stated protocol ("only promising changes go to
held-out evaluation"), this result does NOT qualify as promising and
was NOT escalated to the held-out set -- doing so would have spent
held-out budget testing something already shown to be noise on the
tuning set. The crossover hybrid, in this form, does not produce a
real, reproducible improvement.

Total experimental scope across both investigation sessions: ~9 distinct
mechanisms (5 pure-QPSO-mechanism changes + 4 hybrid crossover
configurations), all tested with paired seeds against identical
infrastructure, all converging to the same honest result: the current
baseline configuration is the best found, and no tested modification
(quantum-only or hybrid) produces a statistically robust improvement
over it on the tuning set.

Re-verified: QPSO()'s default behavior remains bit-identical
(seed=1, Delhi instance: 8675.2198...) -- the frozen held-out benchmark
in results.json/README is untouched and still valid.

---

# Round 3: decoder-mismatch diagnostics, write-back, Borda mbest, renormalization (+ pre-registered held-out confirmation)

Scripts: investigation/step1_diag.py, step23_exp.py, confirm_writeback.py (run from project root).
Logs: step1_diagnostics.log, step23_writeback_borda_renorm.log, confirmatory_heldout_writeback.log.
New QPSO options (all default OFF; default run verified bit-identical, seed=1 Delhi = 8675.2198...):
writeback_mode="sorted_reassign", mbest_mode in {"borda_sorted","borda_mean"}, renormalize, diag.

## Step 1 - premises from four external AI proposals checked against measurements (tuning set)
- pbest keys desynced from the plan that earned their cost: CONFIRMED (local search improves 100% of evaluations;
  key-permutation vs scored-plan normalised Kendall distance ~0.11) because Baldwinian local search never writes gains to keys.
- "native updates are often permutation-neutral": CONTRADICTED (98.7% change the permutation).
- "key distance decoupled from permutation change": CONTRADICTED (Spearman 0.92-0.94).
- "mbest collapses to a flat 0.5 vector": CONTRADICTED (std across customers ~0.19-0.22; spread vector = 0.29, flat = 0).
- mean |mbest - x| ~ 0.04 (~1 key spacing at n=24): the 'quantum jump' itself is small; pbest/gbest blend dominates movement (inference).

## Steps 2-3 - tuning set, 4 instances x 6 seeds = 24 pairs, paired Wilcoxon, Holm within family
- Write-back (keep particle's own key values, reassign to match improved tour): Q +2.64% (16/24, p=.027, Holm .16);
  PSO +3.81% (19/24, p=.001, Holm .009).
- Borda mbest: Q -2.07% (sorted), -0.58% (mean) -> no benefit.   Renormalization: Q -1.37%, PSO -0.51% -> no benefit.
- Quantum vs classical (same infra): baseline +1.82% (p=.021, Holm .063); with write-back +0.67% (p=.115).
  TUNING-SET PREDICTION: write-back narrows the quantum-vs-classical gap. (Not borne out on held-out; see below.)

## Pre-registered confirmatory test - held-out instances, FRESH seeds 14-19, 48 paired runs, Holm over 4 hypotheses
Design fixed before running; run was interrupted once after 4 instances and RESUMED unchanged from saved partial results.
- H1 QPSO+WB vs QPSO(frozen):            +5.96%, 44/48, Holm p < 0.0001
- H2 PSO+WB  vs PSO(frozen):             +2.81%, 34/48, Holm p = 0.0004
- H3 QPSO+WB vs PSO+WB (same infra):     +2.54%, 36/48, Holm p = 0.0008   <- quantum update now significantly ahead
- H4 QPSO+WB vs memetic GA:              +1.29%, 25/48, Holm p = 0.154    <- not significantly different (NOT 'beats')
- Descriptive: vs plain GA +17.1% (48/48), ACO +36.7% (48/48), TunedLS +28.9% (47/48).
- Mean gap to OR-Tools (5s, same objective): +2.7% (per instance: -0.1, +1.6, +8.9, +2.6, +1.3, +1.3, +2.1, +3.8).
- Sanity: frozen QPSO / memetic GA reproduce their earlier gaps on fresh seeds (9.6% vs 8.9%; 4.2% vs 4.6%).

## Caveats that must accompany any claim
1. These held-out INSTANCES were already seen once (frozen benchmark). Change selected on tuning data, fresh seeds, pre-registered,
   but a third check on genuinely new instances is advisable before changing slide claims.
2. Neither QPSO nor PSO hyperparameters were re-tuned after write-back (both tuned under the old infrastructure) - symmetric, but open.
3. Tuning-set (24 pairs) and held-out (48 pairs) disagree on the size of the quantum-vs-classical gap (+0.7% vs +2.5%): true effect likely
   positive but its magnitude is uncertain.
4. H4 is a failure to reject, not evidence of equivalence. QPSO+WB mean cost is lower than memetic GA on 7/8 instances but per-run paired
   wins are 25/48. Language must be 'statistically indistinguishable from', never 'outperforms'.
5. Write-back is shared infrastructure (applied to QPSO and PSO; the GA already used Lamarckian write-back).

---

# Round 4: pre-registered REPLICATION on genuinely new instances (investigation/confirm_new_instances.py)
New instances: Delhi Aug-12 noon (seed 31); Delhi Aug-19 Rakshabandhan festival traffic 9am (seed 23); six Solomon-style (seed_base 500).
Fresh algorithm seeds 30-35 (48 paired runs). Same 4 hypotheses, Holm over 4. Replication criterion fixed in advance: H1 and H3 significant, same sign.
- H1 QPSO+WB vs QPSO(frozen):         +5.25%, 46/48, Holm p < 0.0001   REPLICATED
- H2 PSO+WB  vs PSO(frozen):          +3.18%, 41/48, Holm p < 0.0001
- H3 QPSO+WB vs PSO+WB (same infra):  +2.28%, 35/48, Holm p = 0.0002   REPLICATED (same sign, significant)
- H4 QPSO+WB vs memetic GA:           +0.54%, 28/48, Holm p = 0.257    not significant (same pattern as confirmation 1)
- Descriptive: vs plain GA +12.2% (48/48), ACO +32.2% (46/48), TunedLS +19.8% (48/48).
- Mean gap to OR-Tools +2.7% (Delhi: +0.5% normal-day noon, +1.1% festival-day 9am; synthetic +1.8% to +4.7%).
- Pooled statistics and bootstrap CIs across both confirmations: pooled_confirmation_stats.log.
Remaining caveats: hyper-parameters not re-tuned after write-back (symmetric); H4 is a failure to reject, not equivalence;
the deck and results.json still carry the older frozen benchmark until the pipeline is regenerated with write-back as default.

---

# Round 6: OR-Tools hardening, anytime comparison, Route 1 (extended inter-route local search)
Scripts (run from project root): investigation/ortools_harden_and_anytime.py, route1_tuning.py, route1_confirm.py.
Logs: ortools_hardening_and_anytime.log, route1_tuning_screen.log, route1_confirmation.log, route1_confirmation_detail.log.
New options, all default OFF (default run re-verified bit-identical): QPSO/GA ls_extended (+ ls_ext_within=0.02, ls_ext_k=6);
solve_ortools_cvrptw(first_solution, metaheuristic, scale, precise).

## OR-Tools hardening (chosen on TUNING instances only)
15 configs (5 first-solution x 3 metaheuristics, exact x100 integer precision) vs the original. Chosen: LOCAL_CHEAPEST_INSERTION + GUIDED_LOCAL_SEARCH.
Original config was under-tuned (Delhi tuning instance 8904 vs 8579, 3.7% worse). The hardened config is NOT uniformly better on new instances
(e.g. Delhi 3pm: orig 9045 vs hard 9275) -> the fair reference is the better of the two.

## Route 2: anytime comparison at matched wall-clock (8 new instances, 3 seeds) - pre-registered criterion NOT MET (QPSO < OR-hard on 1/8)
mean gap to OR-hard: QPSO +8.0% (1.5s) / +5.3% (3.5s) / +2.4% (8s); memetic GA +5.3% / +4.1% / +3.9%. QPSO keeps improving with budget, the GA plateaus
(QPSO ahead of GA at ~8s on 6/8 instances; suggestive only, sign-test p~0.29). The single QPSO 'win' ties the original OR config.

## Route 1 tuning screen (4 instances x 6 seeds, Holm over 2)
QPSO+EXT vs QPSO +1.55% (17/24, Holm p=.003); GA+EXT vs GA +2.26% (17/24, Holm p=.001). Gap to hardened OR@5s: QPSO+EXT +0.5%, GA+EXT +0.1%.

## Route 1 PRE-REGISTERED confirmation (new instances: Delhi Aug-12 9am s111, Delhi Aug-19 festival noon s117, Solomon-style seed_base 1100; seeds 60-65; 48 pairs; Holm over 4)
H1 QPSO+EXT vs QPSO +1.88% (37/48) Holm<.0001 | H2 PSO+EXT vs PSO +3.26% (40/48) Holm<.0001 |
H3 QPSO+EXT vs PSO+EXT +0.52% (18 wins, rest mostly ties) Holm=.036 | H4 QPSO+EXT vs GA+EXT +0.21% (17/48) Holm=.71 (no difference).
vs hardened OR-Tools at matched ~3.7 s: QPSO+EXT mean gap +0.4% (Delhi-only +1.5%); strictly lower on 3/8 but two of those are within 0.03% (ties);
vs OR-hard@10s: +0.9%. Without EXT the same comparison is +2.4%.
CEILING EFFECT: on some instances every method (incl. OR-Tools) reaches the same cost, so differences concentrate in few instances.
Interpretation: EXT is shared infrastructure; it closes the gap to OR-Tools for the whole framework (GA +0.6%, PSO +1.0%, QPSO +0.4%) and
shrinks the quantum-vs-classical margin from ~+2.4% to ~+0.5%. It does not show the quantum update is responsible.

---

# Round 7: two pre-registered "conditions where QPSO might beat OR-Tools" - NEITHER CRITERION MET
Script: investigation/final_conditions.py (resumable). Log: final_conditions_summary.log.
All methods use the confirmed shared stack (write-back + extended inter-route local search). OR-Tools = better of hardened/original at the same
wall-clock. OR-Tools wrapper gained warm_start= so the disruption test re-optimises FROM the surviving plan (a cold-start comparison would have
manufactured a win for the archive). Note: C1 ran on 7 instances (an instance-count bug dropped one on the first pass, fixed on the resume).

C1 DISRUPTION (8x slowdown on the incumbent's busiest route; matched 0.05 s budget):
  archive beats WARM-STARTED OR-Tools on 2/7 (criterion >=5/6: NOT MET); mean gap archive +6.6%.
  BUT: QPSO warm-restart (8 iterations, same 0.05 s class) is -2.3% vs warm OR-Tools and wins 3/7, including two large wins on R-type
  instances (R1 73369 vs 97087, -24%). The archive itself recovers ~97% of the disruption loss instantly.
C2 BUDGET (~30 s matched): QPSO beats OR-Tools on 1/8 (criterion NOT MET); mean gap +0.65%, Delhi-only +1.16%; 4/8 exact ties.
  QPSO < memetic GA on 3/8. At 30 s the extra budget mostly produces ties, not wins.
CONCLUSION: no tested condition made QPSO reliably beat a tuned OR-Tools. The defensible claim is parity-on-average at n=24 (mean gap ~0.4-0.9%,
Delhi ~1.2-1.5%), a statistical tie with the memetic GA, and large significant wins over plain GA, ACO and tuned local search. The one genuinely
promising (but not criterion-meeting) result is fast disruption re-planning on R-type instances.

---

# Round 8: last screen of quantum-specific ideas on the corrected stack (investigation/final_quantum_ideas.py, analyze_ideas.py) - NO CANDIDATE
22 variants, TUNING set, 24 pairs each, Holm within family; decision rule fixed in advance (candidate = mean >= +1% AND raw p < 0.05).
F1 re-test of earlier rejections (scalar-u, elite-mBest, both, Borda x2, rank-jump): all within -0.2%..+0.3%, p >= 0.14 -> earlier rejections stand.
F2 beta sweep: all within noise; 1.0->0.5 is worse (-0.81%, raw p=0.025, Holm 0.15). Current 0.45->0.1 sits on a flat optimum.
F3 tunneling kernel (shared infra; applied to QPSO AND classical PSO): CTQW relocation vs swap +0.05% (Q) / +0.69% (P, raw p=.051, Holm .30);
   CTQW vs matched-entropy heat kernel -0.20% (Q) / +0.02% (P): no quantum attribution possible. Quantum vs classical update under each mode: +0.48% .. -0.17%, all n.s.
F4 boundary-aware jump scale: -0.19% / +0.08%, n.s.
Nothing was promoted to a confirmation run. Raw numbers: final_quantum_ideas_raw.json; analysis: final_quantum_ideas_analysis.log.
SUMMARY OF ALL ROUNDS: ~30 mechanisms tested; only write-back and the extended inter-route search survived pre-registered replication, both infrastructure
fixes that help classical methods as much as quantum ones.

---

# Round 9: pipeline regenerated with the confirmed stack (QPSO.STACK_DEFAULTS switch in run_everything.py; class defaults unchanged)
Held-out benchmark (same instances/seeds as before): QPSO vs plain GA +18.3%, ACO +43.8%, tuned LS +33.4% (p<0.0001, 100% wins); vs classical PSO +0.46% (p=0.20, n.s.);
vs memetic GA -0.2% (p=0.57, indistinguishable); mean gap to OR-Tools (best of two configs, 5 s) +0.7%.
Side effects found: hierarchical decomposition no longer beats flat QPSO on quality (n=72: flat 16003/12.1 s vs hier 17111/6.9 s); real-data impact figures shifted
(festival stale-plan penalty 7.2%/9.5%, re-routing regret 12.8%) -> quote as ranges. Recovery still no benefit (2/5 seeds). Pareto front 6/12.
