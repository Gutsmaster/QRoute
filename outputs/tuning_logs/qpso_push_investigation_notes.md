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
