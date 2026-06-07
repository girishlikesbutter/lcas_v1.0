---
title: "s108 — drop-C-pass + multistart-coarse-rank: the discriminator WORKS (blind Band A on 119, 0 phantoms) but the cross-cloud-multistart search is ~100x too slow for the 15-min budget"
type: experiment
sources:
  - experiments/s108a_stage0_oracle_probe.py
  - experiments/s108b_no_cpass_cross.py
  - experiments/s108c_multistart_coarse_rank.py
  - experiments/s108d_multistart_density_probe.py
  - results/s108/seed119/stage0_oracle.json
  - results/s108/seed119/no_cpass_cross_summary.json
  - results/s108/seed119/multistart_coarse_summary.json
  - results/s108/seed119/multistart_density.json
related:
  - experiments/s107_discrimination_test.md
  - experiments/s106_hybrid_loss_polish.md
  - experiments/s100_5step_coverage_proto.md
  - experiments/s098_local_densify_116.md
created: 2026-05-29
updated: 2026-05-29
confidence: high on the discriminator working (6 Band A, 0 phantoms, blind ranking+polish); high on the cost wall (both levers measured, deterministic); medium on the 15-min impossibility (N=1 seed, projections not a full run)
---

# TL;DR
The s107 handoff plan ("multistart inside the cross filter") is **dead** — Stage 0 showed the C-pass brightness gate rejects **0/23** multistart roots of the s106 oracle pair, because the truth-near root only fits brightness *after* the polish (raw 4.25°-off root misses C by 0.99 mag; `stage0_oracle.json`). The redesign — **drop C-pass, rank multistart roots by coarse-K full-LC RMSE** — **works as a discriminator**: on a 1675-pair subsample it lands **6 Band A, 0 phantoms**, the oracle reproduces s106 (0.0300, 0.57° off) via a fully blind ranking+polish path, and coarse-K cleanly separates truth-near (≤0.75) from junk (≥0.80; only the body-twin multi-sol intrudes) (`multistart_coarse_summary.json`). **But it cannot meet the 15-min budget:** dropping C-pass leaves **N=3,169,400** survivors (`no_cpass_cross_summary.json`), and both cost levers fail — cheaper multistart can't find the winding below n_dir=16 (full-run **71.5 hr**, `multistart_density.json`), and even a 10° coarse cross is 48k pairs = 65 min while drifting the nearest-truth rep to 4.4°. Cross-cloud-multistart is structurally ~100x over budget on aliased seed 119. **Bankable finding: the coarse-K-on-best-winding-root discriminator is clean. Redirect: the standing anchor-baseline-aliasing blocker, not a faster brute search.**

# What
s107 closed the "does the s106 polish phantom?" question (NO) but left the delivery question: how to feed a multistart-best seed (≲5° off truth-ω) to the polish at production scale. The handoff proposed "multistart inside the cross filter (replace single-shoot's one ω per pair with multistart's ~30, rank by coarse-K RMSE)." s108 tests that, in cheapest-first stages, and measures whether it can hit the 15-min/seed goal.

# How
Staged, each gating the next (all seed 119, Pool(24), v2 surrogate, [0.1,1.6] dps bracket, blind — truth only as post-hoc dir/qa/qb LABELS):

- **Stage 0** (`s108a`): oracle-pair (a=31,b=1471) unit test. multistart_shoot (17 dirs × 10 mags) → for each root: C-pass brightness check at C + coarse-K full-LC RMSE + s106 abc-window polish. Asks P1 (truth-near root exists?), P2 (does it pass C-pass?), P3 (does coarse-K rank it #1?), P4 (does it polish Band A?).
- **Step A** (`s108b_no_cpass`): re-cross all 4M repA×repB on connectability(geo<1e-3°)+|ω|-bracket ONLY (no C-pass). Measures the survivor count N and the multistart-on-all cost; confirms the oracle is admitted.
- **Step B** (`s108c`): on a stratified subsample of the no-C-pass survivors (+forced 75 near-orient pairs +oracle): multistart each → coarse-K per root → keep best-coarse per pair → rank → polish top-150. The redesigned-pipeline analogue of s107. Plus an offline intersection check: are the Band-A pairs in the old C-pass survivor set?
- **Probe 1** (`s108d`): sweep multistart (n_dir, n_mag) on the 75 near-orient + 500 junk pairs; measure truth-near recovery vs guesses and project the full-run cost.
- **Probe 2** (offline, inline): decimate repA/repB to coarser cells (deg 2→10), report rep counts, nearest-truth-rep distance, and projected cross wall at the n_dir=16 rate (12.3 pairs/s).

Past-error self-check: times[0]==0 gauge satisfied (abc window + coarse fwd/bwd legs start at 0); no truth injection in any residual; BLAS pinned before fork; the 65-hr projection was the *reason* the full run was NOT launched (kill-at-2× discipline applied pre-emptively).

# Result

## Stage 0 — oracle-pair unit test (`stage0_oracle.json`)
| check | outcome |
|---|---|
| single-shoot baseline | dir 104.07°, \|ω\| 0.2378 dps, C-pass FAIL (2.19 mag), coarse 3.71, polish → **Band D 1.8372** |
| multistart | 23 roots, best dir **4.25°**; **0/23 pass C-pass** |
| truth-near root (4.25°) | coarse-K **0.6262 (rank 1/23)**, C-pass err **0.9877 (FAIL)**, polish → **Band A 0.0300 (0.57°)** |
| verdict | P1 ✓ P2 ✗ P3 ✗ P4 ✗ → **GATE FAIL** (C-pass rejects every root; coarse-K ranks the truth-near root #1) |

The truth-near root only fits brightness post-polish (polish tightens 4.25°→0.57°). C-pass evaluates the raw root at epoch C (~9 turns out, dt_ac=2222s) where a 4.25° error amplifies to ~1 mag → structurally anti-truth.

## Step A — no-C-pass cross (`no_cpass_cross_summary.json`)
- **N = 3,169,400 survivors (79.23% of 4M)** in 438s — connectability+bracket barely cuts; C-pass had been a **113× filter** (3.17M→27,946).
- **Oracle admitted** (via single-shoot 0.2378 dps, in-bracket). Only **75** near-truth-orientation pairs (qa_off+qb_off<6°) exist.
- multistart-on-all projection: 3.17M × 1.76s / 24 = **232,423s ≈ 65 hr**.

## Step B — multistart + coarse-K rank, 1675-pair subsample (`multistart_coarse_summary.json`)
| metric | value |
|---|---|
| Band A (of polished top-150) | **6** (pol_dir 0.28–1.64°) |
| phantoms | **0** |
| oracle via blind path | coarse-rank **3/1675**, polish **0.02996 Band A, 0.57°** (= s106 bit-identically) |
| best-coarse: truth-near vs junk | near-orient 0.43–0.75 (low) ; junk min 0.5162 (=body-twin), median 1.93 |
| junk pairs below oracle's 0.6262 | **1** (the qa/qb≈180° body-twin — a real s043 multi-sol, not a false positive) |
| ALL 6 Band-A pairs in old C-pass survivor set? | **0/6** (verified vs s107/cross.npz) — C-pass kills every winner |

## Probe 1 — multistart density (`multistart_density.json`)
truth-near recovery (best-coarse root with dir≤10° AND coarse≤0.80) per (n_dir,n_mag):
| config | shoots | near_recov /75 | oracle (coarse/dir) | full-run proj |
|---|---|---|---|---|
| (2,6) | 18 | 0 | 1.303 / 59.26 | 7.3 hr |
| (4,10) | 50 | 1 | 1.303 / 59.26 | 19.6 hr |
| (8,10) | 90 | 1 | 1.303 / 59.26 | 36.3 hr |
| (16,10) | 170 | **9** | **0.626 / 4.25** | **71.5 hr** |

The oracle's truth root is found **only at n_dir=16**; fewer dirs are faster but land on the wrong winding. Cheaper multistart is dead.

## Probe 2 — coarse cross resolution (offline)
| deg | repsA×repsB | pairs(×0.79) | nearest-truth A/B | proj cross wall |
|---|---|---|---|---|
| 2 | 2000×2000 | 3.16M | 1.02 / 0.68 | 4282 min |
| 6 | 405×742 | 237k | 1.76 / 1.73 | 322 min |
| 8 | 274×461 | 100k | 2.27 / 3.37 | 135 min |
| 10 | 197×310 | 48k | 4.43 / 4.49 | **65 min** |

No resolution is both fast enough (≤15 min) and fine enough (nearest-truth rep inside the ~5° polish basin). The isophote clouds are intrinsically large.

# Why this matters
- **The discriminator is proven and clean** — coarse-K full-LC RMSE on the best multistart winding root is a blind, phantom-free way to surface truth-near pairs (6 Band A, 0 phantoms, body-twin is the only "intruder" and it is a legitimate solution). This is the bankable, reusable result of the whole s100→s108 cross-cloud line.
- **C-pass is structurally anti-truth here and must not gate** — it rejects the brightness at a far epoch on the un-polished root; every Band-A winner is dropped by it.
- **Cross-cloud-multistart cannot meet the 15-min budget on aliased seed 119.** Per-pair cost is irreducible (n_dir<16 misses the winding) and the cross is intrinsically quadratic-large (isophote clouds don't shrink enough under coarsening). ~100× over budget; no incremental knob (body-twin halving, tighter tolerance) closes a ~280× gap.
- **Redirect:** the reason 119 needed cross-cloud at all is anchor-baseline aliasing (PROGRESS standing #1) — the cheap single-anchor densify that inverted 116 in 11.7 min (s099) lacked a truth-near-ω candidate for 119. Fix the input (anchor selection), don't brute-force a bad input.

# Numbers
- Stage 0: oracle single-shoot dir 104.07°, |ω| 0.2378 dps, C-pass err 2.19 mag, coarse 3.71, polish Band D 1.8372; multistart 23 roots, truth-near root coarse 0.6262 (rank 1/23), C-pass err 0.9877 (FAIL), polish Band A 0.0300 dir 0.57° (source: `results/s108/seed119/stage0_oracle.json`)
- Step A: N=3,169,400 (79.23%), oracle admitted, 75 near-orient pairs, wall 438s, proj 65 hr (source: `results/s108/seed119/no_cpass_cross_summary.json`)
- Step B: 1675 subsample, 6 Band A, 0 phantoms, oracle rank 3 polish 0.02996; near-orient best-coarse 0.43–0.75, junk min 0.5162 med 1.93, 1 junk<0.6262 (body-twin); all 6 Band A NOT in old C-pass set; wall B1 249s B2 80s (source: `results/s108/seed119/multistart_coarse_summary.json`, `.npz`)
- Probe 1: recovery {(2,6):0,(2,10):0,(4,6):0,(4,10):1,(8,6):0,(8,10):1,(16,10):9}/75; oracle found only at n_dir=16; full-run 7.3→71.5 hr (source: `results/s108/seed119/multistart_density.json`)
- Probe 2: deg=10 → 48k pairs, nearest-truth 4.4°, 65 min cross (source: inline computation, s108d rate 12.3 pairs/s)

# Artefacts
- `results/s108/seed119/stage0_oracle.json` — Stage 0 per-root C-pass/coarse/polish + verdict
- `results/s108/seed119/no_cpass_cross.npz` (218MB, NOT committed) + `no_cpass_cross_summary.json` — the 3.17M survivor set + funnel stats
- `results/s108/seed119/multistart_coarse.npz` + `multistart_coarse_summary.json` — Step B per-pair best-coarse/band/dir + the 6 Band A
- `results/s108/seed119/multistart_density.json` — Probe 1 sweep + cost projections
- logs: `no_cpass_cross.log`, `multistart_coarse.log`, `multistart_density.log`

# Out of scope
- **Short-baseline cross** (re-pick A,B <1 rotation apart so single-shoot finds the unique winding → no multistart → cheap cross + this proven discriminator). Untested whether a short baseline still discriminates. The most direct salvage of the cross-cloud idea.
- **Anchor-baseline aliasing fix** (the recommended redirect) — make the cheap single-anchor densify (s098/s099) work on 119 via better anchor selection. Not attempted here.
- The full blind 65-hr run was NOT executed (deliberately, per kill-at-2×).
- N=1 seed (119). 116 and other seeds untested under this redesign.

# Cross-references
- [[s107_discrimination_test]] — closed the phantom question; s108 tests its proposed delivery fix and finds it too slow.
- [[s106_hybrid_loss_polish]] — the polish under test; s108 reproduces its 0.0300 via a blind path.
- [[s100_5step_coverage_proto]] — the cross filter whose C-pass s108 removes.
- [[s098_local_densify_116]] / s099 — the cheap single-anchor 11.7-min pipeline the redirect would adapt to 119.
- [[feedback_omega_prior_physical_bracket]] — [0.1,1.6] dps bracket.
- NOTE: a CONCURRENT session created `experiments/s108{b_cost_discrim_probe,c_polish_tolerance,d_blind_invert,e_joint_polish_tolerance,f_multistart_budget}.py` + `results/s108{b,c,e,f}/` during this session (timestamps 07:07–07:25). Those are NOT part of this writeup and were left untracked.
