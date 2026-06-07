# CONTRIBUTING — Inversion Research Project

> Rules for where new files go, how they're named, and how they're reviewed. **Every new script, result directory, plot, or report created by the research loop (or the user) MUST follow this document.**

**Last updated:** 2026-04-16
**Companion docs:**
- `/DATA_INVARIANTS.md` — load-bearing pipeline parameters
- `notebooks/inversion/RENAMES.md` — old → new path mapping (cleanup commit)
- `notebooks/inversion/wiki/` — research wiki (Obsidian vault)

---

## 1. Naming conventions

### Scripts

```
notebooks/inversion/{NN_series}/m{NNN}_{semantic_name}.py
```

- `NN_series` — series directory (see § 2 below)
- `m{NNN}` — experiment number, zero-padded to 3 digits (e.g. `m070`, `m115`, `m127`)
- `{semantic_name}` — 2–5 lowercase words separated by underscores, describing *what the script does*. E.g. `flipped_omega_grid_search`, `warmstart_polish`, `surrogate_de_multistart`

**Examples:**
```
12_brightness_surface/m127_flipped_omega_grid_search.py
05_peak_graph_pipeline/m102_fullmse_selection.py
09_glint_analysis/m046_generate_trajectories.py
```

**What's banned:**
- `microNNN.py` (no semantic content)
- `microNNN_something.py` (inconsistent "micro" prefix)
- Unnumbered experiment scripts (`exp_bruteforce.py`, `demo_v3.py` — these are legacy; new scripts always get a number)

### Result directories

```
data/results/inversion_diagnostics/m{NNN}_{semantic_name}/
```

Matches the producing script's new name. Subdirectories inside follow a fixed layout:

```
m{NNN}_{semantic_name}/
├── batch_summary.json             # cross-seed rollup (if multi-seed)
├── seed_{NNN}/
│   ├── result.json                # per-seed headline: q0/w_dir/w_mag errors, classification
│   ├── run.log                    # stdout tee
│   ├── stage_a_{name}.npz         # checkpoint per stage (ALL candidates, not just winner)
│   ├── stage_b_{name}.npz
│   └── figures/                   # per-seed plots (PNG, HTML)
└── figures/                       # cross-seed plots
```

**Shared (non-experiment) data** lives one level up under a dedicated folder:

```
data/results/inversion_diagnostics/shared/            # trajectory datasets, brightness tables, etc.
data/results/inversion_diagnostics/analyses/          # inline one-off re-scorings, census outputs
```

### Wiki pages

```
notebooks/inversion/wiki/wiki/experiments/m{NNN}_{semantic_name}.md
```

Mirrors the script name 1:1. Branch and concept pages already use semantic names — keep them stable; only content gets updated.

### Reports

Lightweight writer/reviewer/analyst reports generated during the loop go in:

```
notebooks/inversion/{NN_series}/reports/m{NNN}_{writer|reviewer|analyst}.md
```

Major findings that outlive the loop get promoted to a wiki page.

## 2. Where things go — series directories

The 16 series directories group scripts by topic. Use the lowest-numbered series that fits; if none fits, create a new series with the next number and document it here + in the wiki index.

| Series dir | Topic |
|---|---|
| `00_pipeline_reference/` | Tutorial notebooks (not experiments — light-curve pipeline demos) |
| `01_global_search/` | Global DE / PSO / multi-start optimizer experiments |
| `02_isobrightness_filtering/` | Brightness-matching candidate filtering |
| `03_omega_bridging/` | ω-axis bridging between near-similar brightness candidates |
| `04_basin_characterization/` | Hessian, basin-width, landscape geometry |
| `05_peak_graph_pipeline/` | Peak-graph + full-MSE selection pipeline (long-running series) |
| `06_omega_bridging_benchmarks/` | Benchmark comparisons for ω-bridging methods |
| `07_L_conservation/` | Angular-momentum conservation checks |
| `07_multi_epoch_scoring/` | ⚠ collision with 07_L_conservation — migration pending |
| `07_winding_enumeration/` | ⚠ collision — migration pending |
| `08_integration/` | Integration / nudged / peak-shape filtering |
| `09_glint_analysis/` | Glint identification, trajectory generation (hosts m046/m048 trajectory generators) |
| `10_glint_inversion/` | Glint-based inversion experiments |
| `11_casadi_formulation/` | CasADi / symbolic-differentiation attempts (HISTORICAL) |
| `12_brightness_surface/` | Surrogate-model + attitude-isoshell + gradient-based inversion (current active branch) |

**Pending:** the three `07_*` collisions resolve in RENAMES.md. Until resolved, add new scripts to `12_brightness_surface/` (current active series) rather than 07.

**Library code** lives in `notebooks/inversion/lib/`. Functions used by ≥ 2 scripts MUST graduate there. One-off helpers stay in the script.

**Archive:** legacy scripts without experiment numbers (`exp_*.py`, `*_demo.py`, `*_v2.py`) live under `{NN_series}/archive/`. Not deleted, but not part of the live experiment tree.

## 3. Required boilerplate for new experiment scripts

Every new `m{NNN}_*.py` script MUST include:

```python
#!/usr/bin/env python3
"""
m{NNN}: {one-line title}

Hypothesis: {falsifiable claim}
Method: {short summary}
Expected outcome: {what would confirm/refute}

Writer: {agent name or "user"}
Reviewer: {agent name or "user"}
DATA_INVARIANTS: see /DATA_INVARIANTS.md — this script cites it.
"""

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

# ... standard imports ...

SEED = int(os.environ['M{NNN}_SEED'])                   # seed via env var
OUT_BASE = 'm{NNN}_{semantic_name}'                     # result dir name

# Tee stdout → run.log inside OUT_BASE/seed_{seed}/
# Time each stage, report wall time at the end
# Checkpoints FIRST: define np.savez(...) skeletons per stage BEFORE computation
# Save ALL candidates per stage (not just winner): q0, omega, all three errors, costs
# save_results() writes atomic JSON with: traj_seed, winner, all_candidates, timing, classification
```

Classification thresholds (per memory):
- **OK**: q0 < 5° AND w_dir < 5° AND |w_mag| < 5%
- **PARTIAL**: any metric in [5, 10]
- **FAIL**: any metric > 10

## 4. Reviewer checklist (augments DATA_INVARIANTS § 7)

Mechanical checks:
- [ ] Filename matches `m{NNN}_{semantic_name}.py`.
- [ ] Docstring cites DATA_INVARIANTS.md.
- [ ] Checkpoints `np.savez(...)` skeletons written BEFORE the computation code (not interleaved).
- [ ] All candidates saved per stage, not just the winner.
- [ ] All three errors (q0, w_dir, w_mag) saved per candidate.
- [ ] `Pool(N)` matches DATA_INVARIANTS § 6.
- [ ] `N_MAGS >= 20`.
- [ ] SEED from environment variable.
- [ ] Tee logging to `run.log`.
- [ ] Timing around every stage > 5 s.
- [ ] `save_results()` atomic JSON.
- [ ] OUT_BASE matches the filename's `m{NNN}_{semantic_name}`.
- [ ] No modifications to `src/` or `lib/` (unless explicitly in scope for this experiment).
- [ ] No git commits made by the writer.

Data-plumbing checks: **see DATA_INVARIANTS.md § 7.**

## 5. Commit discipline

Rename commits (this cleanup session) MUST:
- Use `git mv` for blame preservation (never delete + write).
- Be grouped by logical unit (one commit per series, or one per rename batch).
- Reference the RENAMES.md mapping in the commit body.

Ongoing experiment commits MUST:
- Include the script AND its wiki page AND (once generated) its result dir's `batch_summary.json`.
- Commit body names the hypothesis + the headline verdict (CONFIRMED / REFUTED / PARTIAL / INVALID).
- Never use `--no-verify`.
- No `Co-Authored-By: Claude` trailer (per user preference, memory `feedback_no_coauthor.md`).

## 6. Where new documentation goes

| Artifact | Location |
|---|---|
| Invariant change | `/DATA_INVARIANTS.md` + log entry |
| Naming / location rule change | this file (`CONTRIBUTING.md`) |
| New experiment finding | `wiki/wiki/experiments/m{NNN}_{semantic}.md` |
| Cross-cutting concept | `wiki/wiki/concepts/{name}.md` |
| Research branch decision | `wiki/wiki/branches/{name}.md` |
| Historical narrative | `wiki/wiki/journey.md` |
| Running experiment map | `notebooks/inversion/EXPERIMENTS.md` |
| Dead ends / what not to revisit | `notebooks/inversion/DEAD_ENDS.md` |
| Reviewer / analyst reports | `{NN_series}/reports/m{NNN}_{role}.md` |

## 7. When this file changes

Update this file FIRST, then update the scripts / dirs / wiki accordingly in the same commit. Do not let governance drift behind practice.
