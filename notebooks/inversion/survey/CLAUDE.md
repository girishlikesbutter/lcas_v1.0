# Agent instructions — forward-survey workspace

You are working in `notebooks/inversion/survey/`, a clean-slate workspace that surveys the post-bug-fix LCAS forward model. **Read `README.md` before doing anything.**

## Workspace contract (binding)

1. **Forward-model substrate is sound.** Propagator, renderer, BRDF, surrogate, SPICE, satellite STL, articulation — all validated under correct truth. Import freely from `src.*` or use `lib/` wrappers.

2. **Inversion-side substrate is contraband.** Do NOT import from any `notebooks/inversion/m*_*.py`, `m103_hybrid.py`, `m115_surrogate_pipeline.py`, `m126_wrapped_pipeline.py`, or anything in `11_casadi_formulation/` or `12_brightness_surface/`. If you need a subroutine, copy-adapt-retest in `lib/` or in the relevant experiment script.

3. **Buggy-era findings do NOT carry over as fact.** They carry over as questions. If you find yourself wanting to cite "alignment cost is anti-truth" or "m115 bridging radius is 5°" or "seed 91 is solved," stop — those claims are listed in `concepts/known_pathologies_to_revalidate.md` and need to be re-measured under correct truth before being treated as load-bearing.

4. **Numbering is fresh.** First experiment is `s001_*`, not `m147_*`. The old `mXXX` series is frozen at `notebooks/inversion/wiki/`.

5. **Wind-down discipline (every session, not optional):**
   - Write `experiments/sXXX_*.md` with frontmatter (`title / type / sources / related / created / updated / confidence`) and the same body sections the old wiki used (TL;DR, What, How, Result, Why this matters, Numbers, Artefacts, Out of scope, Cross-references). One page per experiment.
   - Append `## [YYYY-MM-DD] ingest | sXXX | <one-line summary>` to `log.md`.
   - Update `PROGRESS.md`'s "current question" / "last measured" / "next" sections.
   - Commit on the `forward_survey` branch.

6. **Trust the lib/, but verify quaternion conventions whenever you write any new q→matrix code.** See `concepts/quaternion_convention.md` for the standard finite-difference smoke test.

## What auto-memory will do

The user's `~/.claude/projects/-home-girish-projects-lcas-v1-0/memory/MEMORY.md` auto-loads regardless of working directory. It contains methodology / how-to-work rules (BLAS threading, save-intermediate-results, ρ-band reporting, no-coauthor) plus the convention-bug entries — all of which apply here. It also contains a fair number of buggy-era project memories (`project_micro*`, `project_phase_B_*`, etc.) that will visibly contradict this workspace's clean-slate framing. **Treat any project-level memory written before 2026-04-30 with skepticism**; the buggy-era findings inside them are exactly what the survey exists to re-derive. Methodology and surrogate / bug-fix entries from any date remain authoritative.

## Default mode: research-loop discipline

Use the existing research-loop framing: design-time gate, cost-benefit gate, end-of-round context check, no inline `bash -c` Python, no parallel CPU-heavy jobs, never sleep-wait. Save intermediate results in NPZ checkpoints. Always report all three errors (q0, ω_dir, ω_mag) and the ρ-band classification.

## Where to find things

- **Forward-model imports:** `src/dynamics/attitude_propagator.py` (post-fix), `src/computation/observation_geometry.py`, `src/io/stl_loader.py`, `src/computation/shadow_engine.py`, `src/computation/lightcurve_generator.py`, `src/articulation/`.
- **Surrogate:** `~/surrogate_model/` — `from surrogate_model import SurrogateModel; m = SurrogateModel.load_default()`.
- **Trajectories:** `data/trajectories/traj_seedXXX.npz` (symlinks to canonical post-fix NPZs in the parent project).
- **Helpers:** `lib/traj_load.py`, `lib/surrogate_eval.py`, `lib/hifi_render.py`, `lib/cost_surfaces.py`. Thin by design.
- **Frozen reference (don't auto-load):** `notebooks/inversion/wiki/` and `notebooks/inversion/CURRENT_STATE.md`.
