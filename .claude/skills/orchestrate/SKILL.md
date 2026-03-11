---
name: orchestrate
description: "Run parallel inversion experiments with multiple Claude Code workers. Use when planning, dispatching, or reviewing a round of parallel micro-experiments. Triggers on: orchestrate, parallel experiments, run workers, dispatch experiments, experiment round."
---

# Parallel Experiment Orchestrator

Coordinates parallel inversion micro-experiments across multiple Claude Code worker terminals. You are the orchestrator — you plan, generate prompts, and review results. The user relays prompts to workers and reports back.

---

## The Flow

There are 4 phases with checkpoints between them. At each checkpoint you produce a **resumption prompt** the user can copy, rewind to the checkpoint, and paste back to you — recovering all context spent on discussion.

---

## Phase 1: Orient

**Your job:** Get oriented and brief the user.

1. Read `notebooks/inversion/EXPERIMENTS.md` (the single source of truth)
2. Read recent git log (`git log --oneline -20`)
3. Check for any in-progress results (`ls data/results/inversion_diagnostics/micro*.json | tail -10`)
4. Read the latest findings report if one exists (`docs/reports/`)

**Output:** A concise briefing:
- Where we are (active thread, last completed)
- What's blocking / what's next
- Your proposed experiments (typically 3 parallel workers)

**Then discuss** with the user until you agree on the experiments. This may take several turns.

**CHECKPOINT: Orient complete**

After agreement, output:

```
--- CHECKPOINT: Orient complete ---
Agreed experiments:
1. [micro_id]: [one-liner]
2. [micro_id]: [one-liner]
3. [micro_id]: [one-liner]

Resumption prompt (copy this, rewind to checkpoint, paste back):
────────────────────────────────────────
We've reviewed EXPERIMENTS.md and agreed on 3 parallel experiments:
1. [micro_id]: [full description with method, inputs, outputs]
2. [micro_id]: [full description with method, inputs, outputs]
3. [micro_id]: [full description with method, inputs, outputs]
Budget: [X] min per worker. Generate worker prompts.
────────────────────────────────────────
```

---

## Phase 2: Generate Prompts

**Your job:** Generate one self-contained prompt per worker.

Before generating prompts, read the existing micro-experiment scripts to understand code patterns:
- Recent scripts in `notebooks/inversion/07_*/` or `08_*/`
- `notebooks/inversion/lib/experiment_setup.py` for setup boilerplate
- Any data files the worker will need (inspect with python one-liners)

### Approval gate

After presenting all prompts, **wait for user approval** before creating worktrees. The user may want to discuss or adjust prompts first.

### Worktree creation

Once the user approves (says "go", "approved", "lgtm", etc.), **create the worktrees yourself** using the Bash tool. Use `~/projects/lcas-workers/` (NOT `/tmp` — the repo is ~5GB per worktree and tmpfs is too small).

First, check for stale state from a previous round:
```bash
git worktree list                    # any leftover worktrees?
git branch --list 'exp/*'            # any leftover exp/ branches?
# Clean up if needed:
# git worktree remove ~/projects/lcas-workers/worker1
# git branch -D exp/stale-branch
```

Then create fresh worktrees (`-b` creates the branch implicitly — no separate `git branch` needed):
```bash
mkdir -p ~/projects/lcas-workers
git worktree add ~/projects/lcas-workers/worker1 -b exp/[branch-name-1] inversion_q_w
git worktree add ~/projects/lcas-workers/worker2 -b exp/[branch-name-2] inversion_q_w
git worktree add ~/projects/lcas-workers/worker3 -b exp/[branch-name-3] inversion_q_w
```

Then tell the user the directories are ready. The user will tmux into each and run `claude` themselves.

### Worker prompt template

Each worker prompt MUST follow this structure:

```
You are running a focused inversion experiment on the LCAS project.

## Task: [micro_id] — [title]

[Clear description of the question being asked]

## Method

[Step-by-step experimental procedure, numbered]

## Code patterns

[Exact imports and function signatures to use — paste real code, not descriptions.
Include setup_experiment() call with exact parameters.
Include any helper functions the worker needs, copied from existing experiments.]

## Files to create

- Script: `notebooks/inversion/[series_dir]/[micro_id].py`
- Results JSON: `data/results/inversion_diagnostics/[micro_id].json`
- Plot PNG: `data/results/inversion_diagnostics/[micro_id].png`

## Constraints

- Use multiprocessing.Pool(8) for parallel work
- Readable code with clear variable names — prioritise clarity over brevity
- Do NOT compress code to save lines — readability matters more than length

## Git

Commit the script once it runs successfully:
  `feat: [micro_id] — [description]`
Commit results separately:
  `results: [micro_id] [description]`
Do NOT amend commits. Always make new ones.

## IMPORTANT: Scope of work

Your job is to write the script, run it, and save results (JSON + PNG).
If the script crashes, fix it and re-run.
Do NOT redesign the approach or optimise performance.
Do NOT modify existing library code (src/, lib/).
If results are unexpected or runtime is long, just report what happened and stop.
```

### Prompt quality checklist

Before outputting each prompt, verify:
- [ ] Exact imports and function calls are included (not just descriptions)
- [ ] Setup parameters match the standard test case
- [ ] Data file paths are correct and keys are specified
- [ ] The "do not redesign" guardrail is present
- [ ] Git commit messages are specified
- [ ] Output file paths are specified

**CHECKPOINT: Prompts generated**

```
--- CHECKPOINT: Prompts generated ---
3 worker prompts ready. Worktree commands provided.

Resumption prompt (copy after workers complete, rewind to here, paste back):
────────────────────────────────────────
Workers are done. Review results on these branches:
- exp/[branch-1]: [micro_id] — [one-liner]
- exp/[branch-2]: [micro_id] — [one-liner]
- exp/[branch-3]: [micro_id] — [one-liner]
Worktrees at ~/projects/lcas-workers/worker{1,2,3}. Check results and report.
────────────────────────────────────────
```

---

## Phase 3: Review

**Triggered by:** User reports workers are done (or reports issues mid-flight).

**Your job:** Examine each branch's results.

For each worker, you can read files directly from the worktree paths (useful for mid-flight checks or if commits haven't landed yet):
- `~/projects/lcas-workers/worker1/notebooks/inversion/08_integration/...`
- `~/projects/lcas-workers/worker1/data/results/inversion_diagnostics/...`

For each worker branch:
1. `git log --oneline exp/[branch] -10` — check commits landed
2. Read the results JSON (from worktree path or via `git show exp/[branch]:path`)
3. Read the script (verify it's reasonable)
4. Look at findings if any FINDINGS.md was created

**Output:** A synthesis across all workers:
- What worked, what failed
- Root causes of any failures
- Whether the results change the pipeline design
- What the next round of experiments should be (if any)

Discuss with user until you agree on interpretation and next steps.

**CHECKPOINT: Review complete**

```
--- CHECKPOINT: Review complete ---
Key findings:
- [micro_id]: [result summary]
- [micro_id]: [result summary]
- [micro_id]: [result summary]
Decision: [what we agreed to do next]

Resumption prompt (copy, rewind, paste):
────────────────────────────────────────
Review of 3 parallel experiments complete. Findings:
1. [micro_id]: [one-liner result]
2. [micro_id]: [one-liner result]
3. [micro_id]: [one-liner result]
Decision: [next action]
Update EXPERIMENTS.md with these findings, then [propose next round / stop].
────────────────────────────────────────
```

---

## Phase 4: Update

**Your job:** Update the experiment record and clean up git state.

1. Update `notebooks/inversion/EXPERIMENTS.md`:
   - Add new series/entries to Section 2 (The Map)
   - Update Section 1 (Resume Point) with new status
   - Add to Section 3 (Superseded Work) if anything is now dead
   - Add to Section 4 (Decision Log) with date and findings
2. If results warrant it, write/update a findings report in `docs/reports/`
3. Cherry-pick or merge useful commits from worker branches onto the main working branch
4. Clean up worktrees and branches (this order matters):
   ```bash
   git worktree remove ~/projects/lcas-workers/worker1
   git worktree remove ~/projects/lcas-workers/worker2
   git worktree remove ~/projects/lcas-workers/worker3
   git branch -D exp/[branch-1] exp/[branch-2] exp/[branch-3]
   ```
   `git worktree add -b` implicitly creates branches. If you don't delete them after merging, they accumulate as stale refs.

---

## Mid-flight interventions

If the user reports a worker is stuck or producing bad results mid-flight:

1. Check what's happening: read the worker's script, check for running processes
2. Diagnose the issue
3. Generate a **follow-up prompt** for the user to paste to the worker
4. The follow-up prompt MUST include: "Do NOT commit anything. Just write the files and run. I will handle git at the end."

---

## Rules

- **Workers do not iterate.** They write, run, save, and stop. Diagnosis and redesign is YOUR job as orchestrator.
- **Workers do not modify library code.** They only create scripts in `notebooks/inversion/` and save results to `data/results/`.
- **Git worktrees are mandatory.** Each worker gets an isolated copy. No cross-contamination.
- **Resumption prompts at every checkpoint.** The user may rewind to recover context.
- **Standard test case** (unless explicitly changed): Intelsat 901, true_omega_deg=(0.5, -0.3, 2.0), n_obs=500, noise_sigma=0.05, end_time='2020-02-05T11:00:00', peaks at epochs [183, 260, 360].
