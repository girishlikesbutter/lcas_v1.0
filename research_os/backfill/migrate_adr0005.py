#!/usr/bin/env python3
"""ADR-0005 migration: promote the four method-shaped goal_nodes to Pipelines.

One-shot, idempotent. Realises the decided model (see decisions/0005):
  - DELETE the two pure-method CLOSED branches (densify, joint-grid-pivot); repoint
    their runs' goal_node UP to the chapter the pipeline now serves, + add tests[].
  - SPLIT the two LIVE ones: keep goal_cross-cloud-bridging (chapter) and
    goal_windowed-photometry-polish (retitled to the pure question); the new pipelines
    serve them; their runs gain tests[] only (goal_node unchanged).
  - Demonstrate the blocked-by payoff edge on two grounded, cross-pipeline runs.

Run AFTER the 4 pipeline_*.json + hard-shoot-trap.json exist. Then run loop/refresh.py
(recomputes child_runs/spent from goal_node) and loop/validate.py (must stay closed).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_os/
REC = os.path.join(ROOT, "records")
GOALS = os.path.join(ROOT, "goals")


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _save(path, obj):
    with open(path, "w") as fh:
        fh.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


# run_id -> (new_goal_node_or_None, pipeline_to_test)
REPOINT = {
    "s059k_densify_ndirs": ("goal_cloud-architectures", "pipeline_densify"),
    "s069_replicate_s059k": ("goal_cloud-architectures", "pipeline_densify"),
    "s082_joint_grid_pivot": ("goal_blind-pipeline", "pipeline_joint-grid-pivot"),
}
# tests-only (goal_node stays): run_id -> pipeline
TESTS_ONLY = {
    **{r: "pipeline_cross-cloud" for r in [
        "s076_horizon_density", "s078_nll_residual_ab", "s091_per_pair_timing",
        "s092_cross_cloud_116", "s093_fullc_score_survivors", "s094_window_score",
        "s095_calibration_threshold_transfer", "s096_density_usefulness_116",
        "s108_cross_cloud_multistart_cost_wall",
    ]},
    **{r: "pipeline_single-wind-lc-window-polish" for r in [
        "s105_pairs_to_omega_decomposition", "s106_hybrid_loss_polish",
        "s107_discrimination_test",
    ]},
}
# blocked-by demonstration (grounded): run_id -> glossary_term id
BLOCKED_BY = {
    "s105_pairs_to_omega_decomposition": "hard-shoot-trap",
    "s108_cross_cloud_multistart_cost_wall": "finite-diff-omega-aliasing",
}
DELETE_GOALS = ["goal_densify-ndirs", "goal_joint-grid-pivot"]
RETITLE = {
    "goal_windowed-photometry-polish": "Break the hard-shoot trap (+ discriminate far pairs)",
}


def _add(rec, field, val):
    cur = rec.get(field) or []
    if val not in cur:
        cur.append(val)
    rec[field] = cur


def main():
    log = []
    for rid, (new_goal, pipe) in REPOINT.items():
        p = os.path.join(REC, f"{rid}.json")
        d = _load(p)
        if new_goal and d.get("goal_node") != new_goal:
            log.append(f"  {rid}: goal_node {d['goal_node']} -> {new_goal}")
            d["goal_node"] = new_goal
        _add(d, "tests", pipe)
        _save(p, d)
    for rid, pipe in TESTS_ONLY.items():
        p = os.path.join(REC, f"{rid}.json")
        d = _load(p)
        _add(d, "tests", pipe)
        _save(p, d)
        log.append(f"  {rid}: tests += {pipe}")
    for rid, term in BLOCKED_BY.items():
        p = os.path.join(REC, f"{rid}.json")
        d = _load(p)
        _add(d, "blocked_by", term)
        _save(p, d)
        log.append(f"  {rid}: blocked_by += {term}")
    for rt, title in RETITLE.items():
        p = os.path.join(GOALS, f"{rt}.json")
        d = _load(p)
        log.append(f"  {rt}: title -> {title!r}")
        d["title"] = title
        _save(p, d)
    for gid in DELETE_GOALS:
        p = os.path.join(GOALS, f"{gid}.json")
        if os.path.exists(p):
            os.remove(p)
            log.append(f"  DELETED goal {gid}")
    # prune deleted goals from _index, fix retitled entries
    idx_path = os.path.join(GOALS, "_index.json")
    idx = _load(idx_path)
    idx = [e for e in idx if e["id"] not in DELETE_GOALS]
    for e in idx:
        if e["id"] in RETITLE:
            e["title"] = RETITLE[e["id"]]
    _save(idx_path, idx)
    log.append(f"  _index: pruned {DELETE_GOALS}, retitled {list(RETITLE)}")

    print("ADR-0005 migration applied:")
    print("\n".join(log))


if __name__ == "__main__":
    main()
