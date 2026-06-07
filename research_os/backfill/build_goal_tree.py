#!/usr/bin/env python3
"""W-B backfill, stage 1: emit the goal-tree skeleton + experiment->node map + run manifest.

This script owns the SYNTHESIS layer of the backfill:
  - the goal tree (thesis + chapters + branch/question nodes, incl. dead/revivable),
  - the deterministic experiment -> goal_node classification,
  - a frontmatter-derived run manifest the extraction subagents consume.

It does NOT read experiment bodies (status/metrics) — that is the subagents' job.
Outputs (canonical, git-tracked):
  research_os/goals/<id>.json         one file per node
  research_os/goals/_index.json       derived index (regenerable convenience)
  research_os/backfill/run_manifest.json   subagent input (frontmatter + node)

Run:  python research_os/backfill/build_goal_tree.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repo root
RO = ROOT / "research_os"
GOALS = RO / "goals"
SCHEMAS = RO / "schemas"
EXP = ROOT / "notebooks/inversion/survey/experiments"
WRITEUP_PREFIX = "notebooks/inversion/survey/experiments"
BACKFILL_DATE = "2026-06-01T00:00:00Z"   # when these node OBJECTS were authored (backfill)

try:
    from jsonschema import Draft202012Validator
except ImportError:
    sys.exit("pip install jsonschema")

# ---------------------------------------------------------------- goal tree ---
# (id, node_kind, parent, title, state, trunk?, last_measured-or-None)
# last_measured = (run_id, summary, iso_date)
TH = "goal_attitude-inversion"
NODES = [
    (TH, "thesis", None,
     "Recover satellite attitude state (q0, ω) from a single ground-based light curve",
     "open", True,
     ("s112_slow_tumbler_pipeline_report",
      "blind multi-sol inversion lands on slow tumbler 116 (q0 0.19°, ω-dir 0.03°); fast aliased 119 needs the cross-cloud route; slow-sweep 10/42/31 broken by a decimation bug",
      "2026-05-29")),

    # ---- chapters ----
    ("goal_forward-substrate", "chapter", TH,
     "Forward-model substrate: correct, conserved, versioned, fast", "open", False, None),
    ("goal_cost-landscape", "chapter", TH,
     "Surrogate cost surface vs (q0, ω): shape, ρ-bands, joint polish", "open", False, None),
    ("goal_omega-from-lc", "chapter", TH,
     "Constrain ω from the light curve alone (priors & brackets)", "open", False, None),
    ("goal_basin-polhode-twin", "chapter", TH,
     "Solution-space structure: basin width, polhode geometry, body-twin", "open", False, None),
    ("goal_cloud-architectures", "chapter", TH,
     "Anchor / cloud candidate generation (cascade, threading, anchor-selection)", "revivable", False, None),
    ("goal_cross-cloud-bridging", "chapter", TH,
     "Two-point BVP ω-solve across anchor clouds + the discriminator", "open", False, None),
    ("goal_blind-pipeline", "chapter", TH,
     "End-to-end ≤15-min blind Band-A inversion on stratified seeds", "open", False, None),
    ("goal_literature", "chapter", TH,
     "Novelty & positioning vs Robinson–Frueh", "open", False, None),

    # ---- CH1 forward-substrate ----
    ("goal_convention-bug", "branch", "goal_forward-substrate",
     "ω-sign propagator convention bug: detect, fix, validate, replicate", "closed", False,
     ("s067_postfix_propagator_validation",
      "4/4 post-fix gates PASS; cohort drift ~0; fix commit d5705ff invalidated s001-s066", "2026-05-12")),
    ("goal_jacobi-propagator", "branch", "goal_forward-substrate",
     "Closed-form Jacobi-elliptic q(t) torque-free propagation", "closed", False,
     ("s074_elliprj_path2", "elliprj Path-2 default, 4.76 ms/cand Pool(24), all gates PASS", "2026-05-20")),
    ("goal_gauge-times0", "branch", "goal_forward-substrate",
     "times[0]==0 propagator-gauge bug audit", "closed", False,
     ("s097_times0_gauge_bug_audit", "bug found+fixed; reversed s093/s094/s095 conclusions", "2026-05-22")),

    # ---- CH2 cost-landscape ----
    ("goal_cost-at-truth", "branch", "goal_cost-landscape",
     "Cost-at-truth & surrogate landscape characterization", "closed", False,
     ("s081_hifi_rho_bands_twin", "surrogate↔hi-fi agreement holds; surrogate-MSE the universal cost", "2026-05-20")),
    ("goal_joint-polish", "branch", "goal_cost-landscape",
     "Joint (q0, ω) LM polish + ρ-band classification", "closed", False,
     ("s064_jacobi_polish", "polhode-basis LM polish 21-27× in-basin speedup", "2026-05-13")),

    # ---- CH3 omega-from-lc ----
    ("goal_lc-omega-priors", "branch", "goal_omega-from-lc",
     "ω from LC features (peak-spacing, regression) — DEAD class", "closed", False,
     ("s055c_aux_priors", "LC-only ω-direction priors structurally dead (28-feat 16.4% LOO MAPE)", "2026-05-08")),
    ("goal_ls-bracket", "branch", "goal_omega-from-lc",
     "Lomb–Scargle blind |ω| bracket (live tool)", "closed", False,
     ("s099_blind_fast_invert_116", "s019 bracket held-out 20/20 on seeds 100-119", "2026-05-26")),
    ("goal_omega-mag-basin", "branch", "goal_omega-from-lc",
     "|ω|-magnitude good-fit basin sharpness; 1-D refine", "closed", False,
     ("s084_omega_refine_viability", "1-D |ω|-refine viable (0 aliases); binding constraint = anchor q-accuracy", "2026-05-21")),

    # ---- CH4 basin-polhode-twin ----
    ("goal_basin-width", "branch", "goal_basin-polhode-twin",
     "ω-mag basin width vs |ω| / pol_diam (cohort)", "closed", False,
     ("s053_cohort_polhode_survey", "pol_diam Spearman -0.94 vs basin width (n=100)", "2026-05-07")),
    ("goal_polhode-prior", "branch", "goal_basin-polhode-twin",
     "Polhode-conditioned ω prior (|L|, label, phase)", "revivable", False,
     ("s063d_polhode_identity_filter", "polhode label not preserved through cascade noise; survives for upstream sampling", "2026-05-12")),
    ("goal_body-twin", "branch", "goal_basin-polhode-twin",
     "Body-twin: canonical-hemisphere search-space halving", "closed", False,
     ("s081_hifi_rho_bands_twin", "twin LC equivalence holds at hi-fi; ~2× dedup", "2026-05-20")),

    # ---- CH5 cloud-architectures ----
    ("goal_cloud-data-seedgen", "branch", "goal_cloud-architectures",
     "Cloud-data architecture as a seed generator", "closed", False,
     ("s059j_cloud_data_omega_grid", "ω-grid architecture validated end-to-end; headline yield was ORACLE-TAINTED (retracted)", "2026-05-08")),
    ("goal_anchor-selection", "branch", "goal_cloud-architectures",
     "Anchor selection: coarse-find then dense-resample", "closed", False,
     ("s085_anchor_accuracy_gate", "sharp-|C_t| anchor delivers fast-seed q 1.65-3.30° @400k", "2026-05-21")),
    ("goal_threading", "branch", "goal_cloud-architectures",
     "Thread (q, ω) pairs from anchor under the connectability tube", "closed", False,
     ("s061c_seed28_thread", "cluster-disappearance works; lineage tracking needs constant-ω pairs", "2026-05-11")),
    ("goal_densify-ndirs", "branch", "goal_cloud-architectures",
     "Densified ω-grid + full-LC polish → Band A on slow tumblers", "closed", False,
     ("s069_replicate_s059k", "Band A on cohort slow seeds (89, 10) replicated post-fix", "2026-05-12")),

    # ---- CH6 cross-cloud-bridging ----
    ("goal_bvp-shoot", "branch", "goal_cross-cloud-bridging",
     "Two-point BVP ω-solve (shoot) conditioning", "closed", False,
     ("s088_bvp_shoot_conditioning", "BVP beats finite-diff 1-2 OOM; clears 1.25° gate at σ_q≲2.0°", "2026-05-22")),
    ("goal_third-anchor", "branch", "goal_cross-cloud-bridging",
     "Third-anchor over-determination (disambiguator)", "closed", False,
     ("s089_third_anchor_overdetermination", "joint collapses winding ladder to unique truth; no fast-seed noise-floor gain", "2026-05-22")),
    ("goal_cross-cloud-discriminator", "branch", "goal_cross-cloud-bridging",
     "Discriminator among cross-cloud survivors (full-LC RMSE / windowed)", "open", False,
     ("s108_cross_cloud_multistart_cost_wall", "coarse-K discriminator clean (6 Band A, 0 phantoms) but search ~100× over 15-min budget on 119", "2026-05-29")),
    ("goal_l-vector-discriminator", "branch", "goal_cross-cloud-bridging",
     "L_J2000 cross-anchor matching as a discriminator", "closed", False,
     ("s079_regime_stratified_l_basins", "L-vector basins regime-stratified; N>1 cat-4 framing still open", "2026-05-14")),

    # ---- CH7 blind-pipeline ----
    ("goal_joint-grid-pivot", "branch", "goal_blind-pipeline",
     "Branch v2: dense joint (q0, ω) grid → top-K polish", "closed", False,
     ("s082_joint_grid_pivot", "median truth-nearest rank 70823/320000 → Branch v2 FALSIFIED", "2026-05-20")),
    ("goal_anchor-accuracy-gate", "branch", "goal_blind-pipeline",
     "Anchor q-accuracy + ω-direction cliff (the fast-tumbler gate)", "closed", False,
     ("s087_omega_dir_threshold", "fast-seed truth recovery needs ω-dir ≲1.25° (119 cliff)", "2026-05-21")),
    ("goal_blind-fast-invert", "branch", "goal_blind-pipeline",
     "Blind fast densify inversion (≤15-min, |ω|-blind)", "open", False,
     ("s100_5step_coverage_proto", "5-step coverage prototype; |ω|-clamp + anchor-cap inverts 116", "2026-05-28")),
    ("goal_windowed-photometry-polish", "question", "goal_blind-fast-invert",
     "Windowed-photometry polish: break the hard-shoot trap & discriminate", "open", False,
     ("s107_discrimination_test", "0/1107 Band A, 0 phantoms; s106 real but not yet production (basin ~5° in ω-dir)", "2026-05-28")),
    ("goal_anchor-cap-slow-tumbler", "branch", "goal_blind-pipeline",
     "Anchor-cap + |ω|-clamp inverts the slow-tumbler class", "open", False,
     ("s111_anchor_cap_116_invert_and_decimation_bug", "116 inverts (7-attractor multi-sol, truth q0 0.19°); slow-sweep 10/42/31 BROKEN by decimation bug", "2026-05-29")),
    ("goal_cross-cloud-multistart-budget", "branch", "goal_blind-pipeline",
     "Cross-cloud-multistart search cost on the aliased seed", "revivable", False,
     ("s108_cross_cloud_multistart_cost_wall", "discriminator reusable; cross-cloud SEARCH ~100× over budget on aliased 119", "2026-05-29")),

    # ---- CH8 literature ----
    ("goal_rf25-audit", "branch", "goal_literature",
     "Robinson–Frueh 2025 residual-threat paper audit (no encroachment)", "open", False,
     ("s073e_robinson_frueh_2025_full_audit", "RF25 does not pre-empt cat-4 framing; obtain RF74 conference companion", "2026-05-14")),
]

# ---------------------------------------------- experiment -> node classification
# Keyed by the leading sNNN token; sub-token overrides applied first.
EXPLICIT = {
    "s001": "goal_cost-at-truth", "s002": "goal_cost-at-truth", "s003": "goal_cost-at-truth",
    "s004": "goal_cost-at-truth", "s005": "goal_joint-polish", "s006": "goal_cost-at-truth",
    "s007": "goal_lc-omega-priors", "s008": "goal_lc-omega-priors", "s009": "goal_basin-width",
    "s010": "goal_cost-at-truth", "s011": "goal_joint-polish", "s012": "goal_cost-at-truth",
    "s013": "goal_cost-at-truth", "s014": "goal_cost-at-truth", "s015": "goal_joint-polish",
    "s016": "goal_joint-polish", "s017": "goal_cost-at-truth", "s018": "goal_cost-at-truth",
    "s019": "goal_ls-bracket", "s020": "goal_joint-polish", "s021": "goal_cost-at-truth",
    "s022": "goal_cost-at-truth", "s023": "goal_cloud-data-seedgen", "s024": "goal_lc-omega-priors",
    "s025": "goal_cloud-data-seedgen", "s027": "goal_cloud-data-seedgen", "s030": "goal_cloud-data-seedgen",
    "s031": "goal_joint-polish", "s032": "goal_joint-polish", "s033": "goal_cost-at-truth",
    "s034": "goal_joint-polish", "s035": "goal_joint-polish", "s036": "goal_joint-polish",
    "s037": "goal_joint-polish", "s038": "goal_cloud-data-seedgen", "s040": "goal_basin-width",
    "s042": "goal_basin-width", "s043": "goal_body-twin", "s044": "goal_body-twin",
    "s045": "goal_ls-bracket", "s046": "goal_ls-bracket", "s047": "goal_cloud-data-seedgen",
    "s048": "goal_cloud-data-seedgen", "s049": "goal_cloud-data-seedgen", "s050": "goal_cloud-data-seedgen",
    "s051": "goal_polhode-prior", "s052": "goal_polhode-prior", "s053": "goal_basin-width",
    "s054": "goal_blind-fast-invert", "s055": "goal_lc-omega-priors", "s056": "goal_anchor-selection",
    "s057": "goal_cloud-data-seedgen", "s058": "goal_cloud-data-seedgen", "s059": "goal_anchor-selection",
    "s060": "goal_anchor-selection", "s061": "goal_threading", "s062": "goal_jacobi-propagator",
    "s063": "goal_polhode-prior", "s064": "goal_joint-polish", "s065": "goal_convention-bug",
    "s066": "goal_convention-bug", "s067": "goal_convention-bug", "s068": "goal_joint-polish",
    "s069": "goal_densify-ndirs", "s070": "goal_joint-polish", "s071": "goal_jacobi-propagator",
    "s072": "goal_jacobi-propagator", "s073": "goal_l-vector-discriminator", "s074": "goal_jacobi-propagator",
    "s076": "goal_cross-cloud-discriminator", "s077": "goal_l-vector-discriminator",
    "s078": "goal_cross-cloud-discriminator", "s079": "goal_l-vector-discriminator",
    "s081": "goal_body-twin", "s082": "goal_joint-grid-pivot", "s083": "goal_omega-mag-basin",
    "s084": "goal_omega-mag-basin", "s085": "goal_anchor-accuracy-gate", "s086": "goal_anchor-accuracy-gate",
    "s087": "goal_anchor-accuracy-gate", "s088": "goal_bvp-shoot", "s089": "goal_third-anchor",
    "s090": "goal_third-anchor", "s091": "goal_cross-cloud-discriminator", "s092": "goal_cross-cloud-discriminator",
    "s093": "goal_cross-cloud-discriminator", "s094": "goal_cross-cloud-discriminator",
    "s095": "goal_cross-cloud-discriminator", "s096": "goal_cross-cloud-discriminator",
    "s097": "goal_gauge-times0", "s098": "goal_blind-fast-invert", "s099": "goal_blind-fast-invert",
    "s100": "goal_blind-fast-invert", "s105": "goal_windowed-photometry-polish",
    "s106": "goal_windowed-photometry-polish", "s107": "goal_windowed-photometry-polish",
    "s108": "goal_cross-cloud-multistart-budget", "s111": "goal_anchor-cap-slow-tumbler",
    "s112": "goal_anchor-cap-slow-tumbler",
}
# sub-token overrides (checked before the base lookup)
OVERRIDES = [
    ("s055d", "goal_polhode-prior"),
    ("s059j", "goal_cloud-data-seedgen"),
    ("s059k", "goal_densify-ndirs"),
    ("s073b", "goal_jacobi-propagator"),
    ("s073c", "goal_rf25-audit"),
    ("s073e", "goal_rf25-audit"),
    ("s073f", "goal_l-vector-discriminator"),
]


def assign(stem: str) -> str:
    for pfx, node in OVERRIDES:
        if stem.startswith(pfx):
            return node
    base = re.match(r"^(s\d+)", stem).group(1)
    return EXPLICIT[base]   # KeyError = unmapped experiment = coverage failure


def parse_frontmatter(md_path: Path) -> dict:
    text = md_path.read_text()
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end]
    try:
        import yaml
        return yaml.safe_load(block) or {}
    except Exception:
        # minimal fallback: scalar key: value lines only
        fm = {}
        for line in block.splitlines():
            m = re.match(r"^(\w+):\s*(.+)$", line)
            if m:
                fm[m.group(1)] = m.group(2).strip().strip('"')
        return fm


def build_node(spec) -> dict:
    nid, kind, parent, title, state, trunk, last = spec
    node = {
        "schema_version": "1.0.0",
        "id": nid,
        "kind": "goal_node",
        "node_kind": kind,
        "parent": parent,
        "title": title,
        "state": state,
        "spent": {"runs": 0, "wall_s": 0},
        "contract_refs": [],
        "child_runs": [],
        "is_trunk_artifact": trunk,
        "created_at": BACKFILL_DATE,
        "updated_at": BACKFILL_DATE,
    }
    if last:
        run, summary, at = last
        node["last_measured"] = {"run": run, "summary": summary, "at": at + "T00:00:00Z"}
    return node


def main() -> int:
    GOALS.mkdir(parents=True, exist_ok=True)
    (RO / "backfill").mkdir(parents=True, exist_ok=True)
    validator = Draft202012Validator(json.loads((SCHEMAS / "goal_node.schema.json").read_text()))

    node_ids = {s[0] for s in NODES}
    # emit nodes
    fails = 0
    for spec in NODES:
        node = build_node(spec)
        errs = list(validator.iter_errors(node))
        if errs:
            fails += 1
            print(f"NODE FAIL {node['id']}: {[e.message for e in errs]}")
        if node["parent"] is not None and node["parent"] not in node_ids:
            fails += 1
            print(f"NODE FAIL {node['id']}: parent {node['parent']} not a node")
        (GOALS / f"{node['id']}.json").write_text(json.dumps(node, indent=2, ensure_ascii=False) + "\n")

    # classify experiments + build manifest
    stems = sorted(p.stem for p in EXP.glob("*.md"))
    manifest, counts = [], {}
    for stem in stems:
        try:
            node = assign(stem)
        except KeyError as e:
            fails += 1
            print(f"UNMAPPED experiment {stem} (base {e})")
            continue
        if node not in node_ids:
            fails += 1
            print(f"BAD assignment {stem} -> {node} (not a node)")
            continue
        counts[node] = counts.get(node, 0) + 1
        fm = parse_frontmatter(EXP / f"{stem}.md")
        manifest.append({
            "id": stem,
            "goal_node": node,
            "writeup_ref": f"{WRITEUP_PREFIX}/{stem}.md",
            "type": fm.get("type"),
            "title": fm.get("title", stem),
            "created": str(fm.get("created", "")),
            "updated": str(fm.get("updated", "")),
            "confidence": fm.get("confidence"),
            "sources": fm.get("sources") or [],
            "related": fm.get("related") or [],
        })

    (RO / "backfill" / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    # derived index
    index = [{"id": s[0], "node_kind": s[1], "parent": s[2], "title": s[3],
              "state": s[4], "n_experiments": counts.get(s[0], 0)} for s in NODES]
    (GOALS / "_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")

    # report
    print(f"\nnodes: {len(NODES)}  experiments classified: {len(manifest)}/{len(stems)}")
    empty_leaves = [s[0] for s in NODES if s[1] in ("branch", "question") and counts.get(s[0], 0) == 0]
    if empty_leaves:
        print(f"WARNING empty leaf nodes: {empty_leaves}")
    print("\nper-node experiment counts:")
    for s in NODES:
        if s[1] in ("branch", "question"):
            print(f"  {counts.get(s[0],0):2d}  {s[0]}")
    print(f"\n{'ALL OK' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
