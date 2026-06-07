#!/usr/bin/env python3
"""Generate the W-B spine claim cards (PLAN §2 Q8; STATUS.md step 2).

Authoring (synthesis) lives here, in one reviewable place, like build_goal_tree.py.
Writes claim_card instances to research_os/claims/{research,preferences}/.

Design rules honoured:
- supporting_runs / refuting_runs use FULL run-record stems (referential integrity;
  the example cards used short ids — STATUS.md note).
- trust_stamp.oracle_clean is COMPUTED from the supporting run records (= all
  supporting runs oracle_clean), not hand-typed. The s058/s059 retraction rule:
  a card resting on an oracle-tainted run is not oracle_clean.
- depends_on encodes the substrate the LIVE evidence rests on. A claim whose
  evidence is pre-fix-only (propagator@1.0.0) is modelled status=needs_replication
  (the blast-radius cure: the 2.0.0 ω-sign bump invalidated that evidence).
- Corrections are POINTERS: refutation flips status + adds superseded_by/supersedes.

Idempotent: rewrites the same files each run. Run from repo root.
"""
import json, os, glob

REC_DIR = "research_os/records"
OUT_RESEARCH = "research_os/claims/research"
OUT_PREF = "research_os/claims/preferences"
STAMP = "2026-06-01T00:00:00Z"  # backfill authoring date (deterministic)

# --- load run records to compute oracle_clean from supporting runs ---
REC = {}
for f in glob.glob(os.path.join(REC_DIR, "*.json")):
    d = json.load(open(f))
    REC[d["id"]] = d


def oclean(supporting):
    """True iff every supporting run is oracle_clean (vacuously true if empty)."""
    return all(REC[r]["oracle_clean"] for r in supporting if r in REC)


# --- RESEARCH DECK ---------------------------------------------------------
# Each entry: the synthesis. oracle_clean is filled below from REC.
research = [

    # ---- supersede chain 1: the seed-119 blocker (s099 -> s100/s105) ----
    {
        "id": "claim_anchor-aliasing-is-the-blocker",
        "statement": "The binding blocker to blind inversion on seed 119 is anchor-baseline aliasing (the single finite-diff shoot aliasing the truth pair).",
        "supporting_runs": ["s099_blind_fast_invert_116"],
        "refuting_runs": ["s100_5step_coverage_proto"],
        "confidence": "low",
        "scope": {"N": 1, "kind": "N1"},
        "depends_on": ["propagator@2.0.0", "surrogate@v2", "scorer@1.0.0"],
        "status": "superseded",
        "superseded_by": "claim_hard-shoot-trap-is-the-blocker",
        "links_to_writeups": ["notebooks/inversion/survey/experiments/s099_blind_fast_invert_116.md"],
    },
    {
        "id": "claim_hard-shoot-trap-is-the-blocker",
        "statement": "The binding blocker on seed 119 is the HARD-SHOOT TRAP, not anchor-baseline aliasing: s100 showed 12M coverage closes anchor starvation yet end-to-end still tops out Band D, and s105 showed hard-connecting ~1deg-off endpoints over 3.1 turns bakes a 4.25deg omega error -> Band D, while LM multistart on the EXACT pair is Band A.",
        "supporting_runs": ["s100_5step_coverage_proto", "s105_pairs_to_omega_decomposition"],
        "refuting_runs": [],
        "confidence": "medium",
        "scope": {"N": 1, "kind": "N1"},
        "depends_on": ["propagator@2.0.0", "surrogate@v2", "scorer@1.0.0"],
        "status": "live",
        "supersedes": "claim_anchor-aliasing-is-the-blocker",
        "links_to_writeups": [
            "notebooks/inversion/survey/experiments/s100_5step_coverage_proto.md",
            "notebooks/inversion/survey/experiments/s105_pairs_to_omega_decomposition.md",
        ],
    },

    # ---- supersede chain 2: full-LC RMSE (times0 gauge bug reversed it) ----
    {
        "id": "claim_full-lc-rmse-fails-needs-windowing",
        "statement": "Full-LC RMSE fails to rank truth and windowed photometry is required to discriminate the truth basin. (Pre-audit conclusion of s093/s094/s095 — later shown to be an artefact of the times[0]==0 propagator-gauge bug, which mis-referenced trajectories ~108deg.)",
        "supporting_runs": [],
        "refuting_runs": ["s097_times0_gauge_bug_audit"],
        "confidence": "low",
        "scope": {"N": 1, "kind": "N1"},
        "depends_on": ["propagator@2.0.0", "surrogate@v2", "scorer@1.0.0"],
        "status": "superseded",
        "superseded_by": "claim_full-lc-rmse-discriminator-works",
        "links_to_writeups": ["notebooks/inversion/survey/experiments/s097_times0_gauge_bug_audit.md"],
    },
    {
        "id": "claim_full-lc-rmse-discriminator-works",
        "statement": "After the times[0]==0 gauge fix, full-LC RMSE is a STRONG discriminator: it ranks truth 4/79k on seed 116; windowing HURTS; density is the lever. (N=1 seed 116.)",
        "supporting_runs": ["s093_fullc_score_survivors", "s097_times0_gauge_bug_audit"],
        "refuting_runs": [],
        "confidence": "medium",
        "scope": {"N": 1, "kind": "N1"},
        "depends_on": ["propagator@2.0.0", "surrogate@v2", "scorer@1.0.0"],
        "status": "live",
        "supersedes": "claim_full-lc-rmse-fails-needs-windowing",
        "links_to_writeups": ["notebooks/inversion/survey/experiments/s093_fullc_score_survivors.md"],
    },

    # ---- retraction: cloud-data PROVEN-on-89 (oracle injection) ----
    {
        "id": "claim_cloud-data-proven-on-seed89",
        "statement": "The cloud-data architecture is PROVEN to land Band A on seed 89. RETRACTED: the Band A rows were oracle-injected truth-cluster polish (truth looked up by cluster_id); without injection truth ranks 107/325 (seed 89) / 390/25k (seed 28). The headline yield was an oracle-leak artefact, not an architecture result.",
        "supporting_runs": [],
        "refuting_runs": ["s058_lm_polish_clusters"],
        "confidence": "low",
        "scope": {"N": 1, "kind": "N1"},
        "depends_on": ["propagator@1.0.0", "surrogate@v2", "scorer@1.0.0"],
        "status": "retracted",
        "force_oracle_clean": False,  # retraction rule: rests on oracle-tainted run
        "links_to_writeups": [
            "notebooks/inversion/survey/experiments/s058_lm_polish_clusters.md",
            "memory/feedback_oracle_injection_taints_yield.md",
        ],
    },

    # ---- needs_replication umbrella: pre-fix s001-s066 ----
    {
        "id": "claim_prefix-numerical-findings-stale",
        "statement": "All pre-fix s001-s066 SPECIFIC NUMBERS (seed ids, rho-values, basin widths) rest on propagator 1.0.0 (the omega-sign bug: dq/dt=+0.5 instead of -0.5, L_J2000 drift 36-137% over 60-min LCs). The 2.0.0 fix (commit d5705ff) restores conservation and regenerates the cohort; methodology and qualitative findings survive, the numbers need re-measurement.",
        "supporting_runs": ["s066_lj2000_nonconservation", "s067_postfix_propagator_validation"],
        "refuting_runs": [],
        "confidence": "high",
        "scope": {"N": 120, "kind": "cohort"},
        "depends_on": ["propagator@1.0.0", "surrogate@v2", "scorer@1.0.0"],
        "status": "needs_replication",
        "links_to_writeups": [
            "memory/project_convention_bug_2026_04_30.md",
            "memory/feedback_propagator_omega_sign_convention.md",
        ],
    },

    # ---- load-bearing LIVE (post-fix anchored) ----
    {
        "id": "claim_body-twin-halves-search-space",
        "statement": "Antipodal body-frame quaternion twins produce bit-exact light curves (max|delta|=2.29e-08 mag, 6 orders below noise; re-confirmed in hi-fi post-fix by s081). Deduplicating via a canonical hemisphere halves the search space at every pipeline stage. Convention-independent symmetry.",
        "supporting_runs": ["s043_twin_hifi_verify", "s081_hifi_rho_bands_twin"],
        "refuting_runs": [],
        "confidence": "high",
        "scope": {"N": 3, "kind": "N1"},
        "depends_on": ["propagator@2.0.0", "surrogate@v2", "scorer@1.0.0"],
        "status": "live",
        "links_to_writeups": ["notebooks/inversion/survey/experiments/s043_twin_hifi_verify.md"],
    },
    {
        "id": "claim_omega-bracket-held-out-20-of-20",
        "statement": "The blind s019 LS-|omega| harmonic-division bracket, clamped to the physical [0.1,1.5] deg/s prior and applied HELD-OUT to cohort seeds 100-119, brackets truth-|omega| on 20/20 seeds. Enables blind two-stage inversion (116 in 705s).",
        "supporting_runs": ["s099_blind_fast_invert_116"],
        "refuting_runs": [],
        "confidence": "high",
        "scope": {"N": 20, "kind": "cohort"},
        "depends_on": ["propagator@2.0.0", "surrogate@v2", "scorer@1.0.0"],
        "status": "live",
        "links_to_writeups": [
            "notebooks/inversion/survey/experiments/s099_blind_fast_invert_116.md",
            "notebooks/inversion/survey/experiments/s019_ls_bracket_omega_mag.md",
        ],
    },
    {
        "id": "claim_windowed-photometry-breaks-hard-shoot-trap",
        "statement": "On seed 119, free-omega LM-minimization of photometry over a forward A->C window (window-as-objective) escapes the hard-shoot trap and recovers Band A (rho 0.0300, omega-err 0.57deg, |omega| 1.5025 deg/s), where hard-connecting the endpoints gives Band D (rho 0.687). SCOPE: N=1, oracle-seeded pair — discrimination against far/phantom pairs is UNTESTED.",
        "supporting_runs": ["s106_hybrid_loss_polish"],
        "refuting_runs": [],
        "confidence": "medium",
        "scope": {"N": 1, "kind": "N1"},
        "depends_on": ["propagator@2.0.0", "surrogate@v2", "scorer@1.0.0"],
        "status": "live",
        "gates_passed": ["conservation-smoke"],
        "links_to_writeups": ["notebooks/inversion/survey/experiments/s106_hybrid_loss_polish.md"],
    },

    # ---- load-bearing but PRE-FIX-ONLY -> needs_replication (blast radius) ----
    {
        "id": "claim_pol-diam-predicts-basin-width",
        "statement": "Polhode DIAMETER predicts joint-LM basin width better than |omega| (Spearman rho=-0.94, p<0.001, cohort n=100) and far better than the polhode LABEL (rho=+0.11, uncorrelated). EVIDENCE IS PRE-FIX (s042/s053 ran on propagator 1.0.0); basin-width numbers are exactly the class the omega-sign fix invalidated -> needs re-measurement under 2.0.0.",
        "supporting_runs": ["s053_cohort_polhode_survey", "s042_basin_radius_cohort"],
        "refuting_runs": [],
        "confidence": "medium",
        "scope": {"N": 100, "kind": "cohort"},
        "depends_on": ["propagator@1.0.0", "surrogate@v2", "scorer@1.0.0"],
        "status": "needs_replication",
        "links_to_writeups": [
            "notebooks/inversion/survey/experiments/s053_cohort_polhode_survey.md",
            "memory/project_omega_mag_basin_scales_with_omega.md",
        ],
    },
    {
        "id": "claim_lc-only-omega-priors-dead",
        "statement": "The entire LC-only omega-magnitude-prior class is structurally dead: peak-spacing gives 7.4% oracle MAPE, a 28-feature regression 16.4% LOO MAPE -- neither localizes |omega|. The class-level conclusion likely survives the fix, but the quantitative evidence (s003/s007/s008) ran on propagator 1.0.0 and is unreplicated under 2.0.0.",
        "supporting_runs": ["s007_omega_mag_peak_spacing_pilot", "s008_lc_feature_regression_omega"],
        "refuting_runs": [],
        "confidence": "medium",
        "scope": {"N": 100, "kind": "cohort"},
        "depends_on": ["propagator@1.0.0", "surrogate@v2", "scorer@1.0.0"],
        "status": "needs_replication",
        "links_to_writeups": [
            "notebooks/inversion/survey/experiments/s008_lc_feature_regression_omega.md",
            "memory/feedback_lc_spectral_omega_prior_dead.md",
        ],
    },
]

# --- PREFERENCE DECK -------------------------------------------------------
preferences = [
    {
        "id": "claim_omega-reported-in-degps",
        "statement": "Report |omega| in deg/s, never rad/s (x57.29578). Standing preference -- applies to every metric, plot axis, and writeup.",
        "links_to_writeups": ["memory/feedback_omega_mag_units_degrees.md"],
    },
    {
        "id": "claim_surrogate-first-hifi-last",
        "statement": "Every intermediate step (search, ranking, clustering, LM polish, sanity checks) runs on the surrogate. Hi-fi is reserved for a SINGLE render per polished basin winner that passes surrogate-rho < 4, to classify the rho-band -- it is never an optimisation step.",
        "links_to_writeups": [
            "memory/feedback_surrogate_first_hifi_last.md",
            "memory/feedback_lm_cost_use_surrogate.md",
        ],
    },
    {
        "id": "claim_save-intermediate-checkpoints",
        "statement": "EVERY script saves q0, omega, errors and predicted hi-fi LCs to NPZ/JSON after each expensive stage. Design the save blocks BEFORE the computation code; re-score from cached checkpoints rather than re-running.",
        "links_to_writeups": [
            "memory/feedback_checkpoint_always.md",
            "memory/feedback_save_results.md",
            "memory/feedback_checkpoint_design_first.md",
        ],
    },
    {
        "id": "claim_no-parallel-cpu-heavy-runs",
        "statement": "Do not stack CPU-saturating jobs. Set OMP/OPENBLAS/MKL threads=1 before Pool(N) for the surrogate; hi-fi Pool needs gc.collect() between renders (Pool(24) OOMs without it on the 30 GB box).",
        "links_to_writeups": [
            "memory/feedback_no_parallel_cpu.md",
            "memory/feedback_blas_threads_for_pool.md",
            "memory/feedback_pool_size_with_trimesh_gc.md",
        ],
    },
    {
        "id": "claim_physical-omega-bracket",
        "statement": "Intersect every blind LS-|omega| bracket with the physical [0.1,1.5] deg/s prior. Unclamped, s019 gave a 34.9 deg/s ceiling (23x too high) -> a fake 72-winding multi-shoot. Truth seed-119 |omega| = 1.482 deg/s.",
        "links_to_writeups": ["memory/feedback_omega_prior_physical_bracket.md"],
    },
]


def build_research(c):
    forced = c.pop("force_oracle_clean", None)
    oc = forced if forced is not None else oclean(c["supporting_runs"])
    card = {
        "schema_version": "1.0.0",
        "id": c["id"],
        "kind": "claim_card",
        "deck": "research",
        "statement": c["statement"],
        "supporting_runs": c["supporting_runs"],
        "refuting_runs": c["refuting_runs"],
        "trust_stamp": {
            "confidence": c["confidence"],
            "gates_passed": c.get("gates_passed", []),
            "oracle_clean": oc,
        },
        "scope": c["scope"],
        "depends_on": c["depends_on"],
        "status": c["status"],
        "superseded_by": c.get("superseded_by"),
        "supersedes": c.get("supersedes"),
        "links_to_writeups": c.get("links_to_writeups", []),
        "created_at": STAMP,
        "updated_at": STAMP,
    }
    return card


def build_pref(c):
    return {
        "schema_version": "1.0.0",
        "id": c["id"],
        "kind": "claim_card",
        "deck": "preference",
        "statement": c["statement"],
        "supporting_runs": [],
        "refuting_runs": [],
        "trust_stamp": {"confidence": "high", "gates_passed": [], "oracle_clean": True},
        "scope": {"N": 0, "kind": "N1"},
        "depends_on": [],
        "status": "live",
        "superseded_by": None,
        "supersedes": None,
        "links_to_writeups": c.get("links_to_writeups", []),
        "confirmed_by": "girish",
        "created_at": STAMP,
        "updated_at": STAMP,
    }


def write(card, outdir):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, card["id"] + ".json")
    with open(path, "w") as fh:
        json.dump(card, fh, indent=2)
        fh.write("\n")
    return path


def main():
    n = 0
    for c in research:
        p = write(build_research(c), OUT_RESEARCH)
        n += 1
        print("research  ", os.path.basename(p))
    for c in preferences:
        p = write(build_pref(c), OUT_PREF)
        n += 1
        print("preference", os.path.basename(p))
    print(f"\nwrote {n} claim cards")


if __name__ == "__main__":
    main()
