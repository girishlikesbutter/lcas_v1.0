"""s044 — validate ``lib.twin.canonical`` against algebra + the surrogate + hi-fi.

Per s043 the body-twin LC equivalence is bit-exact in hi-fi; the
``canonical`` helper deduplicates the (q0, ω) search space accordingly.
This script exercises the algebra (idempotency, twin-pair → same canonical,
involution) and the LC-equivalence claim (synthesise a non-canonical
state, canonicalise, compare LCs).

Algebraic checks need no I/O. The LC-equivalence check uses the truth
SPICE state for a probe seed (no truth comparison — we just check that
``LC(state) == LC(canonical(state))`` for an arbitrary state and the
canonicaliser's output).

Outputs:
    results/s044_canonical_validation/run.log
    results/s044_canonical_validation/summary.json
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))

from lib.twin import (  # noqa: E402
    Q_180X,
    R_180X,
    canonical,
    canonical_batch,
    is_canonical,
    is_canonical_batch,
    quat_mul,
    twin,
)

OUT_DIR = SURVEY_DIR / "results" / "s044_canonical_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20260506
N_RANDOM = 10_000  # algebraic batch size
PROBE_SEED = 89  # for hi-fi LC equivalence (s043 covered 23, 28, 89)


# --------------------------------------------------------------------------- #
# Algebraic checks
# --------------------------------------------------------------------------- #


def _random_unit_quat(rng: np.random.Generator, n: int) -> np.ndarray:
    """Random unit quaternions in (w, x, y, z) order, uniform on S3."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def _random_omega(rng: np.random.Generator, n: int, mag_dps: float = 1.0) -> np.ndarray:
    """Random angular-velocity vectors, isotropic direction, magnitude up to mag_dps."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    mags = rng.uniform(0.05, mag_dps, size=n) * (np.pi / 180.0)
    return v * mags[:, None]


def check_quat_mul_vs_scipy(rng: np.random.Generator) -> dict:
    """quat_mul(a, b) (wxyz) must agree with scipy Rotation.from_quat product (xyzw)."""
    q1 = _random_unit_quat(rng, 200)
    q2 = _random_unit_quat(rng, 200)
    max_err = 0.0
    for a, b in zip(q1, q2):
        ours = quat_mul(a, b)
        # scipy: xyzw input order; product is (R_a * R_b) which equals q_a ⊗ q_b
        # (Hamilton, scalar-first) — but scipy's `*` on Rotations is
        # right-to-left composition. Rotation.from_quat([x,y,z,w]) returns the
        # rotation; product of two Rotations (a * b) yields the rotation
        # whose quaternion is q_a ⊗ q_b (Hamilton). Verified by smoke test:
        # we just take the result and compare modulo sign (q ~ -q).
        Ra = Rotation.from_quat([a[1], a[2], a[3], a[0]])
        Rb = Rotation.from_quat([b[1], b[2], b[3], b[0]])
        Rprod = Ra * Rb
        x, y, z, w = Rprod.as_quat()
        scipy_q = np.array([w, x, y, z])
        e = min(
            np.linalg.norm(ours - scipy_q),
            np.linalg.norm(ours + scipy_q),  # sign ambiguity
        )
        max_err = max(max_err, e)
    return {"max_quat_mul_vs_scipy_err": max_err, "passed": max_err < 1e-12}


def check_twin_involution(rng: np.random.Generator) -> dict:
    """twin(twin(q, ω)) must equal (-q, ω) (or +q, ω after sign-normalisation)."""
    q = _random_unit_quat(rng, 500)
    w = _random_omega(rng, 500, mag_dps=2.0)
    max_err_q = 0.0
    max_err_w = 0.0
    for qi, wi in zip(q, w):
        q1, w1 = twin(qi, wi)
        q2, w2 = twin(q1, w1)
        # Expected: q2 == -qi (modulo float arithmetic), w2 == wi.
        e_q = min(np.linalg.norm(q2 - qi), np.linalg.norm(q2 + qi))
        e_w = np.linalg.norm(w2 - wi)
        max_err_q = max(max_err_q, e_q)
        max_err_w = max(max_err_w, e_w)
    return {
        "max_q_err": max_err_q,
        "max_omega_err": max_err_w,
        "passed": max_err_q < 1e-13 and max_err_w < 1e-13,
    }


def check_canonical_idempotent(rng: np.random.Generator) -> dict:
    """canonical(canonical(x)) must return bit-equal arrays."""
    q = _random_unit_quat(rng, N_RANDOM)
    w = _random_omega(rng, N_RANDOM, mag_dps=2.0)
    qc, wc = canonical_batch(q, w)
    qcc, wcc = canonical_batch(qc, wc)
    max_dq = float(np.max(np.abs(qcc - qc)))
    max_dw = float(np.max(np.abs(wcc - wc)))
    return {
        "max_q_diff": max_dq,
        "max_omega_diff": max_dw,
        "passed": max_dq == 0.0 and max_dw == 0.0,
    }


def check_canonical_pair_collapses(rng: np.random.Generator) -> dict:
    """canonical(twin(x)) must equal canonical(x) bit-for-bit."""
    q = _random_unit_quat(rng, N_RANDOM)
    w = _random_omega(rng, N_RANDOM, mag_dps=2.0)
    # Compute twins via batch arithmetic — same formula as in canonical_batch.
    q_twin = np.column_stack([-q[:, 1], q[:, 0], -q[:, 3], q[:, 2]])
    w_twin = w * np.array([1.0, -1.0, -1.0])

    qc1, wc1 = canonical_batch(q, w)
    qc2, wc2 = canonical_batch(q_twin, w_twin)
    max_dq = float(np.max(np.abs(qc1 - qc2)))
    max_dw = float(np.max(np.abs(wc1 - wc2)))
    return {
        "max_q_diff": max_dq,
        "max_omega_diff": max_dw,
        "passed": max_dq == 0.0 and max_dw == 0.0,
    }


def check_canonical_membership(rng: np.random.Generator) -> dict:
    """is_canonical(canonical(x)) must be True for all sampled states."""
    q = _random_unit_quat(rng, N_RANDOM)
    w = _random_omega(rng, N_RANDOM, mag_dps=2.0)
    qc, wc = canonical_batch(q, w)
    keep = is_canonical_batch(qc, wc)
    return {
        "n_total": int(keep.size),
        "n_canonical": int(keep.sum()),
        "passed": bool(keep.all()),
    }


def check_scalar_vs_batch(rng: np.random.Generator) -> dict:
    """Scalar canonical and canonical_batch must produce bit-identical outputs."""
    q = _random_unit_quat(rng, 1000)
    w = _random_omega(rng, 1000, mag_dps=2.0)
    qc_b, wc_b = canonical_batch(q, w)
    max_dq = 0.0
    max_dw = 0.0
    for i in range(q.shape[0]):
        qc_s, wc_s = canonical(q[i], w[i])
        max_dq = max(max_dq, float(np.max(np.abs(qc_s - qc_b[i]))))
        max_dw = max(max_dw, float(np.max(np.abs(wc_s - wc_b[i]))))
    return {
        "max_q_diff": max_dq,
        "max_omega_diff": max_dw,
        "passed": max_dq == 0.0 and max_dw == 0.0,
    }


def check_constants() -> dict:
    """Q_180X = (0,1,0,0); R_180X = diag(1,-1,-1); R_180X must be the rotation
    matrix of Q_180X, and (Q_180X)^2 = -1 (Hamilton)."""
    R = Rotation.from_quat([Q_180X[1], Q_180X[2], Q_180X[3], Q_180X[0]]).as_matrix()
    err_R = float(np.max(np.abs(R - R_180X)))
    q_sq = quat_mul(Q_180X, Q_180X)
    expected_neg_id = np.array([-1.0, 0.0, 0.0, 0.0])
    err_sq = float(np.max(np.abs(q_sq - expected_neg_id)))
    return {
        "max_R_err": err_R,
        "max_q_sq_neg_identity_err": err_sq,
        "passed": err_R < 1e-15 and err_sq < 1e-15,
    }


# --------------------------------------------------------------------------- #
# Hi-fi LC equivalence: arbitrary non-canonical state vs its canonical rep
# --------------------------------------------------------------------------- #


def check_hifi_lc_equivalence(rng: np.random.Generator) -> dict:
    """Synthesise a non-canonical (q, ω); compare hi-fi LC at (q, ω) vs canonical(q, ω).

    Per s043 the body-twin map is bit-exact in hi-fi; canonical(q, ω) is
    either (q, ω) itself or twin(q, ω), so this check should also be
    bit-exact (or trivially zero when the synthesised state is already
    canonical).
    """
    from lib.hifi_render import build_context, render_hifi  # noqa: E402

    print(f"  Building hi-fi context for seed {PROBE_SEED} ...", flush=True)
    ctx = build_context(PROBE_SEED)
    # Synthesise an arbitrary non-canonical state. Sample uniform q0 and a
    # body-frame ω, then force ω_y < 0 so the state is non-canonical.
    q = _random_unit_quat(rng, 1)[0]
    w = _random_omega(rng, 1, mag_dps=1.0)[0]
    if w[1] > 0:
        w[1] = -w[1]  # ensure non-canonical
    assert not is_canonical(q, w), "probe state should be non-canonical by construction"

    qc, wc = canonical(q, w)
    assert is_canonical(qc, wc), "canonical(state) should be canonical"

    print(f"  Rendering hi-fi LC at synthetic (q, ω) ...", flush=True)
    t0 = time.time()
    mag_orig = render_hifi(q, w, ctx)
    t_orig = time.time() - t0
    print(f"  Rendering hi-fi LC at canonical(q, ω) ...", flush=True)
    t0 = time.time()
    mag_canon = render_hifi(qc, wc, ctx)
    t_canon = time.time() - t0

    valid = np.isfinite(mag_orig) & np.isfinite(mag_canon)
    diff = mag_orig[valid] - mag_canon[valid]
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff ** 2)))
    rho = rms / 0.05
    return {
        "probe_seed": PROBE_SEED,
        "q_orig_wxyz": q.tolist(),
        "omega_orig_rad": w.tolist(),
        "q_canon_wxyz": qc.tolist(),
        "omega_canon_rad": wc.tolist(),
        "n_valid": int(valid.sum()),
        "max_abs_mag": max_abs,
        "rms_mag": rms,
        "rho": rho,
        "render_orig_s": t_orig,
        "render_canon_s": t_canon,
        # Same s043 threshold: noise floor 0.05 mag.
        "passed": max_abs < 0.05,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main():
    print(f"=== s044 canonical validation ===", flush=True)
    rng = np.random.default_rng(RNG_SEED)

    results = {}

    print("\n[1/7] Constants (Q_180X, R_180X, Q_180X^2 = -1) ...", flush=True)
    results["constants"] = check_constants()
    print(f"      {results['constants']}", flush=True)

    print("\n[2/7] quat_mul vs scipy product ...", flush=True)
    results["quat_mul_vs_scipy"] = check_quat_mul_vs_scipy(rng)
    print(f"      {results['quat_mul_vs_scipy']}", flush=True)

    print("\n[3/7] twin(twin(x)) involution (modulo q sign) ...", flush=True)
    results["twin_involution"] = check_twin_involution(rng)
    print(f"      {results['twin_involution']}", flush=True)

    print("\n[4/7] canonical(canonical(x)) idempotency ...", flush=True)
    results["idempotency"] = check_canonical_idempotent(rng)
    print(f"      {results['idempotency']}", flush=True)

    print("\n[5/7] canonical(twin(x)) == canonical(x) (pair collapses) ...", flush=True)
    results["pair_collapses"] = check_canonical_pair_collapses(rng)
    print(f"      {results['pair_collapses']}", flush=True)

    print("\n[6/7] is_canonical(canonical(x)) for all x ...", flush=True)
    results["membership"] = check_canonical_membership(rng)
    print(f"      {results['membership']}", flush=True)

    print("\n[7/7] scalar canonical vs canonical_batch agreement ...", flush=True)
    results["scalar_vs_batch"] = check_scalar_vs_batch(rng)
    print(f"      {results['scalar_vs_batch']}", flush=True)

    algebraic_passed = all(r["passed"] for r in results.values())
    print(
        f"\nAlgebraic checks: {'ALL PASS' if algebraic_passed else 'FAIL'}",
        flush=True,
    )

    print("\n[hi-fi] Non-canonical (q, ω) vs canonical(q, ω) on probe seed ...", flush=True)
    results["hifi_lc_equivalence"] = check_hifi_lc_equivalence(rng)
    r = results["hifi_lc_equivalence"]
    print(
        f"      max|Δmag| = {r['max_abs_mag']:.3e}, RMS = {r['rms_mag']:.3e}, "
        f"ρ = {r['rho']:.3e}, passed = {r['passed']}",
        flush=True,
    )

    def _coerce(o):
        # Make numpy scalars JSON-serialisable.
        if isinstance(o, dict):
            return {k: _coerce(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_coerce(x) for x in o]
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return o

    summary = _coerce(
        {
            "n_random": N_RANDOM,
            "rng_seed": RNG_SEED,
            "probe_seed": PROBE_SEED,
            "checks": results,
            "all_passed": all(r["passed"] for r in results.values()),
        }
    )

    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)

    if not summary["all_passed"]:
        print("\n=== VERDICT: FAIL ===", flush=True)
        sys.exit(1)
    print("\n=== VERDICT: ALL PASS ===", flush=True)


if __name__ == "__main__":
    main()
