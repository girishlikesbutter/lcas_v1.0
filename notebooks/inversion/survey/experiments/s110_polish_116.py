"""s110 — surrogate-v2 full-LC polish of the s100 capped-116 blind candidates.

s100 (clamp + anchor-cap) returns 15 BLIND coarse-rep candidates for seed 116
(results/s100/seed116/invert.npz); coarse-rep RMSE means nothing for band/multi-
sol acceptance until polished. This polishes each candidate with the canonical
production polish (s058.lm_polish: surrogate-v2 full-LC, 6-DOF rotvec+omega, LM)
wrapped in the multi-mag-start [0,±3,±6]% the memory prescribes, then returns the
DISTINCT attractor set (the multi-solution goal). NO hi-fi — surrogate only.

Candidates are anchor states (q_a, omega) at ep_a; back-propagate to t0 first,
then polish. Errors vs truth are DIAGNOSTIC labels only (truth-recovery is not
the gate; the multi-solution set is). All |omega| in deg/s.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, json, time, importlib
from pathlib import Path
from multiprocessing import get_context
import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY)); sys.path.insert(0, str(SURVEY / "experiments"))
import lib.traj_load as tl
from lib.hifi_render import build_context
from lib.shoot import m048_inertia, omega_dir_err_deg
from lib.jacobi_propagator import propagate_jacobi_path2
s058 = importlib.import_module("s058_lm_polish_clusters")  # reuse lm_polish

R2D = 180.0 / np.pi
SEED = int(os.environ.get("S110_SEED", 116))
INERTIA = m048_inertia()
MAG_STARTS = (1.0, 0.94, 0.97, 1.03, 1.06)   # [0, -6, -3, +3, +6] % on |omega|
N_WORK = 24
OUT = SURVEY / "results" / "s110"; OUT.mkdir(parents=True, exist_ok=True)

_CTX = _TARGET = None


def state_at_t0(q_a, w_a, times0, ep_a):
    if ep_a == 0:
        return np.asarray(q_a, float), np.asarray(w_a, float)
    tb = (times0[:ep_a + 1] - times0[ep_a])[::-1]
    qb, wb = propagate_jacobi_path2(np.asarray(q_a, float), np.asarray(w_a, float), INERTIA, tb)
    return qb[-1], wb[-1]


def _winit(ctx, target):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    global _CTX, _TARGET
    _CTX, _TARGET = ctx, target


def _polish_job(job):
    i, scale, q0, w0 = job
    res = s058.lm_polish(np.asarray(q0), np.asarray(w0) * scale, _CTX, _TARGET,
                         label=f"cand{i}_s{scale}")
    return i, scale, res


def main():
    t0 = time.time()
    ctx = get_context("fork")
    dz = np.load(SURVEY / "results" / "s100" / f"seed{SEED:03d}" / "invert.npz", allow_pickle=True)
    qa = dz["qa"]; om = dz["omega"]; ep_a = int(dz["ep_a"]); times0 = dz["times0"]
    floor = float(dz["truth_floor"])
    d = tl.load_truth(SEED)
    q0_t = d["q0_wxyz"].astype(float); w0_t = d["omega0_rad"].astype(float)
    wmag_t = np.linalg.norm(w0_t) * R2D

    print(f"===== s110 surrogate-v2 polish | seed {SEED} | {len(qa)} candidates x "
          f"{len(MAG_STARTS)} mag-starts | truth |w|={wmag_t:.4f} deg/s =====", flush=True)
    print(f"s100 surrogate truth-floor (full-LC RMSE) = {floor:.4f}", flush=True)

    # back-propagate anchor states -> t0 seeds
    seeds_t0 = [state_at_t0(qa[i], om[i], times0, ep_a) for i in range(len(qa))]

    hifi_ctx = build_context(seed=SEED)
    target = hifi_ctx["mag_hifi_truth"]

    jobs = [(i, s, seeds_t0[i][0], seeds_t0[i][1]) for i in range(len(qa)) for s in MAG_STARTS]
    print(f"polishing {len(jobs)} (candidate x mag-start) jobs on Pool({N_WORK}) ...", flush=True)
    ts = time.time()
    results = []
    with ctx.Pool(N_WORK, initializer=_winit, initargs=(hifi_ctx, target)) as p:
        for r in p.imap_unordered(_polish_job, jobs):
            results.append(r)
    print(f"polish done in {time.time()-ts:.0f}s", flush=True)

    # best mag-start per candidate (min surrogate MSE)
    best = {}
    for i, scale, res in results:
        if i not in best or res["surrogate_mse_polished"] < best[i]["surrogate_mse_polished"]:
            best[i] = res
    polished = []
    for i in sorted(best):
        r = best[i]
        q0p = np.array(r["q0_pol_wxyz"]); w0p = np.array(r["om0_pol_rad"])
        q0_err = float(2 * np.degrees(np.arccos(min(1.0, abs(float(q0p @ q0_t))))))
        dir_err = float(omega_dir_err_deg(w0p, w0_t))
        wmag = float(np.linalg.norm(w0p) * R2D)
        polished.append(dict(cand=i, surr_rho=r["surrogate_rho_polished"],
                             surr_rmse=float(np.sqrt(r["surrogate_mse_polished"])),
                             ratio_floor=float(np.sqrt(r["surrogate_mse_polished"]) / floor),
                             q0_err=q0_err, dir_err=dir_err, wmag=wmag,
                             wmag_err=wmag - wmag_t,
                             q0_pol=q0p.tolist(), w0_pol=w0p.tolist()))

    # cluster distinct attractors: same basin if q0_err-to-each-other<8 AND dir<8
    polished_sorted = sorted(polished, key=lambda r: r["surr_rmse"])
    clusters = []
    for r in polished_sorted:
        q0r = np.array(r["q0_pol"]); w0r = np.array(r["w0_pol"])
        placed = False
        for c in clusters:
            q0c = np.array(c["rep"]["q0_pol"]); w0c = np.array(c["rep"]["w0_pol"])
            dq = 2 * np.degrees(np.arccos(min(1.0, abs(float(q0r @ q0c)))))
            dd = omega_dir_err_deg(w0r, w0c)
            if dq < 8.0 and dd < 8.0:
                c["members"].append(r["cand"]); placed = True; break
        if not placed:
            clusters.append(dict(rep=r, members=[r["cand"]]))

    print(f"\n--- ALL polished candidates (surrogate, sorted by RMSE) ---", flush=True)
    print(f"{'cand':>4} {'surr_RMSE':>9} {'xfloor':>6} {'surr_rho':>8} {'q0_err':>7} "
          f"{'dir_err':>7} {'|w|':>7} {'dw(deg/s)':>9}", flush=True)
    for r in polished_sorted:
        print(f"{r['cand']:>4} {r['surr_rmse']:>9.4f} {r['ratio_floor']:>6.1f} {r['surr_rho']:>8.3f} "
              f"{r['q0_err']:>7.2f} {r['dir_err']:>7.2f} {r['wmag']:>7.4f} {r['wmag_err']:>+9.4f}", flush=True)

    print(f"\n--- DISTINCT ATTRACTORS (multi-solution set) ---", flush=True)
    for j, c in enumerate(clusters):
        r = c["rep"]
        twin = "body-twin-ish" if r["dir_err"] > 150 else ("truth-near" if r["q0_err"] < 15 and r["dir_err"] < 15 else "alt-basin")
        print(f"  A{j+1}: surr_rho {r['surr_rho']:.3f} ({r['surr_rmse']:.4f}, {r['ratio_floor']:.1f}x floor) | "
              f"q0_err {r['q0_err']:.2f} dir {r['dir_err']:.2f} |w| {r['wmag']:.4f} deg/s | "
              f"{len(c['members'])} cands | {twin}", flush=True)

    bandA = [c for c in clusters if c["rep"]["surr_rho"] < 1.0]
    bandAB = [c for c in clusters if c["rep"]["surr_rho"] < 4.0]
    truth_near = [c for c in clusters if c["rep"]["q0_err"] < 15 and c["rep"]["dir_err"] < 15]
    print(f"\nMULTI-SOL SET: {len(clusters)} distinct attractors | "
          f"surr_rho<1: {len(bandA)} | surr_rho<4: {len(bandAB)}", flush=True)
    print(f"DIAGNOSTIC: truth's basin {'PRESENT' if truth_near else 'ABSENT'} in the set"
          + (f" (q0_err {truth_near[0]['rep']['q0_err']:.2f}, dir {truth_near[0]['rep']['dir_err']:.2f}, "
             f"surr_rho {truth_near[0]['rep']['surr_rho']:.3f})" if truth_near else ""), flush=True)

    with open(OUT / f"polish_{SEED}.json", "w") as f:
        json.dump(dict(seed=SEED, floor=floor, truth_wmag_dps=wmag_t,
                       n_attractors=len(clusters), n_rho_lt1=len(bandA), n_rho_lt4=len(bandAB),
                       polished=polished_sorted,
                       clusters=[dict(rep=c["rep"], members=c["members"]) for c in clusters],
                       wall_s=time.time() - t0), f, indent=2, default=float)
    print(f"\nSaved: {OUT/'polish_116.json'}\nWALL: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
