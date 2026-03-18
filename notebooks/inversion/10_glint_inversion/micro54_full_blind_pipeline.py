#!/usr/bin/env python3
"""Micro-54 -- Full blind inversion pipeline (NO ORACLE).

Two-stage architecture:
  Stage 1 (omega candidates): Bridge 360×360 pairs from two glint circles,
           rank by |omega| closeness to peak-count estimate, score top 5000
           by lo-fi LC residual, keep top 5 omega candidates.
  Stage 2 (attitude recovery): For each of 5 omega candidates, run the
           phi sweep (10 hyp × 36 phi) + Nelder-Mead refinement → 5 refined
           (q0, omega) solutions.
  Stage 3 (selection): Hi-fi LC residual on 5 solutions → pick best.

Total: ~20 min per trajectory.  5 trajectories.
"""

import sys, os, time, json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from scipy.optimize import minimize
from numpy.polynomial import polynomial as P

from lib.experiment_setup import setup_experiment, attitude_error_deg
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "micro46_trajectories"

N_PHI_BRIDGE = 36; N_PHI_SWEEP = 36; MAX_BRIDGE = 5000; N_OMEGA_CANDS = 5
PEAK_COEFFS = np.array([0.0417, 0.0397])

def aq(phi, n, pab):
    R0, _ = Rotation.align_vectors([n], [pab])
    q = (Rotation.from_rotvec(phi * n) * R0).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

def ps(q, omega, ta, tt, I):
    dt = tt - ta; tq = np.zeros((len(tt), 4)); tq[np.abs(dt)<=1e-6] = q
    f = dt > 1e-6
    if f.any():
        qf, _ = propagate_attitude(q, omega, np.concatenate([[0.], dt[f]]), "tumbling", I)
        tq[f] = qf[1:]
    b = dt < -1e-6
    if b.any():
        bt = -dt[b][::-1]
        qb, _ = propagate_attitude(q, -omega, np.concatenate([[0.], bt]), "tumbling", I)
        tq[b] = qb[1:][::-1]
    return tq

def mnc(gq, gp, ns):
    t = 0.0
    for i in range(len(gq)):
        q = gq[i]; R = Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
        t += (1.0 - max(np.dot(R.T @ ns[j], gp[i]) for j in range(len(ns))))**2
    return t

def ode(w1, w2):
    d = np.dot(w1,w2); n = np.linalg.norm(w1)*np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(d/(n+1e-30),-1,1)))) if n>1e-15 else 180.

def aab(q1, q2, dt):
    R1 = Rotation.from_quat([q1[1],q1[2],q1[3],q1[0]])
    R2 = Rotation.from_quat([q2[1],q2[2],q2[3],q2[0]])
    return (R2*R1.inv()).as_rotvec()/dt

print("="*70)
print("micro54 -- FULL BLIND INVERSION PIPELINE (no oracle)")
print("="*70)
t_global = time.time()

CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5,-0.3,2.0), end_time_utc='2020-02-05T11:00:00')

m = np.load(str(DATA_DIR/"micro46_trajectories.npz"), allow_pickle=True)
ot = m['observation_times']; pab = m['pab_j2000']; ns = m['unique_normals']
IT = m['inertia_tensor']; q0s = m['q0s']; o0s = m['omega0s']
oma = m['omega_mags']; mh = m['mag_hifi']; ff = m['group_frac_flux']
nn = len(ns)

# 5 trajectories: include traj 84 (known to have close omega in bridge pool)
TEST = [84, 70, 35, 22, 19]
print(f"Test: {TEST}, omega: {[f'{oma[t]:.3f}' for t in TEST]}")

pv = np.linspace(0, 2*np.pi, N_PHI_BRIDGE, endpoint=False)
all_results = []

for traj_idx in TEST:
    t0 = time.time()
    mags = mh[traj_idx]; omt = o0s[traj_idx]; omm = float(oma[traj_idx])
    _, oh = propagate_attitude(q0s[traj_idx], omt, ot, "tumbling", IT)

    pks, _ = find_peaks(-mags, distance=5, prominence=0.3)
    br = pks[mags[pks]<9.0]; npk = len(pks)
    if len(br)<3:
        all_results.append({'traj_idx':int(traj_idx),'error':'too few'})
        print(f"\n  Traj {traj_idx}: SKIP"); continue

    labs = [int(np.argmax(ff[traj_idx,:,p])) for p in br]
    cfs = [float(ff[traj_idx,labs[i],br[i]]) for i in range(len(br))]
    cp = br[[i for i,c in enumerate(cfs) if c>0.77]]
    if len(cp)<3:
        all_results.append({'traj_idx':int(traj_idx),'error':'too few conf'})
        print(f"\n  Traj {traj_idx}: SKIP"); continue

    sm = cp[np.argsort(mags[cp])]; a1,a2 = int(sm[0]),int(sm[1])
    dtab = ot[a2]-ot[a1]
    sc = cp[(cp!=a1)&(cp!=a2)]; sp = pab[sc]; st = ot[sc]
    oe = float(P.polyval(npk, PEAK_COEFFS)); oer = np.deg2rad(oe)
    nw = int(oe*abs(dtab)/360)+2

    print(f"\n  Traj {traj_idx} (|ω|={omm:.3f}, est={oe:.3f}), "
          f"a1={a1} a2={a2} dt={dtab:.0f}s, scoring={len(sc)}, nw={nw}")

    # ===== STAGE 1: Bridge → LC score → top 5 omega candidates =====
    t1 = time.time()
    c1 = [(h,p,aq(p,ns[h],pab[a1])) for h in range(nn) for p in pv]
    c2 = [(h,p,aq(p,ns[h],pab[a2])) for h in range(nn) for p in pv]

    bridges = []
    for h1,p1,q1 in c1:
        for h2,p2,q2 in c2:
            rv = aab(q1,q2,dtab); mg0 = np.rad2deg(np.linalg.norm(rv))
            for w in range(nw+1):
                if w==0: ob=rv
                else:
                    if np.linalg.norm(rv)<1e-15: continue
                    ob = rv + (rv/np.linalg.norm(rv))*(2*np.pi*w/dtab)
                md = abs(np.rad2deg(np.linalg.norm(ob))-oe)/oe
                bridges.append((md,q1,ob,w,h1))
    bridges.sort(key=lambda x:x[0])
    survivors = bridges[:MAX_BRIDGE]
    dt1 = time.time()-t1
    print(f"    S1a: {len(survivors)} bridges, {dt1:.0f}s")

    # LC score all survivors
    obj_lo = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=ot, observed_lightcurve=mags,
        sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
        satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
        compute_shadows_flag=False, articulation_matrices=CTX.art_matrices,
        mode="tumbling", inertia_tensor=IT, show_progress=False)

    t1b = time.time()
    lc_scored = []
    for si,(md,q1,ob,w,h1) in enumerate(survivors):
        try:
            bt = np.array([0.,ot[a1]])
            qb,oob = propagate_attitude(q1,-ob,bt,"tumbling",IT)
            q0c=qb[-1]; o0c=-oob[-1]
            rv = Rotation.from_quat([q0c[1],q0c[2],q0c[3],q0c[0]]).as_rotvec()
            lc = float(obj_lo.evaluate(np.concatenate([rv,o0c])))
            lc_scored.append((lc,q0c,o0c,ob,w,h1))
        except: lc_scored.append((1e10,None,None,ob,w,h1))
        if (si+1)%1000==0: print(f"      [{si+1}/{len(survivors)}] "
                                  f"{time.time()-t1b:.0f}s", flush=True)

    lc_scored.sort(key=lambda x:x[0])
    omega_cands = lc_scored[:N_OMEGA_CANDS]
    dt1b = time.time()-t1b
    print(f"    S1b: LC scored, {dt1b:.0f}s")
    print(f"    Top {N_OMEGA_CANDS} omega candidates:")
    for i,(lc,q0c,o0c,ob,w,h1) in enumerate(omega_cands):
        if o0c is not None:
            de = ode(o0c, omt); qe = attitude_error_deg(q0c,q0s[traj_idx])
            print(f"      #{i+1}: LC={lc:.4f}, ωdir={de:.1f}°, q0={qe:.1f}°")

    # ===== STAGE 2: Phi sweep + NM for each omega candidate =====
    print(f"    STAGE 2: phi sweep + NM for {N_OMEGA_CANDS} candidates...")
    t2 = time.time()
    refined = []

    for ci,(lc0,_,_,omega_cand,w,_) in enumerate(omega_cands):
        if omega_cand is None: continue
        # Pick the anchor (brightest peak)
        anch = a1; anch_time = ot[anch]
        non_anch = sc; g_pabs = sp; g_times = st

        # Phi sweep
        best_cost = np.inf; best_hyp = -1; best_phi = 0.
        for hi in range(nn):
            for phi in pv:
                qa = aq(phi, ns[hi], pab[anch])
                try:
                    gq = ps(qa, omega_cand, anch_time, g_times, IT)
                    c = mnc(gq, g_pabs, ns)
                except: c = 1e10
                if c < best_cost: best_cost = c; best_hyp = hi; best_phi = phi

        # Nelder-Mead refinement (4D: phi + omega)
        def nm_cost(params):
            phi = params[0]; omega = params[1:4]
            qa = aq(phi, ns[best_hyp], pab[anch])
            try:
                gq = ps(qa, omega, anch_time, g_times, IT)
                return mnc(gq, g_pabs, ns)
            except: return 1e10

        x0 = np.array([best_phi, *omega_cand])
        try:
            res = minimize(nm_cost, x0, method='Nelder-Mead',
                           options={'maxfev':400, 'xatol':1e-8, 'fatol':1e-12,
                                    'adaptive':True})
            phi_r = res.x[0]; omega_r = res.x[1:4]
        except: phi_r = best_phi; omega_r = omega_cand

        # Propagate to t=0
        qa_r = aq(phi_r, ns[best_hyp], pab[anch])
        try:
            bt = np.array([0., anch_time])
            qb,ob = propagate_attitude(qa_r, -omega_r, bt, "tumbling", IT)
            q0r = qb[-1]; o0r = -ob[-1]
            refined.append((q0r, o0r, omega_r, best_hyp, phi_r, ci))
        except: pass

    dt2 = time.time()-t2
    print(f"    S2: {len(refined)} refined candidates, {dt2:.0f}s")

    # ===== STAGE 3: Hi-fi LC ranking =====
    t3 = time.time()
    obj_hi = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=ot, observed_lightcurve=mags,
        sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
        satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
        compute_shadows_flag=True, articulation_matrices=CTX.art_matrices,
        mode="tumbling", inertia_tensor=IT, show_progress=False)

    best_hifi = 1e10; best_solution = None
    for q0r, o0r, omega_r, hyp, phi, ci in refined:
        try:
            rv = Rotation.from_quat([q0r[1],q0r[2],q0r[3],q0r[0]]).as_rotvec()
            hifi = float(obj_hi.evaluate(np.concatenate([rv, o0r])))
            if hifi < best_hifi:
                best_hifi = hifi
                best_solution = (q0r, o0r, omega_r, hyp, hifi, ci)
        except: pass

    dt3 = time.time()-t3

    if best_solution is None:
        all_results.append({'traj_idx':int(traj_idx),'error':'no solution'})
        print(f"    S3: no solution"); continue

    q0f,o0f,omf,hyp,hifi,ci = best_solution
    q0_err = attitude_error_deg(q0f, q0s[traj_idx])
    od_err = ode(o0f, omt)
    om_err = abs(np.rad2deg(np.linalg.norm(o0f))-omm)/omm*100
    conv = q0_err<5 and od_err<5
    anti = q0_err>170 and od_err<10
    s = "CONVERGED" if conv else ("~180°" if anti else "FAILED")

    dt_total = time.time()-t0
    print(f"    S3: hi-fi={hifi:.4f}, {dt3:.0f}s")
    print(f"    RESULT: q0={q0_err:.1f}°, ωdir={od_err:.1f}°, ωmag={om_err:.0f}% "
          f"[{s}] ({dt_total:.0f}s)")

    all_results.append({
        'traj_idx': int(traj_idx), 'omega_dps': omm,
        'q0_err': float(q0_err), 'omega0_dir_err': float(od_err),
        'omega0_mag_err': float(om_err), 'hifi_residual': float(hifi),
        'converged': bool(conv), 'antiparallel': bool(anti),
        'from_candidate': ci, 'runtime_s': float(dt_total),
    })

# Summary
print("\n"+"="*70)
print("SUMMARY — FULL BLIND INVERSION")
print("="*70)
v = [r for r in all_results if 'error' not in r]
nc = sum(1 for r in v if r['converged'])
na = sum(1 for r in v if r.get('antiparallel'))
print(f"Converged: {nc}/{len(v)}")
print(f"Antiparallel: {na}/{len(v)}")
print(f"Total success: {nc+na}/{len(v)}")
for r in v:
    s = "OK" if r['converged'] else ("~180" if r.get('antiparallel') else "FAIL")
    print(f"  Traj {r['traj_idx']} (ω={r['omega_dps']:.3f}): "
          f"q0={r['q0_err']:.1f}°, ωdir={r['omega0_dir_err']:.1f}°, "
          f"cand#{r['from_candidate']}, {r['runtime_s']:.0f}s [{s}]")

with open(str(RESULTS_DIR/"micro54_full_blind.json"),'w') as f:
    json.dump({'experiment':'micro54','results':all_results,
               'total_time':time.time()-t_global}, f, indent=2,
              default=lambda x:float(x) if isinstance(x,np.floating)
              else int(x) if isinstance(x,np.integer) else x)

print(f"\nTotal: {time.time()-t_global:.0f}s")
