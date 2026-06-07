"""s113 — slow-tumbler generality (contract_slow-tumbler-generality, v0).

Aggregates the fixed-config slow-sweep (run_slow_sweep.sh: s100 fixed-2deg + s110
surrogate-v2 polish) over seeds 116/10/42/31 into the contract's required artefacts:

  1. surr-rho band table over all 4 seeds' polished attractors (CSV + printed).
  2. per-seed decimation-fix verification (the s100 '[3] dense'/'[4] decimate' log
     lines, showing FIXED 2deg cells and nearest-truth nt NOT widened to 12.5deg).
  3. multi-solution attractor plot per seed (LC overlay + surr-rho bar; s112 style).
  4. summary.json carrying the confirm/refute criteria booleans.

Confirm (contract): each of {42, 31} has >=1 attractor at surr_rho < 4 (Band A∪B-equiv);
116 regression reproduces >=4 at-floor (surr_rho<1) attractors with truth-near present.
Refute: >=1 of {42,31} has 0 attractors surr_rho<4, OR 116 regression degrades.
Surrogate-only: surr_rho<4 attractors are provisional (no hi-fi rho-band), per contract.
All |omega| in deg/s. Truth errors are DIAGNOSTIC labels (acceptance is multi-solution).
"""
import os, sys, json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"

HERE = Path(__file__).resolve().parent
SURVEY = HERE.parent
sys.path.insert(0, str(SURVEY)); sys.path.insert(0, str(HERE))

from lib.hifi_render import build_context
from lib.forward import propagate_to_body_frame
from lib.surrogate_eval import predict as surrogate_predict

SEEDS = [116, 10, 42, 31]
S100 = SURVEY / "results" / "s100"
S110 = SURVEY / "results" / "s110"
OUT = SURVEY / "results" / "s113"; OUT.mkdir(parents=True, exist_ok=True)
BAND_C = {"A": "#1a9850", "B": "#fdae61", "C+": "#d73027"}


def band(rho):
    return "A" if rho < 1 else ("B" if rho < 4 else "C+")


def polish_path(seed):
    for p in (S110 / f"polish_{seed:03d}.json", S110 / f"polish_{seed}.json"):
        if p.exists():
            return p
    return None


def decim_lines(seed):
    """Pull the s100 '[3] dense fill' + '[4] decimate' lines for the fix-verification."""
    log = S100 / f"seed{seed}_capped_run.log"
    if not log.exists():
        return {"dense": None, "decimate": None}
    txt = log.read_text(errors="replace").splitlines()
    g = lambda tag: next((ln.strip() for ln in txt if ln.lstrip().startswith(tag)), None)
    return {"dense": g("[3]"), "decimate": g("[4]")}


def surr_lc(q0, w0, ctx):
    k1, k2, _ = propagate_to_body_frame(
        q0_wxyz=np.asarray(q0, float), omega0_rad=np.asarray(w0, float),
        observation_times=ctx["observation_times"], sun_pos=ctx["sun_pos"],
        obs_pos=ctx["obs_pos"], sat_pos=ctx["sat_pos"],
        inertia_tensor=ctx["inertia_tensor"], mode="tumbling")
    return surrogate_predict(k1, k2, ctx["obs_dist"])


def plot_seed(seed, res, ctx):
    attr = [c["rep"] for c in res["clusters"]]
    floor = res["floor"]; wtruth = res["truth_wmag_dps"]
    n_a = sum(1 for a in attr if a["surr_rho"] < 1)
    n_b = sum(1 for a in attr if 1 <= a["surr_rho"] < 4)
    t_obs = ctx["observation_times"] - ctx["observation_times"][0]
    m_obs = ctx["mag_hifi_truth"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.5),
                                   gridspec_kw={"width_ratios": [2.3, 1]})
    axL.plot(t_obs, m_obs, color="k", lw=2.5, label="observed (truth)", zorder=10)
    for i, a in enumerate(attr):
        b = band(a["surr_rho"])
        if b == "C+":
            continue
        y = surr_lc(a["q0_pol"], a["w0_pol"], ctx)
        tag = "truth" if a["q0_err"] < 5 else ("body-twin" if a["q0_err"] > 170 else "alt")
        lab = (f"A{i+1} [{b}] — {tag}: q0 {a['q0_err']:.1f}°, ω-dir {a['dir_err']:.1f}°, "
               f"|ω| {a['wmag']:.4f} deg/s, ρ {a['surr_rho']:.2f}")
        axL.plot(t_obs, y, "-" if b == "A" else "--", color=BAND_C[b], lw=1.5,
                 alpha=0.85, label=lab)
    axL.invert_yaxis()
    axL.set_xlabel("time from first epoch (s)"); axL.set_ylabel("apparent magnitude")
    axL.set_title(f"Seed {seed} — slow tumbler (|ω|_truth = {wtruth:.4f} deg/s)\n"
                  f"{n_a} Band-A + {n_b} Band-B attractors; {n_a + n_b} multi-solutions shown")
    axL.legend(fontsize=7.5, loc="best", framealpha=0.93); axL.grid(alpha=0.22)
    rhos = [a["surr_rho"] for a in attr]; yy = np.arange(len(attr))
    axR.barh(yy, rhos, color=[BAND_C[band(r)] for r in rhos], edgecolor="white", linewidth=0.5)
    axR.axvline(1.0, color="#1a9850", ls=":", lw=1.5, label="Band A/B (ρ=1)")
    axR.axvline(4.0, color="#fdae61", ls=":", lw=1.5, label="Band B/C (ρ=4)")
    for j, a in enumerate(attr):
        tag = "truth" if a["q0_err"] < 5 else ("twin" if a["q0_err"] > 165 else "alt")
        axR.text(min(rhos[j] + 0.08, max(rhos) * 1.02), j,
                 f"q0 {a['q0_err']:.1f}° / ω-dir {a['dir_err']:.1f}° [{tag}]", va="center", fontsize=7.5)
    axR.set_yticks(yy); axR.set_yticklabels([f"A{j+1}" for j in yy])
    axR.set_xlabel(f"surrogate ρ  (noise floor = {floor:.3f} mag)")
    axR.set_title("attractor quality summary")
    axR.legend(fontsize=7.5, loc="lower right"); axR.grid(alpha=0.22, axis="x")
    axR.set_xlim(0, max(rhos) * 1.5)
    fig.tight_layout(pad=1.2)
    out = OUT / f"slow_tumbler_seed{seed:03d}.png"
    fig.savefig(out, dpi=145, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: results/s113/slow_tumbler_seed{seed:03d}.png")


def main():
    per_seed = {}
    table_rows = []   # flat per-attractor rows for the CSV/printed band table
    for seed in SEEDS:
        pp = polish_path(seed)
        if pp is None:
            per_seed[seed] = {"present": False}
            print(f"seed {seed}: NO polish json (sweep produced no result)")
            continue
        res = json.load(open(pp))
        attr = [c["rep"] for c in res["clusters"]]
        truth_near = [a for a in attr if a["q0_err"] < 15 and a["dir_err"] < 15]
        per_seed[seed] = dict(
            present=True, floor=res["floor"], truth_wmag_dps=res["truth_wmag_dps"],
            n_attractors=res["n_attractors"], n_rho_lt1=res["n_rho_lt1"],
            n_rho_lt4=res["n_rho_lt4"],
            best_rho=min((a["surr_rho"] for a in attr), default=None),
            truth_near=(dict(q0_err=truth_near[0]["q0_err"], dir_err=truth_near[0]["dir_err"],
                             wmag=truth_near[0]["wmag"], surr_rho=truth_near[0]["surr_rho"])
                        if truth_near else None),
            decim=decim_lines(seed))
        for j, a in enumerate(sorted(attr, key=lambda r: r["surr_rho"])):
            tag = ("truth" if a["q0_err"] < 5 else "body-twin" if a["q0_err"] > 170
                   else "alt-basin")
            table_rows.append(dict(seed=seed, rank=j + 1, band=band(a["surr_rho"]),
                                   surr_rho=round(a["surr_rho"], 4),
                                   q0_err=round(a["q0_err"], 2), dir_err=round(a["dir_err"], 2),
                                   wmag_dps=round(a["wmag"], 4), tag=tag))
        ctx = build_context(seed)
        plot_seed(seed, res, ctx)

    # ---- contract scoring ----
    def ok4(s):  # >=1 Band A∪B (surr_rho<4)
        return per_seed[s]["present"] and per_seed[s]["n_rho_lt4"] >= 1
    s116 = per_seed.get(116, {})
    regression_ok = bool(s116.get("present") and s116.get("n_rho_lt1", 0) >= 4
                         and s116.get("truth_near") is not None
                         and s116["truth_near"]["q0_err"] < 1.0)
    confirm = ok4(42) and ok4(31) and regression_ok
    refute = ((per_seed.get(42, {}).get("present") and per_seed[42]["n_rho_lt4"] == 0)
              or (per_seed.get(31, {}).get("present") and per_seed[31]["n_rho_lt4"] == 0)
              or (s116.get("present") and not regression_ok))

    # ---- printed band table ----
    print("\n========== SURR-ρ BAND TABLE (all seeds, contract artefact #1) ==========")
    print(f"{'seed':>4} {'rank':>4} {'band':>4} {'surr_rho':>9} {'q0_err':>7} "
          f"{'dir_err':>7} {'|w|dps':>7}  tag")
    for r in table_rows:
        print(f"{r['seed']:>4} {r['rank']:>4} {r['band']:>4} {r['surr_rho']:>9.4f} "
              f"{r['q0_err']:>7.2f} {r['dir_err']:>7.2f} {r['wmag_dps']:>7.4f}  {r['tag']}")

    print("\n========== DECIMATION-FIX VERIFICATION (artefact #2) ==========")
    for seed in SEEDS:
        d = per_seed.get(seed, {}).get("decim", {"dense": None, "decimate": None})
        print(f"  seed {seed}:")
        print(f"    {d.get('dense')}")
        print(f"    {d.get('decimate')}")

    print("\n========== CONTRACT SCORING ==========")
    for s in (42, 31):
        ps = per_seed.get(s, {})
        print(f"  seed {s}: present={ps.get('present')} n_rho<4={ps.get('n_rho_lt4')} "
              f"-> Band A∪B {'YES' if ok4(s) else 'NO'}")
    print(f"  seed 116 regression: n_rho<1={s116.get('n_rho_lt1')} truth_near={s116.get('truth_near')}"
          f" -> {'OK' if regression_ok else 'DEGRADED'}")
    print(f"  CONFIRM={confirm}  REFUTE={refute}")

    # ---- save artefacts ----
    import csv
    with open(OUT / "band_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "rank", "band", "surr_rho",
                                          "q0_err", "dir_err", "wmag_dps", "tag"])
        w.writeheader(); w.writerows(table_rows)
    summary = dict(contract="contract_slow-tumbler-generality", version="v0",
                   seeds=SEEDS, per_seed=per_seed, band_table=table_rows,
                   regression_ok=regression_ok, confirm=confirm, refute=refute,
                   surrogate_only_caveat="surr_rho<4 attractors provisional; no hi-fi rho-band")
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved: results/s113/band_table.csv")
    print(f"Saved: results/s113/summary.json")


if __name__ == "__main__":
    main()
