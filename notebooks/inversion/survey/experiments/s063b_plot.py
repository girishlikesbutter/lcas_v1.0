"""Plot the s063b 1D cost slices."""
import json
import sys
from pathlib import Path

import numpy as np

SURVEY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(SURVEY / "results" / "s063b" / "summary.json"))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
colors = {"tangent": "tab:green", "normal_E": "tab:orange", "normal_L": "tab:red"}

for ax, seed in zip(axes, [89, 28, 14]):
    s = d[str(seed)]
    for name in ["tangent", "normal_E", "normal_L"]:
        pd = s["per_direction"][name]
        eps = np.array(pd["eps_frac"])
        costs = np.array(pd["costs"])
        rho = np.sqrt(costs) / 0.05
        ax.plot(eps * 100, rho, "o-", color=colors[name],
                label=f"{name} β_frac={s[f'beta_{name}_frac']:.0f}",
                ms=4, lw=1.5)
    ax.axhline(2, color="lightblue", ls=":", lw=1, label="Band A boundary")
    ax.axhline(4, color="khaki", ls=":", lw=1, label="Band B boundary")
    ax.set_xlabel("ε [% of |ω|]")
    if seed == 89:
        ax.set_ylabel("ρ = √MSE / 0.05")
    ax.set_title(f"seed {seed}  |ω|={s['omega_mag_dps']:.2f} dps  ρ_truth={s['rho_truth']:.2f}")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle("s063b — 1D cost slices: polhode tangent vs polhode-normal directions in ω at truth",
             fontsize=12)
fig.tight_layout()
out = SURVEY / "results" / "s063b" / "polhode_curvature_slices.png"
fig.savefig(str(out), dpi=130)
plt.close(fig)
print(f"Saved: {out}")
