#!/usr/bin/env python3
"""Unit test for loop/artifacts.reify — the Slice 1.5 dataclass + tuple-position dispatch.

Proves the serialisation MECHANISM without running an expensive real inversion (analytical
before computational): construct toy result-dataclass instances + a multi-return tuple, reify
them against the REAL substrate Tool cards, and assert each lands as a card + blob with a compact
sample (never the arrays) and round-trips through load(). Writes to a TEMP dir so the store is
never polluted.

    python research_os/loop/test_artifacts.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_os/
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "loop"))
sys.path.insert(0, REPO)

import artifacts  # noqa: E402

from src.computation.inertia_calculator import InertiaResult  # noqa: E402
from src.inversion.optimizers import OptimizationResult  # noqa: E402
from src.inversion.results import InversionResult  # noqa: E402


def _card(tool_id: str) -> dict:
    return json.load(open(os.path.join(ROOT, "substrate", f"{tool_id}.json")))


def _check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise AssertionError(name)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="ai_test_")
    artifacts.INSTANCES = tmp
    artifacts.DATA = os.path.join(tmp, "data")

    inertia = InertiaResult(total_mass=12.5, center_of_mass=np.zeros(3),
                            inertia_tensor=np.eye(3), principal_moments=np.array([1., 2., 3.]),
                            principal_axes=np.eye(3))
    opt = OptimizationResult(params=np.zeros(6), cost=0.42, n_evaluations=137,
                             success=True, message="converged")
    opt_list = [opt, OptimizationResult(params=np.ones(6), cost=0.99, n_evaluations=50,
                                        success=False, message="maxiter")]
    inv = InversionResult(q0=np.array([1., 0, 0, 0]), omega0=np.array([0.01, 0, 0.02]),
                          chi_squared=3.1, rms_residual=0.08,
                          predicted_lightcurve=np.zeros(50), observed_lightcurve=np.zeros(50),
                          observation_times=np.arange(50.0),
                          uncertainties={"std_devs": np.ones(6)})
    # propagate_to_body_frame returns (k1_body, k2_body, quats); material is quats @ position 2
    prop_tuple = (np.zeros((50, 3)), np.zeros((50, 3)), np.zeros((50, 4)))

    cases = [
        ("compute_inertia", inertia, "compose", "inertia-result", "return", "InertiaResult", None),
        ("global_optimize", opt, "compose", "candidate-set", "return", "OptimizationResult", None),
        ("multi_start_optimize", opt_list, "compose", "candidate-set", "return", None, 2),
        ("invert_lightcurve", inv, "compose", "inversion-result", "return", "InversionResult", None),
        ("propagate_to_body_frame", prop_tuple, "compose", "attitude-trajectory", "quats", None, None),
    ]

    seq = 0
    for tool_id, rv, op, exp_type, exp_port, exp_dc, exp_len in cases:
        print(f"\n[{tool_id}]")
        card = _card(tool_id)
        cards, seq = artifacts.reify(rv, card, {"run": f"tr_test_{tool_id}", "step": None},
                                     op=op, seq_start=seq, now="2026-06-08T00:00:00", commit="test")
        _check("exactly one card written", len(cards) == 1)
        c = cards[0]
        _check(f"artifact_type == {exp_type}", c["artifact_type"] == exp_type)
        _check(f"port == {exp_port} (not 'quats@2')", c["port"] == exp_port)
        _check("blob exists on disk", os.path.isfile(os.path.join(REPO, c["path"])))
        _check("card file exists", os.path.isfile(os.path.join(tmp, f"{c['id']}.json")))
        # sample is a DIGEST, never the arrays: no value is a long list OF NUMBERS
        # (a 'fields' name-list is fine; an array dump is not)
        big = [k for k, v in c["sample"].items()
               if isinstance(v, list) and len(v) > 8 and all(isinstance(x, (int, float)) for x in v)]
        _check("sample holds no array blob", not big)
        if exp_dc:
            _check(f"sample names dataclass {exp_dc}", c["sample"].get("dataclass") == exp_dc)
        if exp_len is not None:
            _check(f"sample len == {exp_len}", c["sample"].get("len") == exp_len)
        # round-trip through the Slice-2 load() seam
        loaded = artifacts.load(c["id"])
        if exp_dc == "InertiaResult":
            _check("load round-trips total_mass", abs(loaded["total_mass"] - 12.5) < 1e-9)
        elif exp_port == "quats":
            arr = loaded["value"] if isinstance(loaded, dict) and "value" in loaded else loaded
            _check("load round-trips quats shape (50,4)", np.asarray(arr).shape == (50, 4))
        elif exp_len == 2:
            _check("load round-trips list len 2", isinstance(loaded, list) and len(loaded) == 2)

    # --- Slice 1.5b: file-reference materials (a tool-written kernel FILE, referenced not copied) ---
    from pathlib import Path
    kdir = os.path.join(tmp, "kernels")
    os.makedirs(kdir, exist_ok=True)
    spk = Path(kdir) / "intelsat_901.bsp"
    spk.write_bytes(b"DAF/SPK fake")  # a real file on disk, OUTSIDE research_os/data
    ck = Path(kdir) / "orient.bc"
    sclk = Path(kdir) / "orient.tsc"
    ck.write_bytes(b"DAF/CK fake")
    sclk.write_bytes(b"SCLK fake")

    print("\n[spice_kernel_generate]  (single Path -> reference)")
    card = _card("spice_kernel_generate")
    cards, seq = artifacts.reify(spk, card, {"run": "tr_test_spk", "step": None},
                                 op="compose", seq_start=seq, now="2026-06-08T00:00:00", commit="test")
    _check("exactly one card", len(cards) == 1)
    c = cards[0]
    _check("storage == reference", c.get("storage") == "reference")
    _check("artifact_type == spk-kernel", c["artifact_type"] == "spk-kernel")
    _check("port == spk_path", c["port"] == "spk_path")
    _check("path references the tool-written file (absolute, outside repo)", os.path.abspath(c["path"]) == str(spk))
    _check("sample records exists+ext", c["sample"].get("exists") is True and c["sample"].get("ext") == ".bsp")
    _check("NO blob copy under data/", not os.path.isfile(os.path.join(tmp, "data", f"{c['id']}.npz"))
           and not os.path.isfile(os.path.join(tmp, "data", f"{c['id']}.json")))
    _check("load() returns the file PATH (not bytes)", artifacts.load(c["id"]) == str(spk))

    print("\n[spice_orientation_kernel]  ((ck, sclk) tuple -> two references)")
    card = _card("spice_orientation_kernel")
    cards, seq = artifacts.reify((ck, sclk), card, {"run": "tr_test_orient", "step": None},
                                 op="compose", seq_start=seq, now="2026-06-08T00:00:00", commit="test")
    _check("two cards (ck + sclk)", len(cards) == 2)
    by_port = {c["port"]: c for c in cards}
    _check("ck_path -> ck-kernel reference", by_port["ck_path"]["artifact_type"] == "ck-kernel"
           and by_port["ck_path"].get("storage") == "reference")
    _check("sclk_path -> sclk-kernel reference", by_port["sclk_path"]["artifact_type"] == "sclk-kernel"
           and by_port["sclk_path"].get("storage") == "reference")
    _check("ck load() returns the .bc path", artifacts.load(by_port["ck_path"]["id"]) == str(ck))

    print(f"\nALL artifacts.reify cases PASS — dataclass/tuple-position + file-reference  (tmp: {tmp})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
