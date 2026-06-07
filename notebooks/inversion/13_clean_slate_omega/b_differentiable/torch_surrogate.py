"""Torch v2 surrogate — differentiable port of ~/surrogate_model/surrogate_model/surrogate.py.

Exactly replicates SurrogateModel.predict_magnitude() with end-to-end
autodiff through:
  1. Analytical no-shadow BRDF (vectorized Ashikhmin-Shirley).
  2. 32D feature construction.
  3. Residual MLP ensemble (3 members, 6-layer ReLU 32->256x4->128->1).
  4. log10(phi_noshadow) + avg(delta) -> magnitude.

Uses the same weight NPZs as the numpy SurrogateModel
(~/surrogate_model/surrogate_model/s12_residual_5M_*.npz + s11_geometry.npz).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn


SURROGATE_DIR = Path.home() / "surrogate_model" / "surrogate_model"


class NoShadowEvaluatorTorch(nn.Module):
    """Vectorized Ashikhmin-Shirley BRDF, torch autodiff end-to-end.

    Mirrors NoShadowEvaluator from the numpy surrogate: 4 component groups
    (bus, panels, dish_east, dish_west), inverse-rotate k1/k2/h by the panel
    or dish Z-rotation instead of rotating facet normals. Returns phi_noshadow
    (unnormalized flux), same value the numpy path computes.
    """

    def __init__(self, geometry_path: Path, dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        g = np.load(geometry_path)
        normals = g["normals"]
        areas = g["areas"]
        r_d = g["r_d"]
        r_s = g["r_s"]
        n_phong = g["n_phong"]
        bus_idx = g["bus_idx"]
        panel_idx = g["panel_idx"]
        dish_east_idx = g["dish_east_idx"]
        dish_west_idx = g["dish_west_idx"]

        # Group flag: which groups rotate (bus does not, panel by panel_deg,
        # dish_east by dish_deg, dish_west by -dish_deg).
        group_specs = [
            ("bus",        bus_idx,       None,   None),
            ("panel",      panel_idx,     "p",   +1.0),
            ("dish_east",  dish_east_idx, "d",   +1.0),
            ("dish_west",  dish_west_idx, "d",   -1.0),
        ]

        for name, idx, which, sign in group_specs:
            if len(idx) == 0:
                continue
            self.register_buffer(f"{name}_normals",
                                 torch.as_tensor(normals[idx], dtype=dtype, device=device))
            self.register_buffer(f"{name}_areas",
                                 torch.as_tensor(areas[idx], dtype=dtype, device=device))
            self.register_buffer(f"{name}_r_d",
                                 torch.as_tensor(r_d[idx], dtype=dtype, device=device))
            self.register_buffer(f"{name}_r_s",
                                 torch.as_tensor(r_s[idx], dtype=dtype, device=device))
            self.register_buffer(f"{name}_n_phong",
                                 torch.as_tensor(n_phong[idx], dtype=dtype, device=device))

        # Save group metadata (python side, not tensors)
        self._groups: List[dict] = []
        for name, idx, which, sign in group_specs:
            if len(idx) == 0:
                self._groups.append(None)
                continue
            self._groups.append({"name": name, "which": which, "sign": sign})

    @staticmethod
    def _inv_rot(k1: torch.Tensor, k2: torch.Tensor, h: torch.Tensor,
                 cos_t: torch.Tensor, sin_t: torch.Tensor):
        """Apply R_z(-t) to vectors. cos_t/sin_t shape (N,1)."""
        def rot(v: torch.Tensor) -> torch.Tensor:
            vx = v[:, 0:1]
            vy = v[:, 1:2]
            vz = v[:, 2:3]
            nx = cos_t * vx + sin_t * vy
            ny = -sin_t * vx + cos_t * vy
            return torch.cat([nx, ny, vz], dim=1)
        return rot(k1), rot(k2), rot(h)

    def _facet_flux(self, k1r: torch.Tensor, k2r: torch.Tensor, hr: torch.Tensor,
                    h_dot_k1: torch.Tensor, name: str) -> torch.Tensor:
        normals = getattr(self, f"{name}_normals")
        areas = getattr(self, f"{name}_areas")
        r_d = getattr(self, f"{name}_r_d")
        r_s = getattr(self, f"{name}_r_s")
        n_phong = getattr(self, f"{name}_n_phong")

        nk1 = k1r @ normals.T  # (N, F)
        nk2 = k2r @ normals.T
        nh = hr @ normals.T

        visible = (nk1 > 0) & (nk2 > 0)
        nk1_c = torch.clamp(nk1, min=0.0)
        nk2_c = torch.clamp(nk2, min=0.0)
        nh_c = torch.clamp(nh, min=0.0)

        alpha = 1.0 - nk1_c / 2.0
        beta = 1.0 - nk2_c / 2.0
        rho_d = (28.0 / (23.0 * np.pi)) * r_d * (1.0 - r_s) * \
                (1.0 - alpha ** 5) * (1.0 - beta ** 5)

        hk1 = h_dot_k1.unsqueeze(1)  # (N, 1)
        fresnel = r_s + (1.0 - r_s) * (1.0 - hk1) ** 5
        max_nd = torch.maximum(nk1_c, nk2_c)
        denom = hk1 * max_nd
        safe_denom = torch.where(denom > 1e-10, denom, torch.ones_like(denom))
        rho_s = torch.where(
            denom > 1e-10,
            ((n_phong + 1.0) / (8.0 * np.pi)) * (nh_c ** n_phong / safe_denom) * fresnel,
            torch.zeros_like(denom),
        )
        flux = (rho_d + rho_s) * areas * nk1_c * nk2_c * visible.to(k1r.dtype)
        return flux.sum(dim=1)

    def forward(self, k1: torch.Tensor, k2: torch.Tensor,
                panel_deg: torch.Tensor, dish_deg: torch.Tensor) -> torch.Tensor:
        h = k1 + k2
        h = h / torch.clamp(h.norm(dim=1, keepdim=True), min=1e-10)
        h_dot_k1 = (h * k1).sum(dim=1)

        deg2rad = torch.pi / 180.0
        cos_p = torch.cos(panel_deg * deg2rad).unsqueeze(1)
        sin_p = torch.sin(panel_deg * deg2rad).unsqueeze(1)
        cos_d = torch.cos(dish_deg * deg2rad).unsqueeze(1)
        sin_d = torch.sin(dish_deg * deg2rad).unsqueeze(1)

        phi = torch.zeros(k1.shape[0], dtype=k1.dtype, device=k1.device)
        for g in self._groups:
            if g is None:
                continue
            if g["which"] is None:
                k1r, k2r, hr = k1, k2, h
            elif g["which"] == "p":
                k1r, k2r, hr = self._inv_rot(k1, k2, h, cos_p, sin_p * g["sign"])
            elif g["which"] == "d":
                k1r, k2r, hr = self._inv_rot(k1, k2, h, cos_d, sin_d * g["sign"])
            else:
                raise RuntimeError(g["which"])
            phi = phi + self._facet_flux(k1r, k2r, hr, h_dot_k1, g["name"])
        return phi


class MLPMember(nn.Module):
    """Plain ReLU MLP with weights loaded from a single member NPZ."""

    def __init__(self, npz_path: Path, dtype: torch.dtype = torch.float32):
        super().__init__()
        w = np.load(npz_path)
        n_layers = int(w["n_layers"])
        self.n_layers = n_layers
        layers = []
        for i in range(n_layers):
            Wi = torch.as_tensor(w[f"W{i}"], dtype=dtype)
            bi = torch.as_tensor(w[f"b{i}"], dtype=dtype)
            out_f, in_f = Wi.shape
            lin = nn.Linear(in_f, out_f, bias=True)
            with torch.no_grad():
                lin.weight.copy_(Wi)
                lin.bias.copy_(bi)
            layers.append(lin)
        self.linears = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, lin in enumerate(self.linears):
            h = lin(h)
            if i < self.n_layers - 1:
                h = torch.relu(h)
        return h.squeeze(-1)


class TorchSurrogateV2(nn.Module):
    """Exact torch port of numpy SurrogateModel (v2 residual ensemble).

    Usage:
        model = TorchSurrogateV2.load_default()
        mag = model.predict_magnitude(k1, k2, panel_deg, dish_deg, dist_km)

    All inputs may be numpy or torch tensors. Output is a torch tensor with
    grad flowing back through k1, k2, panel_deg, dish_deg (dist_km is scalar
    wrt gradient).

    The default dtype is float64 for parity with the numpy path — torch CPU
    float32 matches to ~1e-4 which is not tight enough.
    """

    SUN_APPARENT_MAGNITUDE = -26.74

    def __init__(self, weights_paths: List[Path], normalization_path: Path,
                 geometry_path: Path, dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.noshadow = NoShadowEvaluatorTorch(geometry_path, dtype=dtype, device=device)

        self.members = nn.ModuleList([MLPMember(wp, dtype=dtype) for wp in weights_paths])

        n = np.load(normalization_path)
        self.register_buffer("X_mean", torch.as_tensor(n["X_mean"], dtype=dtype))
        self.register_buffer("X_std", torch.as_tensor(n["X_std"], dtype=dtype))
        self.y_mean = float(n["y_mean"])
        self.y_std = float(n["y_std"])

        self.to(device=device, dtype=dtype)

    @classmethod
    def load_default(cls, dtype: torch.dtype = torch.float64,
                     device: torch.device = torch.device("cpu")) -> "TorchSurrogateV2":
        pkg = SURROGATE_DIR
        seeds = [42, 123, 777]
        geom = pkg / "s11_geometry.npz"
        s12_weights = [pkg / f"s12_residual_5M_s{s}_weights.npz" for s in seeds]
        s12_norm = pkg / "s12_residual_5M_normalization.npz"
        if all(w.exists() for w in s12_weights) and s12_norm.exists():
            return cls(s12_weights, s12_norm, geom, dtype=dtype, device=device)
        weights = [pkg / f"s11_residual_s{s}_weights.npz" for s in seeds]
        norm = pkg / "s11_residual_normalization.npz"
        return cls(weights, norm, geom, dtype=dtype, device=device)

    @staticmethod
    def _make_features(k1: torch.Tensor, k2: torch.Tensor,
                       panel_deg: torch.Tensor, dish_deg: torch.Tensor) -> torch.Tensor:
        h = k1 + k2
        h = h / torch.clamp(h.norm(dim=1, keepdim=True), min=1e-10)
        k1k2 = (k1 * k2).sum(dim=1, keepdim=True)

        deg2rad = torch.pi / 180.0
        p = (panel_deg * deg2rad).unsqueeze(1)
        d = (dish_deg * deg2rad).unsqueeze(1)
        cp, sp = torch.cos(p), torch.sin(p)
        cd, sd = torch.cos(d), torch.sin(d)

        def rot_dots(ct, st, v):
            vx = v[:, 0:1]
            vy = v[:, 1:2]
            return ct * vx + st * vy, -st * vx + ct * vy

        pxk1, pyk1 = rot_dots(cp, sp, k1)
        pxk2, pyk2 = rot_dots(cp, sp, k2)
        pxh, pyh = rot_dots(cp, sp, h)

        dexk1, deyk1 = rot_dots(cd, sd, k1)
        dexk2, deyk2 = rot_dots(cd, sd, k2)
        dexh, deyh = rot_dots(cd, sd, h)

        dwxk1, dwyk1 = rot_dots(cd, -sd, k1)
        dwxk2, dwyk2 = rot_dots(cd, -sd, k2)
        dwxh, dwyh = rot_dots(cd, -sd, h)

        return torch.cat([
            k1, k2, h, k1k2,
            sp, cp, sd, cd,
            pxk1, pyk1, pxk2, pyk2, pxh, pyh,
            dexk1, deyk1, dexk2, deyk2, dexh, deyh,
            dwxk1, dwyk1, dwxk2, dwyk2, dwxh, dwyh,
        ], dim=1)

    def _to_t(self, x, shape=None):
        if isinstance(x, torch.Tensor):
            t = x.to(dtype=self.dtype, device=self.device)
        else:
            t = torch.as_tensor(np.asarray(x), dtype=self.dtype, device=self.device)
        if shape is not None:
            t = t.broadcast_to(shape).contiguous()
        return t

    def predict_log_phi(self, k1, k2, panel_deg, dish_deg) -> torch.Tensor:
        k1 = self._to_t(k1)
        k2 = self._to_t(k2)
        if k1.dim() == 1:
            k1 = k1.unsqueeze(0)
        if k2.dim() == 1:
            k2 = k2.unsqueeze(0)
        N = k1.shape[0]
        panel_deg = self._to_t(panel_deg).reshape(-1).broadcast_to((N,)).contiguous()
        dish_deg = self._to_t(dish_deg).reshape(-1).broadcast_to((N,)).contiguous()

        phi_ns = self.noshadow(k1, k2, panel_deg, dish_deg)
        log_phi_ns = torch.log10(torch.clamp(phi_ns, min=1e-15))

        X = self._make_features(k1, k2, panel_deg, dish_deg)
        X_n = (X - self.X_mean) / self.X_std

        delta_n = torch.zeros(N, dtype=self.dtype, device=self.device)
        for m in self.members:
            delta_n = delta_n + m(X_n)
        delta_n = delta_n / len(self.members)
        delta = delta_n * self.y_std + self.y_mean
        return log_phi_ns + delta

    def predict_magnitude(self, k1, k2, panel_deg, dish_deg, obs_dist_km) -> torch.Tensor:
        log_phi = self.predict_log_phi(k1, k2, panel_deg, dish_deg)
        obs_dist_km = self._to_t(obs_dist_km).reshape(-1).broadcast_to(log_phi.shape).contiguous()
        distance_m = obs_dist_km * 1000.0
        mag = (self.SUN_APPARENT_MAGNITUDE
               + 5.0 * torch.log10(distance_m)
               - 2.5 * log_phi)
        return mag


__all__ = ["TorchSurrogateV2", "NoShadowEvaluatorTorch"]
