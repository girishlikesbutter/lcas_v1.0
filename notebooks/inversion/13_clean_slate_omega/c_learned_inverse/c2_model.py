"""c2_model.py

Learned inverse regressor: LC (+ inertial geometry) -> distribution over omega.

Architecture (CPU-friendly, small):
    Input per timestep (8 ch):
        [mag(t),
         sun_unit(t) (3),
         obs_unit(t) (3),
         log10(obs_dist_km(t))]   # normalised via constants below
    Encoder: a 5-layer 1-D CNN with channel widths (8->32->64->128->128->128)
    Pooling: global mean over time -> 128-d vector.
    MDN head: K Gaussian mixture over an omega parameterisation:
        Parameterise omega as (log|omega|_dps, axis_unit_on_S2 in R^3).
        Predict K x (weight logit, 4-d mean, 4-d log_std) => per-mixture
        multivariate diagonal Gaussian in R^4. Axis part is sampled and
        re-normalised; this is a simple mixture-over-R^4 trick that avoids
        the exotic power-spherical/tangent-plane machinery but still gives a
        distribution on S^2 after projection.

Loss:
    MDN negative log-likelihood on the (log|w|_dps, axis) target.
    (4-d target; diag-Gaussian NLL with mixture-softmax aggregation.)

Sampling:
    Given a predicted mixture, draw N samples of (log|w|_dps, axis_4d) by
    (a) picking a component k ~ weights, (b) N(mu_k, sigma_k), (c) splitting
    into mag and axis, normalising axis to unit length, exp of log_mag -> dps.

Also defines a small LCDataset helper that loads shards, normalises, and
yields (features, target) tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalisation constants (per project_memory: we predict in log-space for omega)
# ---------------------------------------------------------------------------

# Typical satellite brightness ranges from ~5 to ~21 mag; we centre on 13 & div by 4.
MAG_MEAN = 13.0
MAG_STD = 4.0

# Observer distance in km; real LEO obs distances are ~1e3 to ~1e5 km.
LOG_DIST_MEAN = 4.0   # ~10^4 km
LOG_DIST_STD = 0.5

# Omega magnitude range targeted: [0.5, 15] dps. Log: [-0.693, 2.708].
LOGWMAG_MEAN = 1.0     # ~2.7 dps centre
LOGWMAG_STD = 1.2      # covers [-0.7, 2.7] at +/- 1.4 sigma


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_features(
    mag_lc: np.ndarray,
    sun_j2k: np.ndarray,
    obs_j2k: np.ndarray,
    obs_dist: np.ndarray,
) -> np.ndarray:
    """Build (B, 8, T) feature tensor from raw per-shard arrays.

    mag_lc: (B, T) or (T,)
    sun_j2k/obs_j2k: (B, T, 3) or (T, 3), already sat-relative or not — both
        will be unit-normed.
    obs_dist: (B, T) or (T,) km.
    """
    if mag_lc.ndim == 1:
        mag_lc = mag_lc[None]
        sun_j2k = sun_j2k[None]
        obs_j2k = obs_j2k[None]
        obs_dist = obs_dist[None]

    sun = sun_j2k / np.maximum(np.linalg.norm(sun_j2k, axis=-1, keepdims=True), 1e-12)
    obs = obs_j2k / np.maximum(np.linalg.norm(obs_j2k, axis=-1, keepdims=True), 1e-12)
    log_dist = np.log10(np.maximum(obs_dist, 1.0))

    mag_n = (mag_lc - MAG_MEAN) / MAG_STD          # (B,T)
    dist_n = (log_dist - LOG_DIST_MEAN) / LOG_DIST_STD   # (B,T)

    feats = np.concatenate(
        [mag_n[..., None], sun, obs, dist_n[..., None]], axis=-1,
    )  # (B, T, 8)
    # CNN expects (B, C, T)
    return np.transpose(feats, (0, 2, 1))


def omega_to_target(omega: np.ndarray) -> np.ndarray:
    """omega (B,3) rad/s -> target (B,4) = [log(|w|_dps)_normed, axis_xyz].

    Axis is on S^2; we regress all 4 coords and re-project on sampling.
    """
    mag_rad = np.linalg.norm(omega, axis=-1)                       # (B,)
    mag_dps = np.degrees(mag_rad)
    log_mag = np.log(np.maximum(mag_dps, 1e-6))
    log_mag_n = (log_mag - LOGWMAG_MEAN) / LOGWMAG_STD
    axis = omega / np.maximum(mag_rad[..., None], 1e-12)           # (B,3)
    return np.concatenate([log_mag_n[:, None], axis], axis=1)      # (B,4)


def target_to_omega(tgt: np.ndarray) -> np.ndarray:
    """Inverse of omega_to_target. tgt (B,4) -> omega (B,3) rad/s."""
    log_mag_n = tgt[..., 0]
    log_mag = log_mag_n * LOGWMAG_STD + LOGWMAG_MEAN
    mag_dps = np.exp(log_mag)
    mag_rad = np.deg2rad(mag_dps)
    axis = tgt[..., 1:4]
    axis = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    return axis * mag_rad[..., None]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LCEncoder(nn.Module):
    """Small 1-D CNN backbone. Input (B, 8, T=500). Output (B, 128)."""

    def __init__(self, in_ch: int = 8, hidden: int = 128):
        super().__init__()
        def conv_block(cin, cout, k=7, s=1):
            return nn.Sequential(
                nn.Conv1d(cin, cout, k, stride=s, padding=k // 2),
                nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
                nn.ReLU(inplace=True),
            )
        self.net = nn.Sequential(
            conv_block(in_ch, 32, k=7),         # 500
            nn.MaxPool1d(2),                    # 250
            conv_block(32, 64, k=7),            # 250
            nn.MaxPool1d(2),                    # 125
            conv_block(64, 128, k=5),           # 125
            nn.MaxPool1d(2),                    # 62
            conv_block(128, 128, k=5),          # 62
            conv_block(128, hidden, k=3),       # 62
        )
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)                          # (B, hidden, T')
        h = h.mean(dim=-1)                       # global avg pool
        return F.relu(self.proj(h))


class MDNHead(nn.Module):
    """Mixture-of-Gaussians head over a 4-d target (log|w|, axis_xyz).

    Output parameters:
      log_weights: (B, K)
      means:       (B, K, 4)
      log_stds:    (B, K, 4)  (diag Gaussian per component)
    """

    def __init__(self, feat_dim: int, K: int = 8, target_dim: int = 4):
        super().__init__()
        self.K = K
        self.D = target_dim
        self.pi = nn.Linear(feat_dim, K)
        self.mu = nn.Linear(feat_dim, K * target_dim)
        self.log_sigma = nn.Linear(feat_dim, K * target_dim)
        # init log_sigma to ~0.5 for stable start (sigma ~1.6)
        nn.init.constant_(self.log_sigma.bias, 0.5)

    def forward(self, feat: torch.Tensor):
        B = feat.shape[0]
        log_w = F.log_softmax(self.pi(feat), dim=-1)            # (B, K)
        mu = self.mu(feat).view(B, self.K, self.D)              # (B, K, D)
        log_sigma = self.log_sigma(feat).view(B, self.K, self.D)
        # Clamp for stability
        log_sigma = torch.clamp(log_sigma, min=-4.0, max=3.0)
        return log_w, mu, log_sigma


class LCInverseMDN(nn.Module):
    def __init__(self, K: int = 8):
        super().__init__()
        self.encoder = LCEncoder(in_ch=8, hidden=128)
        self.head = MDNHead(feat_dim=128, K=K, target_dim=4)
        self.K = K

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)
        return self.head(feat)


# ---------------------------------------------------------------------------
# MDN NLL loss
# ---------------------------------------------------------------------------

def mdn_nll(
    log_w: torch.Tensor,    # (B, K)
    mu: torch.Tensor,       # (B, K, D)
    log_sigma: torch.Tensor,  # (B, K, D)
    target: torch.Tensor,   # (B, D)
) -> torch.Tensor:
    """Negative log-likelihood of target under mixture of diag-Gaussians."""
    target = target.unsqueeze(1)  # (B, 1, D)
    var = torch.exp(2.0 * log_sigma)
    log_norm = -0.5 * ((target - mu) ** 2 / var).sum(-1) \
               - log_sigma.sum(-1) \
               - 0.5 * mu.shape[-1] * float(np.log(2 * np.pi))
    # log N_k(target) + log_w_k
    log_joint = log_w + log_norm
    log_px = torch.logsumexp(log_joint, dim=-1)  # (B,)
    return -log_px.mean()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_mixture(
    log_w: torch.Tensor,   # (K,)
    mu: torch.Tensor,      # (K, D)
    log_sigma: torch.Tensor, # (K, D)
    n_samples: int,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Draw n_samples from a single-item mixture. Returns np.ndarray (n_samples, D)."""
    weights = torch.softmax(log_w, dim=-1)
    K, D = mu.shape
    # pick component indices
    comp = torch.multinomial(weights, n_samples, replacement=True, generator=generator)
    mu_k = mu[comp]                       # (n_samples, D)
    sigma_k = torch.exp(log_sigma[comp])  # (n_samples, D)
    eps = torch.randn(n_samples, D, generator=generator)
    return (mu_k + sigma_k * eps).cpu().numpy()


def mixture_mode_mean(log_w: torch.Tensor, mu: torch.Tensor) -> np.ndarray:
    """Mean of the highest-weight component as a point-estimate."""
    k = int(torch.argmax(log_w).item())
    return mu[k].cpu().numpy()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test():
    torch.manual_seed(0)
    model = LCInverseMDN(K=8)
    dummy = torch.randn(4, 8, 500)
    log_w, mu, log_sigma = model(dummy)
    assert log_w.shape == (4, 8)
    assert mu.shape == (4, 8, 4)
    assert log_sigma.shape == (4, 8, 4)
    tgt = torch.randn(4, 4)
    loss = mdn_nll(log_w, mu, log_sigma, tgt)
    loss.backward()
    # Check grads finite
    for n, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad {n}"
    print(f"smoke_test OK: loss={loss.item():.4f}, param count={sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    _smoke_test()
