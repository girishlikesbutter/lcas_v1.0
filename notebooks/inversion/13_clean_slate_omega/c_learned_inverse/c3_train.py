"""c3_train.py

Train the MDN regressor on shards produced by c1.

Usage:
    python c3_train.py --data-dir <dir-with-shards> \
        [--n-epochs 50] [--batch-size 256] [--lr 1e-3]
        [--val-frac 0.05] [--tag pilot]

Outputs (to data/.../c_learned_inverse/runs/<tag>/):
    - ckpt_epoch_XXX.pt    (every 10 epochs + final)
    - best.pt              (lowest val NLL)
    - train_log.json       (loss + val metrics per epoch)
    - feature_stats.json   (normalisation metadata)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from c2_model import (
    LCInverseMDN, build_features, omega_to_target, target_to_omega,
    mdn_nll, sample_mixture, LOGWMAG_MEAN, LOGWMAG_STD,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNS_ROOT = (PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
             / "13_clean_slate_omega" / "c_learned_inverse" / "runs")


class ShardDataset(Dataset):
    """Loads all shards into memory as torch tensors.

    At ~1.4 MB per 100 samples (mag+geometry+target), 100k samples is ~14 GB if we
    stored everything float64. We keep float32 throughout. The bottleneck is
    sun+obs+dist geometry (B, 500, 3+3+1) = 14 floats * 500 = 7000 floats per
    sample -> 100k * 7000 * 4 bytes = 2.8 GB. Plus mag_lc (100k * 500 * 4) = 200 MB.
    Fits in RAM on a 16-core box comfortably.
    """

    def __init__(self, shard_paths: List[Path]):
        feats_list, tgt_list = [], []
        for p in shard_paths:
            d = np.load(p)
            mag = d["mag_lc"].astype(np.float32)              # (B, T)
            # Clip any infs
            mag = np.where(np.isfinite(mag), mag, 25.0)
            feats = build_features(mag, d["sun_j2k"].astype(np.float32),
                                   d["obs_j2k"].astype(np.float32),
                                   d["obs_dist"].astype(np.float32))  # (B, 8, T)
            tgt = omega_to_target(d["omega"].astype(np.float32))      # (B, 4)
            feats_list.append(feats.astype(np.float32))
            tgt_list.append(tgt.astype(np.float32))
        self.x = torch.from_numpy(np.concatenate(feats_list, axis=0))
        self.y = torch.from_numpy(np.concatenate(tgt_list, axis=0))
        print(f"ShardDataset: x={tuple(self.x.shape)} y={tuple(self.y.shape)}")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _val_metrics(model: LCInverseMDN, loader: DataLoader, device) -> dict:
    model.eval()
    tot_nll = 0.0
    tot_n = 0
    dir_deg_list = []
    mag_pct_list = []
    mode_nll = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            log_w, mu, log_sigma = model(x)
            loss = mdn_nll(log_w, mu, log_sigma, y)
            tot_nll += loss.item() * x.shape[0]
            tot_n += x.shape[0]

            # top-1 mode mean
            comp = torch.argmax(log_w, dim=-1)             # (B,)
            mu_top = mu[torch.arange(x.shape[0]), comp]    # (B, 4)
            pred_np = mu_top.cpu().numpy()
            y_np = y.cpu().numpy()

            pred_omega = target_to_omega(pred_np)
            true_omega = target_to_omega(y_np)

            # Dir error
            pred_dir = pred_omega / np.maximum(np.linalg.norm(pred_omega, axis=1, keepdims=True), 1e-12)
            true_dir = true_omega / np.maximum(np.linalg.norm(true_omega, axis=1, keepdims=True), 1e-12)
            cos_a = np.clip(np.sum(pred_dir * true_dir, axis=1), -1.0, 1.0)
            dir_deg = np.degrees(np.arccos(cos_a))
            dir_deg_list.append(dir_deg)

            # Mag % error
            pred_mag = np.linalg.norm(pred_omega, axis=1)
            true_mag = np.linalg.norm(true_omega, axis=1)
            mag_pct = (pred_mag - true_mag) / np.maximum(true_mag, 1e-12) * 100.0
            mag_pct_list.append(mag_pct)

    dir_arr = np.concatenate(dir_deg_list)
    mag_arr = np.concatenate(mag_pct_list)
    return {
        "val_nll": tot_nll / tot_n,
        "top1_mean_dir_deg": float(np.mean(dir_arr)),
        "top1_median_dir_deg": float(np.median(dir_arr)),
        "top1_mean_mag_pct": float(np.mean(np.abs(mag_arr))),
        "top1_median_mag_pct": float(np.median(np.abs(mag_arr))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True,
                    help="dir with shard_*.npz from c1")
    ap.add_argument("--tag", type=str, default="pilot")
    ap.add_argument("--n-epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    shard_paths = sorted(data_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise SystemExit(f"No shards in {data_dir}")
    print(f"Found {len(shard_paths)} shards in {data_dir}")

    ds = ShardDataset(shard_paths)
    n_val = int(len(ds) * args.val_frac)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    print(f"train={n_train}, val={n_val}, batch={args.batch_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LCInverseMDN(K=args.K).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"Model: LCInverseMDN K={args.K}, params={nparam}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)

    out_dir = RUNS_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs: {out_dir}")

    stats_path = out_dir / "feature_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "LOGWMAG_MEAN": LOGWMAG_MEAN,
            "LOGWMAG_STD": LOGWMAG_STD,
        }, f, indent=2)
    print(f"Saved: {stats_path}")

    log: list[dict] = []
    best_val = float("inf")
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss_sum, tr_n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            log_w, mu, log_sigma = model(x)
            loss = mdn_nll(log_w, mu, log_sigma, y)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            tr_loss_sum += loss.item() * x.shape[0]
            tr_n += x.shape[0]
        scheduler.step()
        tr_loss = tr_loss_sum / tr_n

        val_metrics = _val_metrics(model, val_loader, device)
        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_nll": tr_loss,
            "elapsed_s": elapsed,
            "lr": optim.param_groups[0]["lr"],
            **val_metrics,
        }
        log.append(entry)
        print(f"Epoch {epoch:3d} | train {tr_loss:.3f} | val {val_metrics['val_nll']:.3f} "
              f"| top1 dir {val_metrics['top1_mean_dir_deg']:.1f}° (med {val_metrics['top1_median_dir_deg']:.1f}) "
              f"| mag {val_metrics['top1_mean_mag_pct']:.1f}% "
              f"| {elapsed:.1f}s")

        # Save best
        if val_metrics["val_nll"] < best_val:
            best_val = val_metrics["val_nll"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "args": vars(args),
                "val_metrics": val_metrics,
            }, best_path)

        # Periodic checkpoint
        if epoch % args.save_every == 0 or epoch == args.n_epochs:
            cp = out_dir / f"ckpt_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "args": vars(args),
                "val_metrics": val_metrics,
            }, cp)
            print(f"Saved: {cp}")

        # Always save log incrementally
        with open(out_dir / "train_log.json", "w") as f:
            json.dump({"args": vars(args), "epochs": log}, f, indent=2)

    print(f"\nFinal best val NLL = {best_val:.3f}")
    print(f"Saved: {best_path}")
    print(f"Saved: {out_dir / 'train_log.json'}")


if __name__ == "__main__":
    main()
