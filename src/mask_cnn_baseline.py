#!/usr/bin/env python3
"""Faithful PyTorch reproduction of Charul & Biyani (2019, arXiv:1906.09925v1):
"To each route its own ETA: A generative modeling framework for ETA prediction" (Masked PixelCNN).

Architecture (per paper Section III.E & IV.B):
  - Input: H x K matrix of quantized consecutive stop-to-stop travel times (H=10 trips window, K stops)
  - Layer 1: Mask A 5x5 Conv2d (strictly causal: excludes current pixel & all future pixels) + ReLU
  - Layers 2..7: 6x Mask B 5x5 Conv2d (causal: includes current pixel, excludes future pixels) + ReLU
  - Output: 1x1 Conv2d -> C=256/512 discrete travel-time bins (0..1024s) + CrossEntropyLoss
  - Supports Mask 1 (full causal), Mask 2 (local previous trip + previous stop — paper's best!), Mask 3.

Runs on Apple M4 Pro MPS, Colab A100 CUDA, or CPU.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]  # repo root


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type: str, mask_mode: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B")
        kH, kW = self.kernel_size
        cy, cx = kH // 2, kW // 2
        mask = torch.zeros_like(self.weight)

        if mask_mode == 1:
            # Full PixelCNN causal mask
            mask[:, :, :cy, :] = 1
            mask[:, :, cy, :cx] = 1
        elif mask_mode == 2:
            # Paper's Mask 2 (best in Table I): immediate past trip same stop + current trip previous stop
            if cy > 0:
                mask[:, :, cy - 1, cx] = 1
            if cx > 0:
                mask[:, :, cy, cx - 1] = 1
        else:
            # Paper's Mask 3: past 2 trips same stop + current trip past 2 stops
            mask[:, :, :cy, cx] = 1
            mask[:, :, cy, :cx] = 1

        if mask_type == "B":
            mask[:, :, cy, cx] = 1
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MaskCNNETA(nn.Module):
    def __init__(self, n_classes: int = 256, hidden: int = 64, n_blocks: int = 6, kernel_size: int = 5, mask_mode: int = 2):
        super().__init__()
        pad = kernel_size // 2
        layers: list[nn.Module] = [
            MaskedConv2d("A", mask_mode, 1, hidden, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_blocks):
            layers.append(MaskedConv2d("B", mask_mode, hidden, hidden, kernel_size, padding=pad))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(hidden, n_classes, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, C, H, K)


def build_route_matrices(
    df_seg: pl.DataFrame,
    route_id: str,
    window_h: int = 10,
    max_eta_s: float = 1024.0,
    n_classes: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (N, H, K) quantized travel-time matrices for a single route from segment_events."""
    r = df_seg.filter(pl.col("route_id") == route_id).sort(["day", "t_start"])
    if len(r) < 100:
        return np.empty((0,)), np.empty((0,)), []

    # Pivot each trip to ordered stop_seq durations
    seqs = sorted(r["stop_seq"].unique().to_list())
    K = len(seqs)
    if K < 5:
        return np.empty((0,)), np.empty((0,)), []
    seq_to_col = {s: idx for idx, s in enumerate(seqs)}

    # Group by (day, trip_id)
    trips = (
        r.group_by(["day", "trip_id"])
        .agg([
            pl.col("t_start").min().alias("t0"),
            pl.col("stop_seq").alias("ss"),
            pl.col("dur_s").alias("dd"),
        ])
        .sort(["day", "t0"])
    )

    # Fill per-day matrices of shape (T_day, K)
    bin_w = max_eta_s / n_classes
    mats = []
    mat_days = []
    # Global median per stop_seq for missing cell imputation
    med_by_seq = dict(
        r.group_by("stop_seq").agg(pl.col("dur_s").median()).iter_rows()
    )
    fallback_row = np.array([med_by_seq.get(s, 60.0) for s in seqs], dtype=np.float32)

    for day, grp in trips.group_by("day", maintain_order=True):
        day_str = day[0] if isinstance(day, tuple) else str(day)
        rows = []
        masks = []
        for row in grp.iter_rows(named=True):
            vec = fallback_row.copy()
            obs = np.zeros(K, dtype=np.float32)
            for s, d in zip(row["ss"], row["dd"]):
                if s in seq_to_col and 1.0 <= d <= max_eta_s:
                    c = seq_to_col[s]
                    vec[c] = d
                    obs[c] = 1.0
            if obs.sum() >= max(3, int(0.3 * K)):
                rows.append(vec)
                masks.append(obs)
        if len(rows) < window_h:
            continue
        arr = np.stack(rows, axis=0)
        msk = np.stack(masks, axis=0)
        for start in range(0, len(arr) - window_h + 1):
            mats.append((arr[start:start + window_h], msk[start:start + window_h]))
            mat_days.append(day_str)

    if not mats:
        return np.empty((0,)), np.empty((0,)), []
    vals = np.stack([m[0] for m in mats], axis=0)
    obs_masks = np.stack([m[1] for m in mats], axis=0)
    return vals, obs_masks, mat_days


def benchmark_route(
    df_seg: pl.DataFrame,
    route_id: str,
    test_day: str,
    epochs: int = 25,
    n_classes: int = 256,
    max_eta_s: float = 1024.0,
) -> dict | None:
    vals, obs_masks, mat_days = build_route_matrices(df_seg, route_id, n_classes=n_classes, max_eta_s=max_eta_s)
    if len(mat_days) < 20:
        return None
    bin_w = max_eta_s / n_classes
    classes = np.clip((vals / bin_w).astype(np.int64), 0, n_classes - 1)
    norm_in = (classes.astype(np.float32) / float(n_classes - 1))[:, None, :, :]  # (N, 1, H, K)

    tr_idx = [i for i, d in enumerate(mat_days) if d < test_day]
    te_idx = [i for i, d in enumerate(mat_days) if d == test_day]
    if len(tr_idx) < 10 or len(te_idx) < 2:
        return None

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MaskCNNETA(n_classes=n_classes, hidden=64, n_blocks=6, kernel_size=5, mask_mode=2).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    X_tr = torch.tensor(norm_in[tr_idx], device=device)
    Y_tr = torch.tensor(classes[tr_idx], device=device)
    M_tr = torch.tensor(obs_masks[tr_idx], device=device)

    X_te = torch.tensor(norm_in[te_idx], device=device)
    V_te = vals[te_idx]
    M_te = obs_masks[te_idx]

    bs = 32
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(X_tr), device=device)
        for i in range(0, len(X_tr), bs):
            idx = perm[i:i + bs]
            logits = model(X_tr[idx])  # (B, C, H, K)
            loss_raw = F.cross_entropy(logits, Y_tr[idx], reduction="none")
            loss = (loss_raw * M_tr[idx]).sum() / M_tr[idx].sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_te)  # (N_te, C, H, K)
        probs = F.softmax(logits, dim=1)
        # Expected value over discrete bins (smoother than argmax)
        bins = torch.arange(n_classes, device=device, dtype=torch.float32).view(1, -1, 1, 1) * bin_w + 0.5 * bin_w
        pred_s = (probs * bins).sum(dim=1).cpu().numpy()

    # Evaluate on the last row (current trip) where observed
    pred_last = pred_s[:, -1, :]
    true_last = V_te[:, -1, :]
    mask_last = M_te[:, -1, :] > 0.5
    if not np.any(mask_last):
        return None
    err = np.abs(pred_last[mask_last] - true_last[mask_last])
    mape = float(np.mean(err / np.maximum(true_last[mask_last], 1.0)) * 100.0)
    return {
        "route_id": route_id,
        "train_windows": len(tr_idx),
        "test_windows": len(te_idx),
        "n_obs": int(mask_last.sum()),
        "mae_s": round(float(np.mean(err)), 2),
        "med_s": round(float(np.median(err)), 2),
        "mape_pct": round(mape, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(ROOT / "data/processed/mtc_segments.parquet"))
    ap.add_argument("--top-routes", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    a = ap.parse_args()

    df = pl.read_parquet(a.parquet)
    days = sorted(df["day"].unique().to_list())
    test_day = days[-2] if len(days) >= 3 else days[-1]
    top_r = df.group_by("route_id").len().sort("len", descending=True).head(a.top_routes)["route_id"].to_list()
    print(f"[Mask-CNN 1906.09925v1] Testing top {len(top_r)} routes on {test_day}")
    results = []
    for rid in top_r:
        t0 = time.time()
        res = benchmark_route(df, rid, test_day, epochs=a.epochs)
        if res:
            print(f"  route={rid:<12} n={res['n_obs']:>5} MAE={res['mae_s']:>6.2f}s Med={res['med_s']:>6.2f}s MAPE={res['mape_pct']:>5.1f}% ({time.time()-t0:.1f}s)")
            results.append(res)
    if results:
        print(f"  AVG MAE across {len(results)} routes: {np.mean([r['mae_s'] for r in results]):.2f}s")


if __name__ == "__main__":
    main()
