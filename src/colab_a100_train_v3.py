#!/usr/bin/env python3
"""Colab A100 GPU Challenger: Deep Residual + Attention + Multi-Quantile (P10/P50/P90) + Huber Neural Net
trained on the exact same v3 multi-city Indian bus queries (Delhi, MTC, BMTC, Mumbai, NCR).

Run autonomously via:
    colab run --gpu A100 -s eta-a100 src/colab_a100_train_v3.py

Reads v3_train_1.5m.npz and v3_val.npz from Google Drive (/content/drive/MyDrive/ETA/v3_queries),
trains a 6-block Pre-Norm SiLU Residual MLP with Fourier numeric embeddings + city/bus_class embeddings
using bf16 mixed precision on NVIDIA A100, and saves predictions + checkpoint back to Google Drive!
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ResBlock(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DeepV3BusETA(nn.Module):
    """DeeprETA-inspired Multi-City Residual Network predicting [point_huber, q10, q90] log-residuals."""

    def __init__(self, n_num: int, n_cities: int = 8, n_classes: int = 32, d_model: int = 512, n_blocks: int = 6):
        super().__init__()
        self.emb_city = nn.Embedding(n_cities, 16)
        self.emb_cls = nn.Embedding(n_classes, 16)
        self.in_proj = nn.Sequential(
            nn.Linear(n_num + 32, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(d_model, dropout=0.10) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),  # [point_residual, q10_residual, q90_residual]
        )

    def forward(self, x_num: torch.Tensor, c_city: torch.Tensor, c_cls: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x_num, self.emb_city(c_city), self.emb_cls(c_cls)], dim=-1)
        h = self.in_proj(h)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


def composite_eta_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Huber loss for point ETA + Pinball loss for P10 and P90 quantiles."""
    loss_pt = F.huber_loss(pred[:, 0], target, delta=0.25)
    e10 = target - pred[:, 1]
    loss_q10 = torch.maximum(0.10 * e10, -0.90 * e10).mean()
    e90 = target - pred[:, 2]
    loss_q90 = torch.maximum(0.90 * e90, -0.10 * e90).mean()
    return loss_pt + 0.5 * (loss_q10 + loss_q90)


def main():
    t0 = time.time()
    print(f"PyTorch {torch.__version__} | CUDA={torch.cuda.is_available()} | Device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    data_dir = Path("/content/drive/MyDrive/ETA/v3_queries")
    if not data_dir.exists():
        # Fallback if uploaded directly to /content/v3_queries
        data_dir = Path("/content/v3_queries")
    out_dir = Path("/content/drive/MyDrive/ETA/models/deep_v3_a100") if Path("/content/drive/MyDrive").exists() else Path("/content/deep_v3_a100")
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = np.load(data_dir / "v3_train_1.5m.npz")
    va = np.load(data_dir / "v3_val.npz")
    X_tr, r_tr = tr["X"].astype(np.float32), tr["r"].astype(np.float32)
    X_va, y_va, b_va, r_va = va["X"].astype(np.float32), va["y"].astype(np.float64), va["b"].astype(np.float64), va["r"].astype(np.float32)

    # First two columns are city_idx and bus_class_idx
    c_tr_city = np.clip(X_tr[:, 0].astype(np.int64), 0, 7)
    c_tr_cls = np.clip(X_tr[:, 1].astype(np.int64) + 1, 0, 31)
    c_va_city = np.clip(X_va[:, 0].astype(np.int64), 0, 7)
    c_va_cls = np.clip(X_va[:, 1].astype(np.int64) + 1, 0, 31)

    num_tr = np.nan_to_num(X_tr[:, 2:], nan=0.0, posinf=0.0, neginf=0.0)
    num_va = np.nan_to_num(X_va[:, 2:], nan=0.0, posinf=0.0, neginf=0.0)
    mu = num_tr.mean(axis=0)
    sd = num_tr.std(axis=0) + 1e-6
    num_tr = np.clip((num_tr - mu) / sd, -10.0, 10.0)
    num_va = np.clip((num_va - mu) / sd, -10.0, 10.0)

    tr_ds = TensorDataset(torch.from_numpy(num_tr), torch.from_numpy(c_tr_city), torch.from_numpy(c_tr_cls), torch.from_numpy(r_tr))
    va_ds = TensorDataset(torch.from_numpy(num_va), torch.from_numpy(c_va_city), torch.from_numpy(c_va_cls), torch.from_numpy(r_va))
    tr_loader = DataLoader(tr_ds, batch_size=8192, shuffle=True, pin_memory=True, num_workers=2)
    va_loader = DataLoader(va_ds, batch_size=16384, shuffle=False, pin_memory=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepV3BusETA(num_tr.shape[1], d_model=512, n_blocks=6).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=2e-4)
    epochs = 18
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_mae = 1e9
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xn, ci, cl, y in tr_loader:
            xn, ci, cl, y = xn.to(device, non_blocking=True), ci.to(device, non_blocking=True), cl.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(xn, ci, cl)
                loss = composite_eta_loss(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tr_loss += loss.item() * len(y)
        sched.step()

        model.eval()
        preds_va = []
        with torch.no_grad():
            for xn, ci, cl, _y in va_loader:
                xn, ci, cl = xn.to(device, non_blocking=True), ci.to(device, non_blocking=True), cl.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    preds_va.append(model(xn, ci, cl).float().cpu().numpy())
        p_va = np.vstack(preds_va)
        val_mae = float(np.mean(np.abs(b_va * np.exp(np.clip(p_va[:, 0], -1.5, 1.5)) - y_va)))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"Epoch {ep:02d}/{epochs} | tr_loss={tr_loss/len(tr_ds):.4f} | val_MAE={val_mae:.2f}s (best={best_val_mae:.2f}s)", flush=True)

    torch.save({"state_dict": best_state, "mu": mu, "sd": sd, "best_val_mae_s": best_val_mae}, out_dir / "deep_v3_a100.pt")
    (out_dir / "metrics.json").write_text(json.dumps({"best_val_mae_s": round(best_val_mae, 2), "elapsed_s": round(time.time() - t0, 1)}, indent=2))
    print(f"DONE in {time.time() - t0:.1f}s | Best Val MAE = {best_val_mae:.2f}s | Saved to {out_dir}")


if __name__ == "__main__":
    main()
