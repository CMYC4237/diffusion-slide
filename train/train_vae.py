# -*- coding: utf-8 -*-
"""训练第一阶 VAE: 谱面图像重建。
用法: python train/train_vae.py [--epochs N] [--batch B] [--lr L] [--device cuda|cpu]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_vae import SlideWindowDataset
from vae import AutoencoderKL, ChartReconstructLoss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", default="data/dataset/slide_clean.jsonl")
    ap.add_argument("--out", default="checkpoints/vae")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-checkpoint", action="store_true", default=False)
    ap.add_argument("--cache", action="store_true", default=False)
    ap.add_argument("--samples_per_epoch", type=int, default=1500)
    ap.add_argument("--split", type=float, default=0.9)
    args = ap.parse_args()

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    full = SlideWindowDataset(args.data, window_frames=args.window, cache_in_mem=args.cache)
    n = len(full)
    n_train = int(n * args.split)
    # 用前 90% 谱面做 train
    train = SlideWindowDataset(args.data, window_frames=args.window, cache_in_mem=args.cache, seed=0)
    val = SlideWindowDataset(args.data, window_frames=args.window, cache_in_mem=False, seed=1)
    # 简单切分: 按窗口起点索引
    train_idxs = list(range(n_train))
    val_idxs = list(range(n_train, n))
    k = min(args.samples_per_epoch, len(train_idxs))
    tr_dl = DataLoader(train, batch_size=args.batch,
                       sampler=torch.utils.data.RandomSampler(train_idxs, num_samples=k, replacement=False),
                       num_workers=args.workers, drop_last=True, pin_memory=True)
    va_dl = DataLoader(val, batch_size=args.batch, sampler=torch.utils.data.SequentialSampler(val_idxs),
                       num_workers=0, pin_memory=True)

    model = AutoencoderKL(use_checkpoint=not args.no_checkpoint).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = ChartReconstructLoss()
    n_params = sum(p.numel() for p in model.parameters())
    amp_dtype = torch.bfloat16 if args.amp else torch.float32
    print(f"train样本 {len(train_idxs)}, val {len(val_idxs)}, 参数 {n_params/1e6:.2f}M, device {args.device}, amp={args.amp}, checkpoint={not args.no_checkpoint}")

    best_val = 1e9
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tr_loss = 0.0
        tr_n = 0
        for x, meta in tr_dl:
            x = x.to(args.device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                rec, z, mean, logvar = model(x)
                rec_loss, _ = loss_fn(rec, x)
            kl = torch.mean(0.5 * (mean.pow(2) + logvar.exp() - 1 - logvar))
            loss = rec_loss + model.kl_weight * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * x.size(0)
            tr_n += x.size(0)
        sched.step()

        # val
        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for x, meta in va_dl:
                x = x.to(args.device)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                    rec, z, mean, logvar = model(x)
                    rec_loss, _ = loss_fn(rec, x)
                kl = torch.mean(0.5 * (mean.pow(2) + logvar.exp() - 1 - logvar))
                loss = rec_loss + model.kl_weight * kl
                va_loss += loss.item() * x.size(0)
                va_n += x.size(0)
        print(f"ep {ep}: train {tr_loss/tr_n:.4f}, val {va_loss/va_n:.4f}, "
              f"lr {sched.get_last_lr()[0]:.2e}, {time.time()-t0:.0f}s")
        if va_loss / va_n < best_val:
            best_val = va_loss / va_n
            ckpt = {
                "model": model.state_dict(),
                "epoch": ep,
                "val_loss": best_val,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.out, "best.ckpt"))
            print(f"  -> saved best.ckpt (val {best_val:.4f})")
        torch.save({"model": model.state_dict(), "epoch": ep},
                   os.path.join(args.out, "last.ckpt"))


if __name__ == "__main__":
    main()
