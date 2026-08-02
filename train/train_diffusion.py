# -*- coding: utf-8 -*-
"""训练第二阶 latent 扩散模型 (难度 + 音频条件)。
用法: python train/train_diffusion.py [--epochs N] [--batch B] [--vae_ckpt ...]
集群多卡: torchrun --nproc_per_node=2 train/train_diffusion.py --dist
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from vae import AutoencoderKL
from diffusion import UNet, DDPM
from convertor import chart_to_array, FPS
from audio_align import align_mel

WINDOW_FRAMES = 1024
LATENT_ROWS = WINDOW_FRAMES // 8          # 128
LATENT_COLS = 32                          # 256 / 8
LATENT_CH = 16


class LatentAudioDataset(Dataset):
    def __init__(self, jsonl_path, audio_meta_path, vae, device, seed=0,
                 cond_drop_p=0.1, mirror_p=0.5):
        with open(jsonl_path, encoding="utf-8") as f:
            self.recs = [json.loads(l) for l in f]
        with open(audio_meta_path, encoding="utf-8") as f:
            self.audio_meta = json.load(f)
        self.vae = vae
        self.device = device
        self.rng = random.Random(seed)
        self.cond_drop_p = cond_drop_p
        self.mirror_p = mirror_p
        self._mel_cache = {}
        # 窗口槽: (谱面下标, 槽号)
        self.slots = []
        for i, r in enumerate(self.recs):
            t_max = r["length_beat"]
            for n in r["notes"]:
                if n.get("seg"):
                    t_max = max(t_max, n["t"] + n["seg"][-1]["dt"])
            h_full = int(np.ceil(t_max * FPS)) + 2
            max_start = max(0, h_full - WINDOW_FRAMES)
            n_slots = max(1, max_start // 256 + 1)
            for w in range(n_slots):
                self.slots.append((i, w))

    def _mel(self, song_id):
        sid = str(song_id)
        if sid not in self._mel_cache:
            if sid not in self.audio_meta:
                return None
            self._mel_cache[sid] = np.load(self.audio_meta[sid]["mel"])
        return self._mel_cache[sid]

    def __len__(self):
        # 每谱面多个窗口槽 (按 256 帧步进), 提高样本量
        return len(self.slots)

    def __getitem__(self, idx):
        i, w = self.slots[idx]
        r = self.recs[i]
        # 窗口起点 (槽 + 随机偏移)
        t_max = r["length_beat"]
        for n in r["notes"]:
            if n.get("seg"):
                t_max = max(t_max, n["t"] + n["seg"][-1]["dt"])
        h_full = int(np.ceil(t_max * FPS)) + 2
        max_start = max(0, h_full - WINDOW_FRAMES)
        s = min(w * 256 + self.rng.randint(0, 256), max_start)
        t_start = s / FPS
        t_end = t_start + WINDOW_FRAMES / FPS

        # 渲染窗口 -> VAE latent
        arr = chart_to_array(r["notes"], r["length_beat"], t_start=t_start, t_end=t_end).astype(np.float32)
        if arr.shape[1] < WINDOW_FRAMES:
            pad = np.zeros((9, WINDOW_FRAMES - arr.shape[1], 256), np.float32)
            arr = np.concatenate([arr, pad], 1)
        if arr.shape[1] > WINDOW_FRAMES:
            arr = arr[:, :WINDOW_FRAMES, :]
        if self.rng.random() < self.mirror_p:
            arr = np.flip(arr, axis=2).copy()
        x = torch.from_numpy(arr)[None].to(self.device)
        with torch.no_grad():
            z = self.vae.encode(x)[0]  # (1, 16, 128, 32)
        latent = z[0].cpu().numpy()

        # 音频对齐特征
        mel = self._mel(r["song_id"])
        if mel is not None:
            bpms = [[b[0], b[1]] for b in r["bpms"]] if r.get("bpms") else [[0.0, r["bpm"]]]
            # 截取窗口对应段
            ctx = align_mel(mel, bpms, n_rows=LATENT_ROWS + 4, latent_rows_per_beat=0.5)
            start_row = int(round(t_start / 0.5))
            ctx = ctx[start_row:start_row + LATENT_ROWS]
            if ctx.shape[0] < LATENT_ROWS:
                ctx = np.pad(ctx, ((0, LATENT_ROWS - ctx.shape[0]), (0, 0)))
        else:
            ctx = np.zeros((LATENT_ROWS, 128), dtype=np.float32)

        lv = r["lv"] if r["lv"] is not None else 0

        # 条件丢弃 (CFG)
        use_cond = self.rng.random() >= self.cond_drop_p
        if not use_cond:
            lv = 0
            ctx = np.zeros_like(ctx)

        return {
            "latent": torch.from_numpy(latent),
            "lv": torch.tensor(lv, dtype=torch.long),
            "audio": torch.from_numpy(ctx),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--vae_ckpt", default="checkpoints/vae_smoke/best.ckpt")
    ap.add_argument("--jsonl", default="data/dataset/slide_clean.jsonl")
    ap.add_argument("--audio_meta", default="data/audio/meta.json")
    ap.add_argument("--out", default="checkpoints/diffusion")
    ap.add_argument("--samples_per_epoch", type=int, default=700)
    ap.add_argument("--dist", action="store_true", default=False)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--cond_drop_p", type=float, default=0.1)
    args = ap.parse_args()

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    if args.dist:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        rank, world = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # VAE (冻结)
    vae = AutoencoderKL(use_checkpoint=False).to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()

    # 扩散模型
    unet = UNet(with_audio=True, with_lv=True, use_checkpoint=True).to(device)
    ddpm = DDPM(unet).to(device)
    opt = torch.optim.AdamW(ddpm.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n_params = sum(p.numel() for p in ddpm.parameters())

    if args.dist:
        ddpm = torch.nn.parallel.DistributedDataParallel(ddpm, device_ids=[rank])

    ds = LatentAudioDataset(args.jsonl, args.audio_meta, vae, device, cond_drop_p=args.cond_drop_p)
    if args.dist:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=world, rank=rank, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(
            ds, num_samples=min(args.samples_per_epoch, len(ds)), replacement=False)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=0, drop_last=True)

    if rank == 0:
        print(f"扩散模型参数 {n_params/1e6:.2f}M, device {device}, dist={args.dist}")

    for ep in range(args.epochs):
        if args.dist:
            sampler.set_epoch(ep)
        ddpm.train()
        t0 = time.time()
        tot = 0.0
        n = 0
        for batch in dl:
            latent = batch["latent"].to(device)
            lv = batch["lv"].to(device)
            audio = batch["audio"].to(device)
            B = latent.size(0)
            t = torch.randint(0, ddpm.timesteps, (B,), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                loss = ddpm.training_losses(latent, t, lv, audio)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            opt.step()
            tot += loss.item() * B
            n += B
        sched.step()
        if rank == 0:
            msg = f"ep {ep}: loss {tot/max(1,n):.4f}, lr {sched.get_last_lr()[0]:.2e}, {time.time()-t0:.0f}s"
            print(msg, flush=True)
            torch.save({"model": ddpm.state_dict(), "epoch": ep, "args": vars(args)},
                       os.path.join(args.out, "last.ckpt"))
            if ep % 5 == 0 or ep == args.epochs - 1:
                torch.save({"model": ddpm.state_dict(), "epoch": ep},
                           os.path.join(args.out, f"ep{ep:03d}.ckpt"))
    if args.dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
