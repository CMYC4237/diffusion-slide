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
# per-channel latent 标准化 (60谱面x3窗口统计): (x-mean)/std, 让扩散面对 ~N(0,1)
LATENT_MEAN = [0.0924, 0.6350, 0.6057, -0.3221, 0.3398, 0.2305, 0.2297, 0.1997,
               0.6799, 1.0758, 0.0394, -0.2274, -0.9928, 0.1380, -0.0272, -0.2712]
LATENT_STD = [0.2940, 0.7292, 0.3580, 0.2502, 0.2233, 0.2907, 0.2922, 0.3284,
              0.3222, 0.4417, 0.6906, 0.3618, 0.4912, 0.5780, 0.2578, 0.4207]


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
        self._latent_cache = {}  # (谱面下标, 槽号) -> latent, 槽固定后可复用
        self.latent_mean = torch.tensor(LATENT_MEAN, dtype=torch.float32, device=device).view(16, 1, 1)
        self.latent_std = torch.tensor(LATENT_STD, dtype=torch.float32, device=device).view(16, 1, 1)
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
        # 窗口起点: 固定槽起点 (w*256), 保证 latent 可缓存
        t_max = r["length_beat"]
        for n in r["notes"]:
            if n.get("seg"):
                t_max = max(t_max, n["t"] + n["seg"][-1]["dt"])
        h_full = int(np.ceil(t_max * FPS)) + 2
        max_start = max(0, h_full - WINDOW_FRAMES)
        s = min(w * 256, max_start)
        t_start = s / FPS
        t_end = t_start + WINDOW_FRAMES / FPS

        # latent 缓存 (槽固定 -> 可复用)
        cache_key = (i, w)
        if cache_key not in self._latent_cache:
            arr = chart_to_array(r["notes"], r["length_beat"], t_start=t_start, t_end=t_end).astype(np.float32)
            if arr.shape[1] < WINDOW_FRAMES:
                pad = np.zeros((10, WINDOW_FRAMES - arr.shape[1], 256), np.float32)
                arr = np.concatenate([arr, pad], 1)
            if arr.shape[1] > WINDOW_FRAMES:
                arr = arr[:, :WINDOW_FRAMES, :]
            x = torch.from_numpy(arr)[None].to(self.device)
            with torch.no_grad():
                z = self.vae.encode(x)[0]  # (1, 16, 128, 32)
            self._latent_cache[cache_key] = ((z[0] - self.latent_mean) / self.latent_std).cpu().numpy()
        latent = self._latent_cache[cache_key]

        # 镜像增强 (latent 空间翻转)
        if self.rng.random() < self.mirror_p:
            latent = np.flip(latent, axis=2).copy()

        # 音频对齐特征
        mel = self._mel(r["song_id"])
        if mel is not None:
            bpms = [[b[0], b[1]] for b in r["bpms"]] if r.get("bpms") else [[0.0, r["bpm"]]]
            # 时间缩放增强: 随机变速 0.9~1.1x (谱面拍结构不变, 音频对齐关系随 bpm 变化)
            rate = 1.0
            if self.rng.random() < 0.4:
                rate = self.rng.uniform(0.9, 1.1)
            bpms_scaled = [[b, bpm * rate] for b, bpm in bpms]
            # 窗口对齐修复: 直接生成从 t_start 起的窗口段 (修复 93.8% 窗口越界全零 bug)
            ctx = align_mel(mel, bpms_scaled, n_rows=LATENT_ROWS + 4, latent_rows_per_beat=0.5,
                            t_start=t_start)[:LATENT_ROWS]
            if ctx.shape[0] < LATENT_ROWS:
                ctx = np.pad(ctx, ((0, LATENT_ROWS - ctx.shape[0]), (0, 0)))
            # 音频增强: 频带 mask (随机抹 1-2 个连续频带, 模拟信息缺失)
            if self.rng.random() < 0.3:
                n_bins = ctx.shape[1]
                w = self.rng.randint(4, 12)
                b0 = self.rng.randint(0, max(1, n_bins - w))
                ctx[:, b0:b0 + w] = -80.0
            # 音频增强: 变调 (mel 频带轴平移 ±1, 模拟音高变化)
            if self.rng.random() < 0.3:
                shift = self.rng.choice([-1, 1])
                ctx = np.roll(ctx, shift, axis=1)
        else:
            ctx = np.full((LATENT_ROWS, 128), -80.0, dtype=np.float32)

        lv = (r["lv"] if r["lv"] is not None else 0) + 1  # 1~26, 0 专用于 CFG null (修复 lv=0 双重语义)

        # 条件丢弃 (CFG): null ctx 用 -80 (无声, 而非 0=最大响度)
        use_cond = self.rng.random() >= self.cond_drop_p
        if not use_cond:
            lv = 0
            ctx = np.full_like(ctx, -80.0)

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
    ap.add_argument("--vae_ckpt", default="checkpoints/vae/best.ckpt")
    ap.add_argument("--jsonl", default="data/dataset/slide_clean.jsonl")
    ap.add_argument("--audio_meta", default="data/audio/meta.json")
    ap.add_argument("--out", default="checkpoints/diffusion")
    ap.add_argument("--samples_per_epoch", type=int, default=700)
    ap.add_argument("--dist", action="store_true", default=False)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--cond_drop_p", type=float, default=0.1)
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--warm-start", action="store_true", default=False,
                    help="加载旧权重继续训练(不恢复 opt/sched/epoch, 用于损失/归一化变更后的热启动)")
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
    raw_ddpm = DDPM(unet).to(device)
    n_timesteps = raw_ddpm.timesteps
    opt = torch.optim.AdamW(raw_ddpm.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n_params = sum(p.numel() for p in raw_ddpm.parameters())

    # 断点续训/热启动: 从 last.ckpt 恢复
    start_ep = 0
    last_path = os.path.join(args.out, "last.ckpt")
    if os.path.exists(last_path) and (args.resume or args.warm_start):
        ck = torch.load(last_path, map_location=device)
        raw_ddpm.load_state_dict(ck["model"])
        if args.resume:
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
            if "sched" in ck:
                sched.load_state_dict(ck["sched"])
            start_ep = ck.get("epoch", -1) + 1
            print(f"[resume] 从 epoch {start_ep} 继续", flush=True)
        elif args.warm_start:
            print(f"[warm-start] 加载权重从 epoch 0 重新训练 (新损失/归一化)", flush=True)

    # DDP: 包装实际被调用的 UNet 层 (training_losses 内部 self.unet(...) 走 DDP forward 才能同步梯度)
    if args.dist:
        ddp_unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[rank])
        raw_ddpm.unet = ddp_unet
    ddpm = raw_ddpm
    ddpm_mod = ddpm  # 自定义方法直接可用

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

    for ep in range(start_ep, args.epochs):
        if args.dist:
            sampler.set_epoch(ep)
        ddpm.train()
        t0 = time.time()
        tot = 0.0
        n = 0
        for step, batch in enumerate(dl):
            latent = batch["latent"].to(device)
            lv = batch["lv"].to(device)
            audio = batch["audio"].to(device)
            B = latent.size(0)
            t = torch.randint(0, n_timesteps, (B,), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                loss = ddpm_mod.training_losses(latent, t, lv, audio)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            opt.step()
            tot += loss.item() * B
            n += B
            # 进度条 (rank0, 每 epoch 约 20 行)
            if rank == 0 and args.epochs > 1:
                total = max(1, len(dl))
                log_every = max(1, total // 20)
                if step % log_every == 0 or step == total - 1:
                    elapsed = time.time() - t0
                    avg = tot / max(1, n)
                    eta = elapsed / (step + 1) * (total - step - 1)
                    pct = 100.0 * (step + 1) / total
                    print(f"\r  [ep {ep}/{args.epochs}] {pct:5.1f}% ({step+1}/{total}) "
                          f"loss {avg:.4f} ETA {eta/60:4.1f}min", end="", flush=True)
        if rank == 0 and args.epochs > 1:
            print("", flush=True)
        sched.step()
        if rank == 0:
            msg = f"ep {ep}: loss {tot/max(1,n):.4f}, lr {sched.get_last_lr()[0]:.2e}, {time.time()-t0:.0f}s"
            print(msg, flush=True)
            sd = ddpm.state_dict()
            sd = {k.replace("unet.module.", "unet."): v for k, v in sd.items()}  # DDP 前缀清理
            torch.save({
                "model": sd,
                "epoch": ep, "args": vars(args),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
            }, last_path)
            if ep % 10 == 0 or ep == args.epochs - 1:
                torch.save({"model": sd, "epoch": ep},
                           os.path.join(args.out, f"ep{ep:03d}.ckpt"))
    if args.dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
