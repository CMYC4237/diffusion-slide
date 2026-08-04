# -*- coding: utf-8 -*-
"""生成质量根因诊断: 1) 生成 latent 分布 vs 真实 2) 去噪还原 3) decode 概率分布 4) CFG 影响。
用法: python train/diag_gen.py [--diff_ckpt ...] [--vae_ckpt ...]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import numpy as np
import torch

from vae import AutoencoderKL
from diffusion import UNet, DDPM
from convertor import chart_to_array
from audio_align import align_mel

LATENT_SCALE = [0.2732, 0.7144, 0.3273, 0.2301, 0.2052, 0.2667, 0.2719, 0.3148,
                0.2903, 0.4250, 0.6842, 0.3241, 0.4672, 0.5662, 0.2370, 0.4088]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff_ckpt", default="checkpoints/diffusion/last.ckpt")
    ap.add_argument("--vae_ckpt", default="checkpoints/vae/best.ckpt")
    ap.add_argument("--jsonl", default="data/dataset/slide_clean.jsonl")
    ap.add_argument("--audio_meta", default="data/audio/meta.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    vae = AutoencoderKL().to(args.device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=args.device)["model"])
    vae.eval()
    unet = UNet(with_audio=True, with_lv=True).to(args.device)
    ddpm = DDPM(unet).to(args.device)
    ddpm.load_state_dict(torch.load(args.diff_ckpt, map_location=args.device)["model"])
    ddpm.eval()
    scale = torch.tensor(LATENT_SCALE, device=args.device).view(16, 1, 1)

    mel = np.load("data/audio/10072_mel.npy")
    # 修复: ctx 从窗口起点 (10 拍) 对齐, 与 z_real 窗口一致 (此前错位 10 拍导致假象)
    ctx = torch.from_numpy(align_mel(mel, [[0.0, 87.0]], n_rows=132, latent_rows_per_beat=0.5,
                                     t_start=10.0)[:128])[None].to(args.device)

    print("=== 1) 生成 latent per-channel std (归一化后应≈1) ===")
    zs = []
    with torch.no_grad():
        for s in range(args.n):
            z = ddpm.sample((1, 16, 128, 32), lv=torch.tensor([12], device=args.device),
                            audio_ctx=ctx, steps=100, cfg_scale=1.5, device=args.device)
            zs.append(z[0].cpu().numpy())
    zs = np.stack(zs)
    ch_std = zs.std(axis=(0, 2, 3))
    print("  per-channel std (归一化后):", np.round(ch_std, 2).tolist())
    print("  理想: 全部≈1.0; 偏差大说明生成分布仍不匹配")

    print("\n=== 2) 去噪还原 (真实 latent, t=100/300/600/900) ===")
    recs = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
    r = next(x for x in recs if x["song_id"] == 10072)
    arr = chart_to_array(r["notes"], r["length_beat"], t_start=10.0, t_end=74.0).astype(np.float32)[:, :1024, :]
    with torch.no_grad():
        z_real = vae.encode(torch.from_numpy(arr)[None].to(args.device))[0] / scale
    for t_val in [100, 300, 600, 900]:
        torch.manual_seed(0)
        t = torch.tensor([t_val], device=args.device)
        noise = torch.randn_like(z_real)
        xt = ddpm.q_sample(z_real, t, noise)
        with torch.no_grad():
            eps_pred = unet(xt, t, torch.tensor([12], device=args.device), ctx)
        # eps MAE 直接测 (x0 还原误差被 1/sqrt(a) 放大数十倍, 有误导性)
        eps_mae = float((eps_pred - noise).abs().mean())
        print(f"  t={t_val}: eps MAE = {eps_mae:.4f} (噪声单位, <0.3 算好)")

    print("\n=== 3) decode 概率分布 (阈值 0.5 二值化影响) ===")
    with torch.no_grad():
        rec = torch.sigmoid(vae.decode(torch.from_numpy(zs[0:1]).to(args.device) * scale))[0].cpu().numpy()
    for c, name in [(0, "tap_mask"), (2, "drag_mask"), (4, "slide_mask")]:
        v = rec[c]
        frac_hi = (v > 0.7).mean() * 100
        frac_mid = ((v > 0.3) & (v < 0.7)).mean() * 100
        frac_lo = (v < 0.3).mean() * 100
        print(f"  {name}: >0.7 占 {frac_hi:.2f}% | 0.3~0.7 模糊 {frac_mid:.2f}% | <0.3 占 {frac_lo:.2f}%")
    print("  模糊像素占比高 -> 阈值硬切导致二值化极端模式")

    print("\n=== 4) CFG 影响: cfg=0 vs 1.5 生成差异 ===")
    torch.manual_seed(0)
    with torch.no_grad():
        z0 = ddpm.sample((1, 16, 128, 32), lv=torch.tensor([12], device=args.device),
                         audio_ctx=ctx, steps=100, cfg_scale=0.0, device=args.device)
    torch.manual_seed(0)
    with torch.no_grad():
        z15 = ddpm.sample((1, 16, 128, 32), lv=torch.tensor([12], device=args.device),
                          audio_ctx=ctx, steps=100, cfg_scale=1.5, device=args.device)
    diff = float((z0 - z15).abs().mean())
    print(f"  cfg=0 vs cfg=1.5 差异: {diff:.4f} (归一化后 latent; 差异大=CFG有效, 接近0=CFG无效)")


if __name__ == "__main__":
    main()
