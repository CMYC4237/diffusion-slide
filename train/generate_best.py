# -*- coding: utf-8 -*-
"""批量生成 + 自动选优: 多 seed 采样, 按目标密度/类型分布打分, 输出最接近真实谱面的一张。
用法: python train/generate_best.py --song 10072 --lv 12 --tries 12 [--cfg 1.5]
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
from convertor import array_to_chart
from audio_align import align_mel
from generate import to_malody_notes, make_mc, t_to_beat

# 真实谱面 lv -> 密度中位数 (note/拍)
LV_DENSITY = {
    1: 0.33, 2: 0.36, 3: 0.51, 4: 0.59, 5: 0.91, 6: 1.06, 7: 1.08,
    8: 1.25, 9: 1.18, 10: 1.27, 11: 1.24, 12: 1.45, 13: 1.66, 14: 1.72,
    15: 1.75, 16: 2.02, 17: 2.18, 18: 2.27, 19: 2.47, 20: 2.40, 21: 2.47,
    22: 2.54, 23: 2.74, 24: 2.96, 25: 3.18,
}
TARGET_TYPES = {0: 0.60, 1: 0.25, 2: 0.15}  # tap/drag/slide 真实占比


def score_notes(notes, lv, length_beat):
    """分数越低越好: 密度对数误差 + 类型分布误差"""
    density = len(notes) / length_beat
    target = LV_DENSITY.get(lv, 1.5)
    s = abs(np.log(density + 0.05) - np.log(target))
    from collections import Counter
    tc = Counter(n["type"] for n in notes)
    for typ, want in TARGET_TYPES.items():
        got = tc.get(typ, 0) / max(1, len(notes))
        s += 1.5 * abs(got - want)
    # 惩罚爆炸/空
    if len(notes) > 1500 or len(notes) < 5:
        s += 2.0
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff_ckpt", default="checkpoints/diffusion/last.ckpt")
    ap.add_argument("--vae_ckpt", default="checkpoints/vae/best.ckpt")
    ap.add_argument("--song", type=int, default=10072)
    ap.add_argument("--lv", type=int, default=12)
    ap.add_argument("--tries", type=int, default=12)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--jsonl", default="data/dataset/slide_clean.jsonl")
    ap.add_argument("--audio_meta", default="data/audio/meta.json")
    ap.add_argument("--out", default="output")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rows", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    recs = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
    ref = next(r for r in recs if r["song_id"] == args.song)
    audio_meta = json.load(open(args.audio_meta, encoding="utf-8"))
    mel = np.load(audio_meta[str(args.song)]["mel"])
    bpms = [[b[0], b[1]] for b in ref["bpms"]] if ref.get("bpms") else [[0.0, ref["bpm"]]]

    vae = AutoencoderKL().to(args.device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=args.device)["model"])
    vae.eval()
    unet = UNet(with_audio=True, with_lv=True).to(args.device)
    ddpm = DDPM(unet).to(args.device)
    ddpm.load_state_dict(torch.load(args.diff_ckpt, map_location=args.device)["model"])
    ddpm.eval()

    ctx = torch.from_numpy(align_mel(mel, bpms, n_rows=args.rows + args.rows // 4 + 4, latent_rows_per_beat=0.5)[:args.rows + args.rows // 4])[None].to(args.device)
    lv_t = torch.tensor([args.lv], device=args.device)

    gen_rows = args.rows + args.rows // 4
    crop = (gen_rows - args.rows) // 2
    best = None
    print(f"批量生成 {args.tries} 张 (song={args.song} lv={args.lv} cfg={args.cfg} 目标密度={LV_DENSITY.get(args.lv, '?')} note/拍, 生成{gen_rows}行裁{crop}行):")
    for seed in range(args.tries):
        torch.manual_seed(seed)
        with torch.no_grad():
            z = ddpm.sample((1, 16, gen_rows, 32), lv=lv_t, audio_ctx=ctx,
                            steps=args.steps, cfg_scale=args.cfg, device=args.device)
            z = z[:, :, crop:crop + args.rows, :]
            z = z * torch.tensor([0.2732, 0.7144, 0.3273, 0.2301, 0.2052, 0.2667, 0.2719, 0.3148, 0.2903, 0.4250, 0.6842, 0.3241, 0.4672, 0.5662, 0.2370, 0.4088], device=z.device).view(16, 1, 1)
            rec = torch.sigmoid(vae.decode(z))[0].cpu().numpy()
        notes = array_to_chart(rec)
        for n in notes:
            n["t"] = round(n["t"], 4)
        from collections import Counter
        tc = Counter(n["type"] for n in notes)
        sc = score_notes(notes, args.lv, args.rows * 0.5)
        mark = " *" if best is None or sc < best[0] else ""
        print(f"  seed={seed}: {len(notes):4d} note (tap={tc[0]} drag={tc[1]} slide={tc[2]}) "
              f"密度={len(notes)/(args.rows*0.5):.2f}/拍 分={sc:.2f}{mark}")
        if best is None or sc < best[0]:
            best = (sc, seed, notes)

    sc, seed, notes = best
    malody = to_malody_notes(notes)
    audio_path = audio_meta.get(str(args.song), {}).get("audio")
    audio_name = os.path.basename(audio_path) if audio_path else None
    mc = make_mc(malody, args.song, title=ref.get("version", "AI"),
                 version=f"AI Lv.{args.lv}", bpm=bpms[0][1], audio_name=audio_name)
    if audio_path and os.path.exists(audio_path):
        from generate import save_mcz
        out_path = os.path.join(args.out, f"ai_best_lv{args.lv}_song{args.song}_seed{seed}.mcz")
        save_mcz(mc, audio_path, out_path)
    else:
        out_path = os.path.join(args.out, f"ai_best_lv{args.lv}_song{args.song}_seed{seed}.mc")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(mc, f, ensure_ascii=False, indent=1)
    print(f"\n最佳: seed={seed} 分={sc:.2f} -> {out_path} (含音频: {bool(audio_path)})")


if __name__ == "__main__":
    main()
