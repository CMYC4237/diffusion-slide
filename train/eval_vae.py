# -*- coding: utf-8 -*-
"""评估 VAE: 用训练好的 VAE 重建真实谱面窗口, 生成对比图 + 重建指标。"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vae import AutoencoderKL
from convertor import chart_to_array, array_to_chart, FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/vae/best.ckpt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--jsonl", default="data/dataset/slide_clean.jsonl")
    ap.add_argument("--out", default="data/vis/vae_recon")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = AutoencoderKL()
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    recs = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
    # 挑不同难度的谱面, 取一个 64 拍窗口
    picks = []
    for lv in ["5", "12", "20"]:
        for r in recs:
            if str(r["lv"]) == lv and r["length_beat"] > 90:
                picks.append((r, 10.0)); break
    picks = picks[:args.n]

    def render_panel(ax, arr, title):
        H = arr.shape[1]
        img = np.zeros((H, 256, 3))
        img[..., 0] += arr[2] * 0.9
        img[..., 1] += arr[4] * 0.9
        img[..., 2] += arr[0] * 0.9
        img += 0.05 * arr[7][..., None]
        start = arr[4] > 1.5
        img[start] = [1.0, 1.0, 0.2]
        ax.imshow(img, aspect="auto", origin="upper")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    with torch.no_grad():
        for k, (r, ts) in enumerate(picks):
            arr = chart_to_array(r["notes"], r["length_beat"], t_start=ts, t_end=ts + 64).astype(np.float32)
            if arr.shape[1] < 1024:
                pad = np.zeros((9, 1024 - arr.shape[1], 256), np.float32)
                arr = np.concatenate([arr, pad], 1)
            x = torch.from_numpy(arr)[None].to(args.device)
            z = model.encode(x)[0]
            rec = torch.sigmoid(model.decode(z))[0].cpu().numpy()
            # 反向还原谱面
            notes = array_to_chart(rec)

            fig, axes = plt.subplots(1, 2, figsize=(16, 11))
            render_panel(axes[0], arr, f"原始 (song={r['song_id']} '{r['version']}')")
            render_panel(axes[1], rec, f"VAE 重建 (latent {tuple(z.shape[1:])})")
            fig.suptitle(f"lv={r['lv']} notes={r['n_notes']}", fontsize=12)
            fig.tight_layout()
            out = os.path.join(args.out, f"recon_{k}_{r['song_id']}_lv{r['lv']}.png")
            fig.savefig(out, dpi=95)
            plt.close(fig)
            # 重建率 (容差)
            orig = {(n["t"], n["x"], n["type"]) for n in r["notes"] if ts <= n["t"] < ts + 64}
            rset = {(n["t"], n["x"], n["type"]) for n in notes}
            hit = sum(1 for o in orig if any(abs(m[0]-o[0]) <= 0.06 and m[1] == o[1] and m[2] == o[2] for m in rset))
            print(f"[{k}] song={r['song_id']} lv={r['lv']}: 窗口内note {len(orig)}, VAE重建后匹配 {hit} ({100*hit/max(1,len(orig)):.1f}%)  -> {out}")


if __name__ == "__main__":
    main()
