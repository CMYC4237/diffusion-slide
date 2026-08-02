# -*- coding: utf-8 -*-
"""VAE 重建细粒度诊断: 精确/容差匹配率, 按类型分解, 宽度误差, 滑条还原度, 假阳性。
用法: python train/eval_vae_detail.py --ckpt checkpoints/vae/best.ckpt
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
from convertor import chart_to_array, array_to_chart

WINDOW = 64.0  # 拍


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/vae/best.ckpt")
    ap.add_argument("--jsonl", default="data/dataset/slide_clean.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_charts", type=int, default=30)
    ap.add_argument("--ts", type=float, default=10.0)
    args = ap.parse_args()

    model = AutoencoderKL()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.to(args.device).eval()

    recs = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
    picks = [r for r in recs if r["length_beat"] > 90][:args.n_charts]
    ts = args.ts

    stats = {"n": {0: 0, 1: 0, 2: 0}, "exact": {0: 0, 1: 0, 2: 0},
             "tol": {0: 0, 1: 0, 2: 0}, "fp": 0, "tp_total": 0,
             "w_err": [], "seg_end_err": []}

    def tol_match(m, o):
        return m["type"] == o["type"] and abs(m["t"] - o["t"]) <= 0.1 and abs(m["x"] - o["x"]) <= 2

    with torch.no_grad():
        for r in picks:
            arr = chart_to_array(r["notes"], r["length_beat"], t_start=ts, t_end=ts + WINDOW).astype(np.float32)
            if arr.shape[1] < 1024:
                pad = np.zeros((9, 1024 - arr.shape[1], 256), np.float32)
                arr = np.concatenate([arr, pad], 1)
            x = torch.from_numpy(arr)[None].to(args.device)
            z = model.encode(x)[0]
            rec = torch.sigmoid(model.decode(z))[0].cpu().numpy()
            notes = array_to_chart(rec)
            for n in notes:
                n["t"] += ts
            orig = [n for n in r["notes"] if ts <= n["t"] < ts + WINDOW]

            # 精确/容差匹配 (贪心, 每个重建 note 至多匹配一个原始 note)
            used = set()
            for o in orig:
                stats["n"][o["type"]] += 1
                best = None
                for m in notes:
                    if id(m) in used:
                        continue
                    if m["type"] != o["type"]:
                        continue
                    dt = abs(m["t"] - o["t"]); dx = abs(m["x"] - o["x"])
                    if dt <= 0.06 and dx == 0:
                        score = (dt, dx)
                    elif dt <= 0.1 and dx <= 2:
                        score = (dt + 1, dx)  # 容差命中
                    else:
                        continue
                    if best is None or score < best[0]:
                        best = (score, m)
                if best:
                    used.add(id(best[1]))
                    exact = best[0][1] == 0 and best[0][0] <= 0.06
                    if exact:
                        stats["exact"][o["type"]] += 1
                    stats["tol"][o["type"]] += 1
                    stats["tp_total"] += 1
                    stats["w_err"].append(abs(best[1]["w"] - o["w"]))
                    if o["type"] == 2 and o.get("seg") and best[1].get("seg"):
                        oe = o["x"] + o["seg"][-1]["dx"]
                        me = best[1]["x"] + best[1]["seg"][-1]["dx"]
                        stats["seg_end_err"].append(abs(me - oe))
            stats["fp"] += len(notes) - len(used)

    print(f"评估 {len(picks)} 个谱面, 窗口 {ts}~{ts+WINDOW} 拍\n")
    for typ, name in [(0, "tap"), (1, "drag"), (2, "slide")]:
        n = stats["n"][typ]
        if n == 0:
            continue
        print(f"  {name:<6} 总 {n:<6} 精确 {stats['exact'][typ]} ({100*stats['exact'][typ]/n:.1f}%)  "
              f"容差(x±2,t±0.1) {stats['tol'][typ]} ({100*stats['tol'][typ]/n:.1f}%)")
    tot = sum(stats["n"].values())
    print(f"\n  合计    总 {tot:<6} 精确 {sum(stats['exact'].values())} ({100*sum(stats['exact'].values())/tot:.1f}%)  "
          f"容差 {sum(stats['tol'].values())} ({100*sum(stats['tol'].values())/tot:.1f}%)")
    print(f"  假阳性(重建多出): {stats['fp']} ({100*stats['fp']/tot:.1f}%)")
    if stats["w_err"]:
        we = np.array(stats["w_err"])
        print(f"  宽度误差: 中位 {np.median(we):.1f}px, ≤4px 占比 {(we<=4).mean()*100:.1f}%, ≤8px {(we<=8).mean()*100:.1f}%")
    if stats["seg_end_err"]:
        se = np.array(stats["seg_end_err"])
        print(f"  滑条终点误差: 中位 {np.median(se):.1f}px, ≤4px 占比 {(se<=4).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
