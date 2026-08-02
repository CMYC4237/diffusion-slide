# -*- coding: utf-8 -*-
"""VAE 训练数据模块: 谱面窗口采样 + 渲染 + 增强。"""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from convertor import chart_to_array

FPS = 16
WINDOW_FRAMES = 1024   # 64 拍窗口
X_SIZE = 256
N_CH = 10


class SlideWindowDataset(Dataset):
    """每个样本 = 一个随机窗口的谱面渲染 (9, WINDOW_FRAMES, 256)。"""

    def __init__(self, jsonl_path, window_frames=WINDOW_FRAMES, mirror_p=0.5,
                 seed=0, cache_in_mem=True):
        with open(jsonl_path, encoding="utf-8") as f:
            self.recs = [json.loads(l) for l in f]
        self.window_frames = window_frames
        self.mirror_p = mirror_p
        self.rng = random.Random(seed)
        # 每个谱面生成多个窗口槽 (按 256 帧步进), 每 epoch 随机偏移
        self.starts = []
        for i, r in enumerate(self.recs):
            t_max = r["length_beat"]
            for n in r["notes"]:
                if n.get("seg"):
                    t_max = max(t_max, n["t"] + n["seg"][-1]["dt"])
            h = int(np.ceil(t_max * FPS)) + 2
            max_start = h - window_frames
            if max_start < 0:
                max_start = 0
            n_slots = max(1, max_start // 256 + 1)
            for w in range(n_slots):
                self.starts.append((i, w))
        # 预渲染缓存 (可选)
        self.cache_in_mem = cache_in_mem
        self._cache = {}
        if cache_in_mem:
            print(f"预渲染缓存中 ({len(self.recs)} 谱面)...")
            for i in range(len(self.recs)):
                self._cache[i] = chart_to_array(self.recs[i]["notes"], self.recs[i]["length_beat"])
            print("缓存完成")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i, w = self.starts[idx]
        r = self.recs[i]
        H_full = int(np.ceil(self._chart_len(r) * FPS)) + 2
        if H_full <= self.window_frames:
            s = 0
        else:
            s = min(w * 256 + self.rng.randint(0, 256), H_full - self.window_frames)
        t_start = s / FPS
        t_end = t_start + self.window_frames / FPS
        if self.cache_in_mem:
            arr = self._cache[i]
            window = arr[:, s:s + self.window_frames, :].astype(np.float32)
        else:
            window = chart_to_array(r["notes"], r["length_beat"], t_start=t_start, t_end=t_end).astype(np.float32)
        # 统一尺寸: pad/crop 到 window_frames (VAE 需要 8 的倍数)
        if window.shape[1] != self.window_frames:
            if window.shape[1] > self.window_frames:
                window = window[:, :self.window_frames, :]
            else:
                pad = np.zeros((window.shape[0], self.window_frames - window.shape[1], window.shape[2]), dtype=np.float32)
                window = np.concatenate([window, pad], axis=1)
        # 镜像增强
        if self.rng.random() < self.mirror_p:
            window = np.flip(window, axis=2).copy()
        x = torch.from_numpy(window)
        meta = {
            "song_id": r["song_id"],
            "lv": r["lv"] if r["lv"] is not None else -1,
            "bpm": r["bpm"],
            "window_start_beat": t_start,
            "mirrored": bool(self.rng.random() < 0),  # 占位(实际上面已翻转)
        }
        return x, meta

    @staticmethod
    def _chart_len(r):
        t_max = r["length_beat"]
        for n in r["notes"]:
            if n.get("seg"):
                t_max = max(t_max, n["t"] + n["seg"][-1]["dt"])
        return t_max


if __name__ == "__main__":
    ds = SlideWindowDataset("data/dataset/slide_clean.jsonl", cache_in_mem=False)
    print(f"窗口样本数: {len(ds)}")
    x, meta = ds[0]
    print("x shape:", x.shape, "meta:", {k: meta[k] for k in meta if k != 'mirrored'})
    # 通道统计
    for c in range(N_CH):
        vals = x[c]
        print(f"  通道{c}: 非零比例 {100*(vals>0).float().mean().item():.2f}%, 值域 [{vals.min():.3f}, {vals.max():.3f}]")
