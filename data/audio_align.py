# -*- coding: utf-8 -*-
"""音频特征按拍对齐: mel 谱 -> 与谱面 latent 时间轴 (每 latent 行) 对齐的特征序列。

latent 时间轴: 谱面窗口 1024 帧 / 8 = 128 行, 每行 = 8 谱面帧 = 0.5 拍 (16 帧/拍)。
"""
import json
import os

import numpy as np

SR = 22050
HOP = 256


def beat_to_seconds(bpms, beat):
    """bpms: [(beat, bpm)] 升序; 返回 beat 处的秒数 (线性累积)"""
    if not bpms:
        return 0.0
    if beat <= bpms[0][0]:
        return 0.0
    sec = 0.0
    for i, (b0, bpm0) in enumerate(bpms):
        b1 = bpms[i + 1][0] if i + 1 < len(bpms) else float("inf")
        if beat <= b1:
            return sec + (beat - b0) * 60.0 / bpm0
        sec += (b1 - b0) * 60.0 / bpm0
    return sec


def align_mel(mel, bpms, n_rows, latent_rows_per_beat=0.5, sr=SR, hop=HOP):
    """把全曲 mel (T, n_mels) 对齐到 latent 行。
    第 i 行的谱面拍 = i * latent_rows_per_beat (拍), 取对应秒的 mel 帧 (线性插值)。
    返回 (n_rows, n_mels) float32
    """
    n_mels = mel.shape[1]
    out = np.zeros((n_rows, n_mels), dtype=np.float32)
    for i in range(n_rows):
        beat = i * latent_rows_per_beat
        sec = beat_to_seconds(bpms, beat)
        fpos = sec * sr / hop
        fi = int(np.floor(fpos))
        if fi >= mel.shape[0]:
            out[i] = mel[-1]
        elif fi < 0:
            out[i] = mel[0]
        else:
            frac = fpos - fi
            if fi + 1 < mel.shape[0]:
                out[i] = mel[fi] * (1 - frac) + mel[fi + 1] * frac
            else:
                out[i] = mel[fi]
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    with open("data/audio/meta.json", encoding="utf-8") as f:
        audio_meta = json.load(f)
    mel = np.load(audio_meta["10072"]["mel"])
    bpms = [[0.0, 87.0]]
    feat = align_mel(mel, bpms, n_rows=128)
    print("对齐特征:", feat.shape, "值域", float(feat.min()), float(feat.max()))
    # 验证: 第 0 行应该 ≈ mel 第 0 帧, 第 8 行(4拍@87bpm=2.76s) ≈ mel 帧 2.76*86.13
    print("行0 vs mel[0] 差:", float(np.abs(feat[0] - mel[0]).mean()))
    f4 = 4 * 60 / 87 * SR / HOP
    print(f"4拍秒数={4*60/87:.2f}s, mel帧={f4:.1f}, 行8差: {float(np.abs(feat[8] - mel[int(f4)]).mean()):.4f}")
