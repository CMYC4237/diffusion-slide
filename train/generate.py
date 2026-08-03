# -*- coding: utf-8 -*-
"""生成验证: 难度 + 音乐 -> 谱面 json (Malody .mc 格式, mode 7)。

用法:
  python train/generate.py --diff_ckpt checkpoints/diffusion/last.ckpt \
      --vae_ckpt checkpoints/vae/best.ckpt --song 10072 --lv 12
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
from convertor import array_to_chart, FPS, W_MAX
from audio_align import align_mel
LATENT_SCALE = 0.632


def t_to_beat(t, denom=16):
    """浮点拍 -> Malody beat [a, b, c] (a + b/c 拍), 量化到 denom"""
    total = int(round(t * denom))
    a, b = divmod(total, denom)
    if a < 0:
        return [0, 0, 1]
    return [a, b, denom]


def to_malody_notes(notes):
    """规范化 note 列表 -> Malody mode7 note 列表"""
    out = []
    for n in notes:
        rec = {"beat": t_to_beat(n["t"]), "x": int(n["x"]), "w": int(n["w"])}
        if n["type"] == 1:
            rec["type"] = 4
        elif n["type"] == 2:
            seg = [{"beat": t_to_beat(s["dt"]), "x": int(s["dx"])} for s in n.get("seg", [])]
            if seg:
                rec["seg"] = seg
        out.append(rec)
    out.sort(key=lambda r: (r["beat"][0] + r["beat"][1] / r["beat"][2], r["x"]))
    return out


def make_mc(chart_notes, song_id, title="AI", artist="AI", version="AI Diff", bpm=120.0, bpms=None,
            audio_name=None):
    """生成 Malody mode7 .mc 内容。audio_name 非空时附加 sound 引用 note (与真实谱面一致)。"""
    notes = list(chart_notes)
    if audio_name:
        notes = notes + [{"beat": [0, 0, 1], "sound": audio_name, "vol": 100, "offset": 0, "type": 1}]
    return {
        "meta": {
            "$ver": 0, "creator": "DiffusionSlide-AI", "background": "",
            "version": version, "preview": 0, "id": 0, "mode": 7,
            "song": {"title": title, "artist": artist, "id": song_id},
            "mode_ext": {},
        },
        "time": [{"beat": [0, 0, 1], "bpm": bpm}],
        "effect": [],
        "note": notes,
        "extra": {"test": {"divide": 4, "speed": 100, "save": 0, "lock": 0, "edit_mode": 0}},
    }


def save_mcz(mc_dict, audio_path, out_path):
    """打包 Malody .mcz (zip): 0/chart.mc + 0/audio.<ext>"""
    import zipfile
    audio_name = os.path.basename(audio_path)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("0/chart.mc", json.dumps(mc_dict, ensure_ascii=False, indent=1))
        z.write(audio_path, f"0/{audio_name}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff_ckpt", default="checkpoints/diffusion_smoke/last.ckpt")
    ap.add_argument("--vae_ckpt", default="checkpoints/vae_smoke/best.ckpt")
    ap.add_argument("--song", type=int, default=10072, help="参考谱面的 song_id (用其 mel 音频)")
    ap.add_argument("--lv", type=int, default=12, help="目标难度 1~25")
    ap.add_argument("--jsonl", default="data/dataset/slide_clean.jsonl")
    ap.add_argument("--audio_meta", default="data/audio/meta.json")
    ap.add_argument("--out", default="output")
    ap.add_argument("--steps", type=int, default=100, help="DDIM 采样步数")
    ap.add_argument("--cfg", type=float, default=0.0, help="classifier-free guidance 强度")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rows", type=int, default=128, help="生成窗口 latent 行数 (默认128=64拍)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 参考谱面 (取 mel/bpm)
    recs = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
    ref = next(r for r in recs if r["song_id"] == args.song)
    with open(args.audio_meta, encoding="utf-8") as f:
        audio_meta = json.load(f)
    mel = np.load(audio_meta[str(args.song)]["mel"])
    bpms = [[b[0], b[1]] for b in ref["bpms"]] if ref.get("bpms") else [[0.0, ref["bpm"]]]

    # 加载模型
    vae = AutoencoderKL().to(args.device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=args.device)["model"])
    vae.eval()
    unet = UNet(with_audio=True, with_lv=True).to(args.device)
    ddpm = DDPM(unet).to(args.device)
    ddpm.load_state_dict(torch.load(args.diff_ckpt, map_location=args.device)["model"])
    ddpm.eval()

    # 条件
    rows = args.rows
    ctx = align_mel(mel, bpms, n_rows=rows + 4, latent_rows_per_beat=0.5)
    ctx = ctx[:rows]
    ctx = torch.from_numpy(ctx)[None].to(args.device)
    lv = torch.tensor([args.lv], device=args.device)

    # 采样 (长窗口分块: 128 行 = 64 拍)
    n_chunks = 1
    latents = []
    with torch.no_grad():
        for c in range(n_chunks):
            z = ddpm.sample((1, 16, rows, 32), lv=lv, audio_ctx=ctx,
                            steps=args.steps, cfg_scale=args.cfg, device=args.device)
            latents.append(z)
    z = torch.cat(latents, dim=2)
    z = z * LATENT_SCALE  # latent 归一化还原

    # 解码 -> 通道图 -> 谱面
    with torch.no_grad():
        rec = torch.sigmoid(vae.decode(z))[0].cpu().numpy()
    notes = array_to_chart(rec)
    malody = to_malody_notes(notes)

    # 音频 (打包进 .mcz)
    audio_path = audio_meta.get(str(args.song), {}).get("audio")
    audio_name = os.path.basename(audio_path) if audio_path else None

    mc = make_mc(malody, args.song, title=ref.get("version", "AI"),
                 version=f"AI Lv.{args.lv}", bpm=bpms[0][1], audio_name=audio_name)
    if audio_path and os.path.exists(audio_path):
        out_path = os.path.join(args.out, f"ai_lv{args.lv}_song{args.song}.mcz")
        save_mcz(mc, audio_path, out_path)
    else:
        out_path = os.path.join(args.out, f"ai_lv{args.lv}_song{args.song}.mc")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(mc, f, ensure_ascii=False, indent=1)

    # 统计
    from collections import Counter
    tc = Counter(n["type"] for n in notes)
    slides = sum(1 for n in notes if n["type"] == 2)
    seg_total = sum(len(n.get("seg", [])) for n in notes)
    print(f"生成完成: {out_path} (含音频: {bool(audio_path)})")
    print(f"  note 总数 {len(notes)}, tap {tc[0]}, drag {tc[1]}, slide {slides} (seg 节点 {seg_total})")
    print(f"  时长 {rows*0.5:.0f} 拍, 平均密度 {len(notes)/(rows*0.5):.1f} note/拍")


if __name__ == "__main__":
    main()
