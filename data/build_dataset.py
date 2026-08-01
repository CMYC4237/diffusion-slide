# -*- coding: utf-8 -*-
"""
构建干净的 Slide (mode 7) 训练数据集：
- 遍历 Slide.zip 全部 248 个 mcz
- 提取所有 mode 7 谱面
- 筛掉任何 note 缺 w 的谱面（共 45 个，约 5.8%）
- 规范化为 JSONL：每行一个谱面
输出: data/dataset/slide_clean.jsonl, data/dataset/stats.json
"""
import io
import json
import os
import re
import zipfile
from collections import Counter

ZIP_PATH = "Sort By mode/Slide.zip"
OUT_DIR = "data/dataset"
OUT_JSONL = os.path.join(OUT_DIR, "slide_clean.jsonl")
OUT_STATS = os.path.join(OUT_DIR, "stats.json")

# note 类型归一化: 0=tap, 1=drag(type=4), 2=slide(带seg)
def normalize_type(n):
    if n.get("seg"):
        return 2
    if n.get("type") == 4:
        return 1
    return 0

def beat_float(b):
    return b[0] + b[1] / b[2]

def parse_lv(version):
    if not isinstance(version, str):
        return None
    m = re.search(r"Lv\.?\s*(\d+)", version)
    return int(m.group(1)) if m else None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    kept = []
    dropped = []          # (mcz, mc, version, 缺w的note数)
    n_mode7 = 0
    mode_ext_vals = Counter()
    per_song = {}         # song_id -> 保留谱面数（同一个歌多个难度）
    audio_info = {}       # mcz -> (音频文件名, 大小)

    with zipfile.ZipFile(ZIP_PATH) as outer:
        for name in outer.namelist():
            if not name.endswith(".mcz"):
                continue
            inner = zipfile.ZipFile(io.BytesIO(outer.read(name)))
            # 记录音频
            oggs = [n for n in inner.namelist() if n.lower().endswith((".ogg", ".mp3", ".m4a", ".wav"))]
            audio_info[name] = [(n, inner.getinfo(n).file_size) for n in oggs]
            for mc_name in inner.namelist():
                if not mc_name.endswith(".mc"):
                    continue
                chart = json.loads(inner.read(mc_name))
                meta = chart.get("meta", {})
                if meta.get("mode") != 7:
                    continue
                n_mode7 += 1
                notes = [n for n in chart.get("note", []) if isinstance(n, dict) and "x" in n]
                # 清洗条件：所有 note 必须有 w
                bad = [n for n in notes if "w" not in n]
                if bad:
                    dropped.append((name, mc_name, meta.get("version", ""), len(bad)))
                    continue
                # 检查非空 mode_ext
                if meta.get("mode_ext"):
                    mode_ext_vals[str(meta.get("mode_ext"))] += 1

                # 规范化
                bpms = []
                for t in chart.get("time", []):
                    if isinstance(t, dict) and "bpm" in t and "beat" in t:
                        bpms.append([beat_float(t["beat"]), t["bpm"]])
                if not bpms:
                    dropped.append((name, mc_name, meta.get("version", ""), -1))
                    continue
                bpm0 = bpms[0][1]

                norm_notes = []
                for n in notes:
                    rec = {
                        "t": round(beat_float(n["beat"]), 4),   # 拍(浮点)
                        "x": n["x"],
                        "w": n["w"],
                        "type": normalize_type(n),
                    }
                    seg = n.get("seg")
                    if seg:
                        rec["seg"] = [
                            {
                                "dt": round(beat_float(s["beat"]), 4),  # 相对起点的拍
                                "dx": s.get("x", 0),                     # 相对位移
                            }
                            for s in seg if isinstance(s, dict)
                        ]
                    norm_notes.append(rec)
                norm_notes.sort(key=lambda r: (r["t"], r["x"]))
                max_t = norm_notes[-1]["t"] if norm_notes else 0.0

                rec = {
                    "song_id": meta.get("song", {}).get("id"),
                    "mcz": name,
                    "mc": mc_name,
                    "version": meta.get("version", ""),
                    "lv": parse_lv(meta.get("version", "")),
                    "creator": meta.get("creator", ""),
                    "bpm": bpm0,
                    "bpms": bpms,
                    "length_beat": round(max_t, 4),
                    "n_notes": len(norm_notes),
                    "notes": norm_notes,
                }
                kept.append(rec)
                sid = rec["song_id"]
                per_song[sid] = per_song.get(sid, 0) + 1

    # 写 JSONL
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "mode7_charts": n_mode7,
        "kept": len(kept),
        "dropped_charts": len(dropped),
        "dropped_pct": round(len(dropped) / n_mode7 * 100, 2),
        "unique_songs": len(per_song),
        "notes_total": sum(r["n_notes"] for r in kept),
        "songs_multiple_charts": sum(1 for v in per_song.values() if v > 1),
        "lv_dist": Counter(str(r["lv"]) for r in kept),
        "mode_ext_nonempty": dict(mode_ext_vals),
        "drop_reasons": {"缺w": sum(1 for d in dropped if d[3] > 0), "无bpm": sum(1 for d in dropped if d[3] < 0)},
    }
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    print("=== 构建完成 ===")
    print(f"mode 7 谱面总数: {n_mode7}")
    print(f"保留(干净): {len(kept)}  筛掉: {len(dropped)} ({len(dropped)/n_mode7*100:.2f}%)")
    print(f"唯一歌曲数: {len(per_song)}")
    print(f"note 总数: {stats['notes_total']}")
    print(f"一歌多谱面的歌数: {stats['songs_multiple_charts']}")
    print(f"Lv 分布: {dict(sorted(stats['lv_dist'].items(), key=lambda x: int(x[0]) if x[0] != 'None' else 99))}")
    print(f"非空 mode_ext: {stats['mode_ext_nonempty']}")
    print(f"输出: {OUT_JSONL} ({os.path.getsize(OUT_JSONL)/1e6:.1f} MB)")
    print(f"输出: {OUT_STATS}")

if __name__ == "__main__":
    main()
