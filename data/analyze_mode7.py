# -*- coding: utf-8 -*-
"""全量统计分析 mode 7 (Slide) 谱面：字段、取值分布、结构特征。"""
import io
import json
import sys
import zipfile
from collections import Counter

ZIP_PATH = sys.argv[1] if len(sys.argv) > 1 else "Sort By mode/Slide.zip"

def beat_to_float(b):
    if not isinstance(b, list) or len(b) < 3:
        return None
    return b[0] + b[1] / b[2]

def main():
    type_counter = Counter()
    seg_len_counter = Counter()
    note_keys = Counter()
    seg_keys = Counter()
    denom_counter = Counter()
    x_vals = Counter()
    w_vals = Counter()
    n_charts = 0
    n_notes = 0
    n_seg = 0
    bar_begin_vals = Counter()
    bpm_vals = Counter()
    chart_stats = []  # (num_notes, 时长拍, bpm均值, version)
    multi_bpm = 0
    all_notes_have_w = True
    w_missing = 0

    with zipfile.ZipFile(ZIP_PATH) as outer:
        for name in outer.namelist():
            if not name.endswith(".mcz"):
                continue
            data = outer.read(name)
            try:
                inner = zipfile.ZipFile(io.BytesIO(data))
            except Exception:
                continue
            for mc_name in inner.namelist():
                if not mc_name.endswith(".mc"):
                    continue
                try:
                    chart = json.loads(inner.read(mc_name))
                except Exception:
                    continue
                if chart.get("meta", {}).get("mode") != 7:
                    continue
                n_charts += 1
                mext = chart.get("meta", {}).get("mode_ext", {})
                if isinstance(mext, dict):
                    bar_begin_vals[mext.get("bar_begin", "无")] += 1
                # time/bpm
                times = chart.get("time", [])
                if len(times) > 1:
                    multi_bpm += 1
                for t in times:
                    if isinstance(t, dict) and "bpm" in t:
                        bpm_vals[round(t["bpm"] * 2) / 2] += 1
                notes = [n for n in chart.get("note", []) if isinstance(n, dict) and "beat" in n and "x" in n]
                n_notes += len(notes)
                max_beat = 0.0
                for n in notes:
                    for k in n:
                        note_keys[k] += 1
                    bt = beat_to_float(n.get("beat"))
                    if bt:
                        max_beat = max(max_beat, bt)
                        denom_counter[bt % 1 and n["beat"][2] or n["beat"][2]] += 1
                    x_vals[n.get("x")] += 1
                    if "w" in n:
                        w_vals[n.get("w")] += 1
                    else:
                        w_missing += 1
                    t = n.get("type")
                    if t is not None:
                        type_counter[t] += 1
                    seg = n.get("seg")
                    if seg:
                        seg_len_counter[len(seg)] += 1
                        n_seg += len(seg)
                        for s in seg:
                            for k in s:
                                seg_keys[k] += 1
                # 时长(拍) = 最后一个note的beat
                bpm0 = times[0]["bpm"] if times and "bpm" in times[0] else 0
                chart_stats.append((len(notes), max_beat, bpm0, chart.get("meta", {}).get("version", "")))

    print(f"=== mode 7 谱面统计 ===")
    print(f"谱面数: {n_charts}, 总note数: {n_notes}, seg节点数: {n_seg}")
    print(f"多bpm段落谱面数: {multi_bpm}")
    print(f"\n-- note 字段 --")
    for k, c in note_keys.most_common():
        print(f"  {k}: {c}")
    print(f"\n-- seg 字段 --")
    for k, c in seg_keys.most_common():
        print(f"  {k}: {c}")
    print(f"\n-- type 取值 --")
    for k, c in type_counter.most_common():
        print(f"  type={k}: {c}")
    print(f"\n-- seg 节点数分布(前15) --")
    for k, c in seg_len_counter.most_common(15):
        print(f"  len={k}: {c}")
    print(f"\n-- x 取值(前20) --")
    for k, c in x_vals.most_common(20):
        print(f"  x={k}: {c}")
    print(f"\n-- w 取值(前20) --")
    for k, c in w_vals.most_common(20):
        print(f"  w={k}: {c}")
    print(f"\n-- beat分母取值 --")
    for k, c in denom_counter.most_common():
        print(f"  /{k}: {c}")
    print(f"\n-- bar_begin --")
    for k, c in bar_begin_vals.most_common():
        print(f"  {k}: {c}")
    print(f"\n-- bpm取值(前15) --")
    for k, c in bpm_vals.most_common(15):
        print(f"  {k}: {c}")
    # 谱面规模
    import statistics
    ns = [s[0] for s in chart_stats]
    bs = [s[1] for s in chart_stats]
    print(f"\n-- 谱面规模 --")
    print(f"  note数: min={min(ns)} med={statistics.median(ns)} max={max(ns)}")
    print(f"  时长(拍): min={min(bs):.1f} med={statistics.median(bs):.1f} max={max(bs):.1f}")
    print(f"\n-- 无w字段的note数: {w_missing}")
    # 打印一个最大的谱面版本分布
    print(f"\n-- 谱面规模 top10 --")
    for s in sorted(chart_stats, key=lambda x: -x[0])[:10]:
        print(f"  notes={s[0]}, len={s[1]:.1f}拍, bpm={s[2]}, version='{s[3]}'")

if __name__ == "__main__":
    main()
