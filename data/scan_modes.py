# -*- coding: utf-8 -*-
"""扫描 Slide.zip 里所有 mcz，统计每个 mc 的模式和字段特征（只读 .mc 文件，不解音频）。"""
import io
import json
import sys
import zipfile
from collections import Counter

ZIP_PATH = sys.argv[1] if len(sys.argv) > 1 else "Sort By mode/Slide.zip"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None  # 只处理前 N 个 mcz

def main():
    mode_counter = Counter()
    field_counter = Counter()
    n_mcz = 0
    n_mc = 0
    samples = {}  # mode -> (song_id, mc_name, 是否含x/w/seg, note样例)
    with zipfile.ZipFile(ZIP_PATH) as outer:
        names = [n for n in outer.namelist() if n.endswith(".mcz")]
        if LIMIT:
            names = names[:LIMIT]
        for name in names:
            n_mcz += 1
            data = outer.read(name)
            try:
                inner = zipfile.ZipFile(io.BytesIO(data))
            except Exception as e:
                print(f"!! {name}: 无法打开内层zip: {e}")
                continue
            for mc_name in inner.namelist():
                if not mc_name.endswith(".mc"):
                    continue
                try:
                    chart = json.loads(inner.read(mc_name))
                except Exception as e:
                    print(f"!! {name}/{mc_name}: JSON解析失败: {e}")
                    continue
                n_mc += 1
                meta = chart.get("meta", {})
                mode = meta.get("mode")
                mode_counter[mode] += 1
                notes = chart.get("note", [])
                has_xw = any("x" in n and "w" in n for n in notes if isinstance(n, dict))
                has_seg = any("seg" in n for n in notes if isinstance(n, dict))
                if has_xw or has_seg:
                    key = (mode, has_xw, has_seg)
                    field_counter[key] += 1
                    if key not in samples:
                        sample = next(n for n in notes if isinstance(n, dict) and (("x" in n and "w" in n) or "seg" in n))
                        samples[key] = (name, mc_name, sample)
            if LIMIT and n_mcz >= LIMIT:
                break
    print(f"共 {n_mcz} 个 mcz, {n_mc} 个 mc 谱面")
    print("\n== mode 分布 ==")
    for m, c in mode_counter.most_common():
        print(f"  mode {m}: {c}")
    print("\n== 含 x/w 或 seg 的 (mode, has_xw, has_seg): 数量 ==")
    for (m, xw, seg), c in field_counter.most_common():
        print(f"  mode {m}, x/w: {xw}, seg: {seg}: {c}")
    print("\n== 样例 ==")
    for (m, xw, seg), (name, mc, sample) in samples.items():
        print(f"  [{name}/{mc}] mode={m}")
        print(f"    {json.dumps(sample, ensure_ascii=False)}")

if __name__ == "__main__":
    main()
