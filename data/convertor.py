# -*- coding: utf-8 -*-
"""
无轨谱面 (Slide, mode 7) <-> 多通道图像 convertor。

通道布局 (10 通道, H x W = 时间帧 x 256px)：
  0. tap_mask    : tap 点中心像素 = 1
  1. tap_width   : tap 的 w/256
  2. drag_mask   : drag 点中心像素 = 1
  3. drag_width  : drag 的 w/256
  4. slide_mask  : 滑条中心线 (含起点) = 1
  5. slide_start : 滑条起点像素 = 1 (独立通道, 解决 sigmoid 无法表达值2的问题)
  6. slide_width : 滑条 w/256 (仅路径像素)
  7. overlap_cnt : 滑条路径重叠计数 (归一化 /8, 0~1)
  8. beat_line   : 每拍边界 = 1
  9. bar_line    : 每 4 拍小节边界 = 1

坐标约定：
  - x 轴 = 横向位置, 1px = 1 坐标单位, 像素列 0..255 (x 中心线四舍五入)
  - y 轴 = 时间轴, 帧率 FPS 帧/拍 (默认 16), 第 0 帧 = 第 0 拍
"""
import json
import numpy as np

FPS = 16          # 帧/拍
X_SIZE = 256      # 像素宽度
BEAT_PER_BAR = 4  # 每小节拍数
W_MAX = 256.0     # w 归一化分母
OVERLAP_NORM = 8.0  # 重叠计数归一化分母

# 通道索引
CH_TAP_MASK, CH_TAP_W, CH_DRAG_MASK, CH_DRAG_W = 0, 1, 2, 3
CH_SLIDE_MASK, CH_SLIDE_START, CH_SLIDE_W, CH_OVERLAP = 4, 5, 6, 7
CH_BEAT, CH_BAR = 8, 9
N_CH = 10

# ---------- 前向: 谱面 -> 通道数组 ----------

def chart_to_array(notes, length_beat, fps=FPS, x_size=X_SIZE, include_grid=True,
                   t_start=0.0, t_end=None):
    """
    notes: 规范化 note 列表 [{t,x,w,type,seg?}]
    t_start/t_end: 渲染的拍范围 (窗口模式, 可只渲染局部)
    返回 (C, H, W) float32 数组, H = ceil((t_end - t_start) * fps) + 2
    """
    if t_end is None:
        t_max = length_beat
        for n in notes:
            if n.get("seg"):
                t_max = max(t_max, n["t"] + n["seg"][-1]["dt"])
    else:
        t_max = t_end
    H = int(np.ceil((t_max - t_start) * fps)) + 2
    f_off = int(round(t_start * fps))  # 全局帧 -> 本地帧 偏移
    arr = np.zeros((N_CH, H, x_size), dtype=np.float32)
    if include_grid:
        # 拍线 + 小节线 (仅窗口内)
        b0 = max(0, int(np.ceil(t_start)))
        for b in range(b0, int(t_max) + 1):
            fg = int(round(b * fps))
            f = fg - f_off
            if 0 <= f < H:
                arr[CH_BEAT, f, :] = 1.0
                if b % BEAT_PER_BAR == 0:
                    arr[CH_BAR, f, :] = 1.0

    def put_frame(fg, x, mask_ch, w_ch, w):
        f = fg - f_off
        if 0 <= f < H and 0 <= x < x_size:
            arr[mask_ch, f, x] = 1.0
            arr[w_ch, f, x] = w / W_MAX

    slide_pixel_sets = []  # 每条滑条的 (本地帧,x) 像素集, 用于 overlap 统计

    for n in notes:
        t, x, w, typ = n["t"], n["x"], n["w"], n["type"]
        if typ == 0:  # tap
            put_frame(int(round(t * fps)), x, CH_TAP_MASK, CH_TAP_W, w)
        elif typ == 1:  # drag
            put_frame(int(round(t * fps)), x, CH_DRAG_MASK, CH_DRAG_W, w)
        elif typ == 2:  # slide
            seg = n.get("seg", [])
            pts = [(t, x)] + [(t + s["dt"], x + s["dx"]) for s in seg]
            px = set()
            # 逐段光栅化: 从节点帧到下一节点帧, 按帧号线性插值 (保证每帧恰一个像素, 连续)
            for k in range(len(pts) - 1):
                t0, x0 = pts[k]
                t1, x1 = pts[k + 1]
                f_node = int(round(t0 * fps))
                f_end = int(round(t1 * fps))
                span = max(f_end - f_node, 1)
                for fg in range(f_node, f_end + 1):
                    f = fg - f_off
                    if f < 0 or f >= H:
                        continue
                    a = (fg - f_node) / span
                    xi = int(round(x0 + (x1 - x0) * a))
                    if 0 <= xi < x_size:
                        px.add((f, xi))
            # 起点像素 (mask + start)
            fs = int(round(t * fps)) - f_off
            xs = int(round(x))
            if 0 <= fs < H and 0 <= xs < x_size:
                px.add((fs, xs))
            # 写入 mask/width 通道
            for f, xi in px:
                arr[CH_SLIDE_MASK, f, xi] = 1.0
                arr[CH_SLIDE_W, f, xi] = w / W_MAX
            # 起点通道
            if 0 <= fs < H and 0 <= xs < x_size:
                arr[CH_SLIDE_START, fs, xs] = 1.0
            slide_pixel_sets.append(px)
    # overlap 计数
    for px in slide_pixel_sets:
        for f, xi in px:
            arr[CH_OVERLAP, f, xi] = min(arr[CH_OVERLAP, f, xi] + 1.0 / OVERLAP_NORM, 1.0)
    return arr


# ---------- 反向: 通道数组 -> 谱面 ----------

def array_to_chart(arr, fps=FPS, x_size=X_SIZE):
    """
    从通道数组还原 note 列表。
    第一版: tap/drag 直接阈值提取; slide 用起点引导 + 8邻域追踪 (重叠区域跳过)。
    返回 notes 列表 (与原格式一致: {t,x,w,type,seg?})
    """
    H = arr.shape[1]
    notes = []
    th = 0.5

    # tap
    mask = arr[0] > th
    ys, xs = np.nonzero(mask)
    for f, x in zip(ys, xs):
        notes.append({"t": round(f / fps, 4), "x": int(x), "w": int(round(arr[1, f, x] * W_MAX)),
                      "type": 0})
    # drag
    mask = arr[2] > th
    ys, xs = np.nonzero(mask)
    for f, x in zip(ys, xs):
        notes.append({"t": round(f / fps, 4), "x": int(x), "w": int(round(arr[3, f, x] * W_MAX)),
                      "type": 1})

    # slide: 起点 = 独立起点通道 ∪ 路径段首 (前一帧无邻接路径的像素, 兼容生成谱面起点缺失)
    starts_mask = arr[CH_SLIDE_START] > th
    path_binary = arr[CH_SLIDE_MASK] > th
    for f in range(1, H):
        prev = path_binary[f - 1]
        cur_xs = np.nonzero(path_binary[f])[0]
        for x in cur_xs:
            lo, hi = max(0, x - 2), min(x_size, x + 3)
            if not np.any(prev[lo:hi]):
                starts_mask[f, x] = True
    sy, sx = np.nonzero(starts_mask)
    used = np.zeros((H, x_size), dtype=bool)
    # 预索引: 每帧的路径像素 x 列表
    frame_px = [np.nonzero(arr[CH_SLIDE_MASK, f, :] > th)[0] for f in range(H)]

    for f0, x0 in zip(sy, sx):
        if used[f0, x0]:
            continue
        # 追踪路径: 预测式贪心 (候选=该帧路径像素, 距离<=32, 允许跳1帧, 跳过重叠区)
        path = [(f0, x0)]
        used[f0, x0] = True
        f, x = f0, x0
        prev_x = x0
        while True:
            best = None
            for jump in (1, 2):
                nf = f + jump
                if nf >= H:
                    break
                for nx in frame_px[nf]:
                    if not (not used[nf, nx] and abs(nx - x) <= 32):
                        continue
                    # 重叠区: 仅在当前像素非重叠时允许跨越
                    if arr[CH_OVERLAP, f, x] < 1.5 / OVERLAP_NORM + 0.05 and arr[CH_OVERLAP, nf, nx] >= 1.5 / OVERLAP_NORM:
                        continue
                    pred = x + (x - prev_x)
                    score = (abs(nx - pred) * 2 + jump)
                    if best is None or score < best[0]:
                        best = (score, jump, nf, nx)
                if best is not None:
                    break
            if best is None:
                break
            _, jump, f, x = best
            for _ in range(jump):
                if path[-1][0] < f - 1:
                    # 填充跳过的帧 (用插值近似)
                    pf, px_ = path[-1]
                    for ff in range(pf + 1, f):
                        interp = px_ + (x - px_) * (ff - pf) / (f - pf)
                        used[ff, int(round(interp))] = True
            path.append((f, x))
            prev_x = x
            used[f, x] = True

        # 组装 seg (dt, dx) —— 取关键点: 起点 + 每帧, 压缩共线
        seg = []
        for k in range(1, len(path)):
            f, x = path[k]
            dt = (f - f0) / fps
            dx = x - x0
            seg.append({"dt": round(dt, 4), "dx": int(dx)})
        if seg:
            notes.append({
                "t": round(f0 / fps, 4), "x": int(x0),
                "w": int(round(arr[CH_SLIDE_W, f0, x0] * W_MAX)),
                "type": 2, "seg": seg,
            })
    notes.sort(key=lambda n: (n["t"], n["x"]))
    return notes


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    # 自测: 取几个真实谱面, 前向->反向, 统计重建率
    from collections import Counter
    recs = [json.loads(l) for l in open("data/dataset/slide_clean.jsonl", encoding="utf-8")]
    total_tap = total_drag = total_slide = 0
    hit_tap = hit_drag = hit_slide = 0
    for r in recs[:40]:
        arr = chart_to_array(r["notes"], r["length_beat"])
        back = array_to_chart(arr)
        orig = {(n["t"], n["x"], n["type"]): n for n in r["notes"]}
        rec = {(n["t"], n["x"], n["type"]): n for n in back}
        for key, n in orig.items():
            t, x, typ = key
            if typ == 0: total_tap += 1
            if typ == 1: total_drag += 1
            if typ == 2: total_slide += 1
            if key in rec:
                if typ == 0: hit_tap += 1
                if typ == 1: hit_drag += 1
                if typ == 2: hit_slide += 1
    print(f"前向->反向 重建率 (前40谱面, 无重叠区域):")
    print(f"  tap:   {hit_tap}/{total_tap} = {100*hit_tap/max(1,total_tap):.2f}%")
    print(f"  drag:  {hit_drag}/{total_drag} = {100*hit_drag/max(1,total_drag):.2f}%")
    print(f"  slide: {hit_slide}/{total_slide} = {100*hit_slide/max(1,total_slide):.2f}%")
