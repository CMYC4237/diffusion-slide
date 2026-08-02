#!/usr/bin/env bash
# 数据准备: Slide.zip -> 干净谱面 jsonl + 音频 mel 缓存
# 先激活虚拟环境: source .venv/bin/activate
set -e
cd "$(dirname "$0")/.."

# 自动选择 python: venv 激活后是 python, 否则 python3
if command -v python >/dev/null 2>&1; then
    PY=python
else
    PY=python3
fi

echo "[1/2] 清洗谱面数据 (筛掉缺 w 谱面)..."
$PY data/build_dataset.py

echo "[2/2] 提取音频 + mel 特征 (约 20-40 分钟)..."
$PY data/extract_audio.py

echo "完成: data/dataset/slide_clean.jsonl + data/audio/ (mel 缓存)"
