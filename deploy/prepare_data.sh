#!/usr/bin/env bash
# 数据准备: Slide.zip -> 干净谱面 jsonl + 音频 mel 缓存
set -e
cd "$(dirname "$0")/.."

echo "[1/2] 清洗谱面数据 (筛掉缺 w 谱面)..."
python data/build_dataset.py

echo "[2/2] 提取音频 + mel 特征 (约 20-40 分钟)..."
python data/extract_audio.py

echo "完成: data/dataset/slide_clean.jsonl + data/audio/ (mel 缓存)"
