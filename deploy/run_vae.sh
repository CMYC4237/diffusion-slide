#!/usr/bin/env bash
# VAE 正式训练: 2x4090 DDP
# 显存实测 (4060, bf16+checkpoint): batch 8 = 7.5GB -> 4090 16GB 安全, 可 batch 12
# 默认只用 0,2 号卡 (你的可用卡); 其他卡用 CUDA_VISIBLE_DEVICES 覆盖
set -e
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-8}
SAMPLES=${SAMPLES:-5000}
GPUS=${GPUS:-0,2}
export CUDA_VISIBLE_DEVICES=$GPUS
echo "使用物理卡: $CUDA_VISIBLE_DEVICES (可用 GPUS=1,3 bash $0 覆盖)"

torchrun --nproc_per_node=2 --master_port=29500 \
    train/train_vae.py \
    --epochs "$EPOCHS" --batch "$BATCH" --lr 3e-4 --workers 4 \
    --samples_per_epoch "$SAMPLES" --out checkpoints/vae --dist

echo "完成: checkpoints/vae/best.ckpt"
