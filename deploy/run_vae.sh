#!/usr/bin/env bash
# VAE 正式训练: 2x4090 DDP
set -e
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-8}
SAMPLES=${SAMPLES:-5000}

torchrun --nproc_per_node=2 --master_port=29500 \
    train/train_vae.py \
    --epochs "$EPOCHS" --batch "$BATCH" --lr 3e-4 --workers 4 \
    --samples_per_epoch "$SAMPLES" --out checkpoints/vae --dist

echo "完成: checkpoints/vae/best.ckpt"
