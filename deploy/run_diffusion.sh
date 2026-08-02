#!/usr/bin/env bash
# 扩散模型全量训练: 2x4090 DDP (依赖 checkpoints/vae/best.ckpt)
# 显存实测 (4060, bf16+checkpoint): batch 8 = 3.1GB -> 4090 16GB 可 batch 16
# 默认只用 0,2 号卡 (你的可用卡); 其他卡用 CUDA_VISIBLE_DEVICES 覆盖
set -e
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-80}
BATCH=${BATCH:-8}
SAMPLES=${SAMPLES:-5000}
VAE_CKPT=${VAE_CKPT:-checkpoints/vae/best.ckpt}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2}

torchrun --nproc_per_node=2 --master_port=29501 \
    train/train_diffusion.py \
    --epochs "$EPOCHS" --batch "$BATCH" --lr 1e-4 \
    --vae_ckpt "$VAE_CKPT" \
    --samples_per_epoch "$SAMPLES" --out checkpoints/diffusion --dist

echo "完成: checkpoints/diffusion/last.ckpt"
