#!/usr/bin/env bash
# 扩散模型全量训练: 2x4090 DDP (依赖 checkpoints/vae/best.ckpt)
# 显存实测 (4060, bf16+checkpoint): batch 8 = 3.1GB -> 4090 16GB 可 batch 16
# 默认只用 0,2 号卡 (你的可用卡); 其他卡用 CUDA_VISIBLE_DEVICES 覆盖
# 断点续训: RESUME=1 bash deploy/run_diffusion.sh (从 checkpoints/diffusion/last.ckpt 继续)
set -e
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-80}
BATCH=${BATCH:-8}
SAMPLES=${SAMPLES:-5000}
VAE_CKPT=${VAE_CKPT:-checkpoints/vae/best.ckpt}
OUT=${OUT:-checkpoints/diffusion}
GPUS=${GPUS:-0,2}
export CUDA_VISIBLE_DEVICES=$GPUS
echo "使用物理卡: $CUDA_VISIBLE_DEVICES (可用 GPUS=1,3 bash $0 覆盖)"
echo "输出目录: $OUT"

RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
    RESUME_FLAG="--resume"
    echo "断点续训模式: 从 checkpoints/diffusion/last.ckpt 继续"
elif [ "${WARM_START:-0}" = "1" ]; then
    RESUME_FLAG="--warm-start"
    echo "热启动模式: 加载旧权重从 epoch 0 重训 (新损失/归一化)"
fi

torchrun --nproc_per_node=2 --master_port=29501 \
    train/train_diffusion.py \
    --epochs "$EPOCHS" --batch "$BATCH" --lr 1e-4 \
    --vae_ckpt "$VAE_CKPT" \
    --samples_per_epoch "$SAMPLES" --out "$OUT" --dist $RESUME_FLAG

echo "完成: $OUT/last.ckpt"
