# Diffusion-Slide 集群部署包

无轨 (Malody Slide, mode 7) 谱面生成 — 两阶段扩散模型。
本目录包含在 2×4090 集群上跑正式训练的一键脚本。

## 0. 前置

- 代码: `git clone https://github.com/CMYC4237/diffusion-slide`
- Python 环境: `pip install torch numpy librosa soundfile` (CUDA 版 torch)
- 数据: 见下方"1. 数据准备"
- 磁盘: 数据约 2GB，checkpoint 约 0.5GB

## 1. 数据准备 (本地或集群, 一次性)

数据源是 `Sort By mode/Slide.zip` (未入库, 需自行获取, 放项目根目录下)。

```bash
bash deploy/prepare_data.sh
# 生成:
#   data/dataset/slide_clean.jsonl   (732 谱面, 23MB)
#   data/audio/*.ogg + *_mel.npy     (248 首, 1.9GB, 耗时约 20-40 分钟)
```

如果集群无法访问原始 zip：把本地生成的 `data/dataset/slide_clean.jsonl`
和整个 `data/audio/` 目录 rsync/scp 到集群项目目录即可，跳过本步。

## 2. 训练 VAE (第一阶, 谱面图像重建)

```bash
bash deploy/run_vae.sh
# 等价于:
# torchrun --nproc_per_node=2 --master_port=29500 train/train_vae.py \
#   --epochs 40 --batch 8 --lr 3e-4 --workers 4 \
#   --samples_per_epoch 5000 --out checkpoints/vae --dist
```

- 显存: batch 8 × 2 卡 16GB 安全 (本地 4060 用 batch 2)
- 产出: `checkpoints/vae/best.ckpt`
- 预期: val loss < 0.03 (smoke 6 epoch 已到 0.05)，重建率简单谱面 100%
- 期间可 `nvidia-smi` 观察双卡利用率

### VAE 训练完评估

```bash
python train/eval_vae.py --ckpt checkpoints/vae/best.ckpt
# 生成 data/vis/vae_recon/*.png 对比图, 打印窗口内重建匹配率
```

## 3. 训练扩散模型 (第二阶, 难度+音频条件)

```bash
bash deploy/run_diffusion.sh
# 等价于:
# torchrun --nproc_per_node=2 --master_port=29501 train/train_diffusion.py \
#   --epochs 80 --batch 4 --lr 1e-4 --vae_ckpt checkpoints/vae/best.ckpt \
#   --samples_per_epoch 5000 --out checkpoints/diffusion --dist
```

- 条件: 难度 Lv 1~25 + 音频 mel (按拍对齐), 训练时 10% 丢弃 (CFG)
- 产出: `checkpoints/diffusion/last.ckpt` + 每 5 epoch 存档
- 显存: batch 4 × 2 卡 (117M 参数 + checkpoint 激活, 16GB 安全)

## 4. 生成谱面

```bash
python train/generate.py --diff_ckpt checkpoints/diffusion/last.ckpt \
    --vae_ckpt checkpoints/vae/best.ckpt \
    --song 10072 --lv 12 --cfg 1.5 --steps 100
# 输出 output/ai_lv12_song10072.mc (Malody 可导入)
```

- `--song`: 用哪首歌的音频 (需要 data/audio 有对应 mel)
- `--lv`: 目标难度 1~25
- `--cfg`: classifier-free guidance 强度 (0=无条件, 1~2 推荐)
- `--steps`: DDIM 步数 (50~200)

## 5. 产物回传

把 `checkpoints/vae/best.ckpt` 和 `checkpoints/diffusion/last.ckpt` 传回本地
(或直接在集群上跑 `train/generate.py` 生成 `.mc` 再传回)。

## 常见问题

- **OOM**: 降低 `--batch` (VAE 2, 扩散 2), 或加 `--no-checkpoint` 换显存/速度
- **多 bpm 谱面**: 已支持 (bpms 表逐段累积拍→秒映射)
- **无 Lv 谱面**: 训练时按 lv=0 (null 条件) 处理, 不影响 CFG
- **训练慢**: 先 `--samples_per_epoch 1000` 验证, 再放量
