# Diffusion Slide — 无轨音游 (Malody Slide, mode 7) 谱面生成

基于 Stable Diffusion 思路的 Malody **无轨 (Slide)** 模式谱面生成项目：
输入 **音乐 + 难度 (Lv 1~25)**，输出 **谱面 JSON**（tap / drag / slide 滑条）。

采用两阶段 latent diffusion（参考 [Mug-Diffusion](https://github.com/Keytoyze/Mug-Diffusion)）：
谱面 → 多通道图像 → VQVAE/KL-AE 编码 latent → 扩散模型（音频 mel 条件 + 难度条件）→ 解码 → 谱面 JSON。

## 数据格式（Malody .mc, mode 7）

- `note.beat`: `[a, b, c]` = 第 `a + b/c` 拍
- `note.x`: 横向位置 (0~255)，`note.w`: 判定宽度
- `note.type: 4`: drag（拖拽判定点）
- `note.seg`: 滑条路径节点，`beat` 相对起点，`x` 相对位移（滑条 = 曲线）

## 通道表示（9 通道, H x W = 时间帧 x 256px）

| 通道 | 内容 |
|---|---|
| 0/1 | tap mask / tap width |
| 2/3 | drag mask / drag width |
| 4/5 | slide 中心线 (起点=2, 路径=1) / slide width |
| 6 | 滑条重叠计数 (归一化) |
| 7/8 | 拍线 / 小节线 |

帧率 16 帧/拍；滑条渲染为 1px 中心线（非填充带），保证并排/交叉滑条结构可分离。

## 项目结构

```
data/
  build_dataset.py      # Slide.zip -> 干净谱面 JSONL (筛掉缺 w 谱面)
  convertor.py          # 谱面 <-> 9通道图像 (含反向提取)
  dataset_vae.py        # VAE 训练数据 (窗口采样 + 镜像增强)
  extract_audio.py      # mcz 提取音频 + librosa mel 缓存
models/
  vae.py                # AutoencoderKL + ChartReconstructLoss
train/
  train_vae.py          # 第一阶 VAE 训练
  eval_vae.py           # 重建评估 (对比图 + 重建匹配率)
```

## 运行

```bash
# 1. 清洗数据 (需 Sort By mode/Slide.zip)
python data/build_dataset.py

# 2. 提取音频特征
python data/extract_audio.py

# 3. 训练 VAE (本地 4060 建议 batch 2 + amp; 集群 4090 可 batch 8)
python train/train_vae.py --epochs 20 --batch 2

# 4. 评估
python train/eval_vae.py --ckpt checkpoints/vae/best.ckpt
```

## 当前进度

- [x] 数据理解与清洗 (732 谱面, 38.5 万 note)
- [x] 谱面 <-> 图像 convertor（前向无损, 反向重建 slide 99.1%）
- [x] 音频特征管线 (248 首 mel 缓存)
- [x] 第一阶 VAE 训练管线（smoke test 通过: 简单谱面重建 100%, 高难 88.6%）
- [ ] 扩散模型（难度 + 音频条件）
- [ ] 生成 → 谱面 JSON 回写
