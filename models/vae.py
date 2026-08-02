# -*- coding: utf-8 -*-
"""AutoencoderKL: 谱面图像 (10, H, 256) -> latent (16, H/8, 32) -> 重建。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch, groups=8, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.SiLU()

    def _forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class ChanSwitch(nn.Module):
    """升维/降维 (1x1 conv), 通道数不变时为 Identity"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_ch=9, middle=64, z_ch=16, ch_mult=(1, 2, 4, 4), num_res_blocks=1,
                 use_checkpoint=False):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, middle, 3, padding=1)
        chs = [middle * m for m in ch_mult]
        blocks = []
        in_ch_ = middle
        for i, ch in enumerate(chs):
            blocks.append(ChanSwitch(in_ch_, ch))
            in_ch_ = ch
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, use_checkpoint=use_checkpoint))
            if i < len(chs) - 1:
                blocks.append(Downsample(ch))
        self.blocks = nn.Sequential(*blocks)
        self.norm_out = nn.GroupNorm(8, chs[-1])
        self.conv_out = nn.Conv2d(chs[-1], 2 * z_ch, 3, padding=1)
        self.z_ch = z_ch

    def forward(self, x):
        h = self.conv_in(x)
        h = self.blocks(h)
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, out_ch=9, middle=64, z_ch=16, ch_mult=(1, 2, 4, 4), num_res_blocks=1,
                 use_checkpoint=False):
        super().__init__()
        chs = [middle * m for m in ch_mult]
        self.conv_in = nn.Conv2d(z_ch, chs[-1], 3, padding=1)
        blocks = []
        for i in range(len(chs) - 1, 0, -1):
            ch = chs[i]
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, use_checkpoint=use_checkpoint))
            blocks.append(Upsample(ch))
            blocks.append(ChanSwitch(ch, chs[i - 1]))
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(chs[i - 1], use_checkpoint=use_checkpoint))
        self.blocks = nn.Sequential(*blocks)
        self.norm_out = nn.GroupNorm(8, chs[0])
        self.conv_out = nn.Conv2d(chs[0], out_ch, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.blocks(h)
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return h


class AutoencoderKL(nn.Module):
    def __init__(self, in_ch=10, middle=64, z_ch=16, ch_mult=(1, 2, 4, 4), kl_weight=1e-4,
                 use_checkpoint=False):
        super().__init__()
        self.encoder = Encoder(in_ch, middle, z_ch, ch_mult, use_checkpoint=use_checkpoint)
        self.decoder = Decoder(in_ch, middle, z_ch, ch_mult, use_checkpoint=use_checkpoint)
        self.kl_weight = kl_weight
        self.z_ch = z_ch

    def encode(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar.clamp(-30, 10))
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        rec = self.decoder(z)
        return rec, z, mean, logvar


class ChartReconstructLoss(nn.Module):
    """通道加权重建损失。
    通道: 0 tap_mask, 1 tap_width, 2 drag_mask, 3 drag_width,
          4 slide_mask(0/1), 5 slide_start(0/1), 6 slide_width,
          7 overlap, 8 beat, 9 bar
    """
    CH_LOSS = {
        0: "bce", 1: "l1_masked", 2: "bce", 3: "l1_masked",
        4: "bce", 5: "bce", 6: "l1_masked",
        7: "l1_masked", 8: "bce", 9: "bce",
    }
    CH_WEIGHT = {
        0: 2.0, 1: 1.0, 2: 2.0, 3: 1.0,
        4: 3.0, 5: 3.0, 6: 1.0, 7: 0.5, 8: 0.2, 9: 0.2,
    }

    def __init__(self, label_smoothing=0.001, pos_weight=100.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight

    def forward(self, pred_logits, target):
        # pred_logits: (B, N_CH, H, W) 未过 sigmoid; target 0~1
        total = 0.0
        losses = {}
        p = torch.sigmoid(pred_logits)
        for c, ltype in self.CH_LOSS.items():
            pl = pred_logits[:, c]
            pt = p[:, c]
            t = target[:, c]
            w = self.CH_WEIGHT[c]
            if ltype == "bce":
                loss = F.binary_cross_entropy_with_logits(
                    pl, t, pos_weight=torch.tensor(self.pos_weight, device=pl.device))
            elif ltype == "l1_masked":
                mask = (t > 0.05).float()
                n = mask.sum().clamp(min=1)
                loss = (mask * (pt - t).abs()).sum() / n
                # 零区域轻惩罚 (防止空洞输出)
                loss = loss + 0.05 * ((1 - mask) * pt).mean()
            total = total + w * loss
            losses[c] = float(loss.item())
        # 互斥: 同一像素 tap 与 drag 不应同时激活
        mutual = (p[:, 0] * p[:, 2]).mean()
        total = total + 1.0 * mutual
        losses["mutual"] = float(mutual.item())
        return total, losses


if __name__ == "__main__":
    vae = AutoencoderKL()
    x = torch.randn(2, 9, 512, 256)
    rec, z, mean, logvar = vae(x)
    print("输入:", x.shape, "latent:", z.shape, "输出:", rec.shape)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"参数量: {n_params/1e6:.2f}M")
    loss_fn = ChartReconstructLoss()
    loss, ls = loss_fn(rec, x)
    print("损失:", loss.item())
