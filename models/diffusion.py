# -*- coding: utf-8 -*-
"""
第二阶: latent 扩散模型 (DDPM, 2D UNet)。
输入: 谱面 latent (B, z_ch, H_l, W_l) (由 VAE encode 得到)
条件: 难度 Lv (1~25) + 音频 mel 特征 (按拍对齐, (B, T_mel, n_mels))
训练: 标准 DDPM epsilon 预测, smooth_l1 / l2 损失
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, ch, t_emb_dim, dropout=0.0, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def _forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.t_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h

    def forward(self, x, t_emb):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x, t_emb, use_reentrant=False)
        return self._forward(x, t_emb)


class Attention(nn.Module):
    def __init__(self, ch, heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.heads = heads

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        hd = C // self.heads
        q = q.view(B, self.heads, hd, H * W).transpose(2, 3)
        k = k.view(B, self.heads, hd, H * W).transpose(2, 3)
        v = v.view(B, self.heads, hd, H * W).transpose(2, 3)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), dim=-1)
        out = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        return x + self.proj(out)


class CrossAttention(nn.Module):
    """音频条件 cross-attention: query=谱面特征, key/value=音频特征"""
    def __init__(self, ch, ctx_dim, heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Linear(ctx_dim, ch)
        self.v = nn.Linear(ctx_dim, ch)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.heads = heads

    def forward(self, x, ctx):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, self.heads, C // self.heads, H * W).transpose(2, 3)  # (B,h,L,d)
        k = self.k(ctx).view(B, -1, self.heads, C // self.heads).transpose(1, 2)   # (B,h,T,d)
        v = self.v(ctx).view(B, -1, self.heads, C // self.heads).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(C // self.heads), dim=-1)
        out = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(self, in_ch=16, out_ch=16, model_ch=128, ch_mult=(1, 2, 4),
                 num_res_blocks=2, t_emb_dim=128, heads=8,
                 audio_ctx_dim=128, audio_n_mels=128, use_checkpoint=False,
                 with_audio=True, with_lv=True, dropout=0.0):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.with_audio = with_audio
        self.with_lv = with_lv
        self.conv_in = nn.Conv2d(in_ch, model_ch, 3, padding=1)
        self.t_emb = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4), nn.SiLU(), nn.Linear(t_emb_dim * 4, t_emb_dim))
        if with_lv:
            self.lv_emb = nn.Embedding(26, t_emb_dim)  # 0=无, 1~25
        chs = [model_ch * m for m in ch_mult]

        # ---- encoder: 每 level 一组 (ResBlock + 可选 CrossAttn) x num_res_blocks, 后接下采样 ----
        self.enc_levels = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch_now = model_ch
        for li, ch in enumerate(chs):
            level = nn.ModuleList()
            for _ in range(num_res_blocks):
                level.append(ResBlock(ch, t_emb_dim, dropout, use_checkpoint))
                if with_audio:
                    level.append(CrossAttention(ch, audio_ctx_dim, heads))
            self.enc_levels.append(level)
            if li < len(chs) - 1:
                self.downs.append(nn.Conv2d(ch, chs[li + 1], 3, stride=2, padding=1))
            ch_now = ch
        self.chs = chs

        # ---- middle ----
        self.mid = nn.ModuleList([
            ResBlock(chs[-1], t_emb_dim, dropout, use_checkpoint),
            Attention(chs[-1], heads),
            ResBlock(chs[-1], t_emb_dim, dropout, use_checkpoint),
        ])

        # ---- decoder: 每层 li 与 encoder 同层 skip 对齐, 仅非首层 upsample ----
        self.dec_levels = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        for li in range(len(chs) - 1, -1, -1):
            ch = chs[li]
            in_ch = ch if li == len(chs) - 1 else 2 * chs[li + 1]  # 首层输入=middle输出; 其余=上层concat后
            level = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                level.append(ResBlock(ch * 2, t_emb_dim, dropout, use_checkpoint))
                if with_audio:
                    level.append(CrossAttention(ch * 2, audio_ctx_dim, heads))
            self.dec_levels.append(level)
            self.up_projs.append(nn.Conv2d(in_ch, ch, 3, padding=1))
        # 最后 proj 回 model_ch
        self.out_proj = nn.Conv2d(model_ch * 2, model_ch, 3, padding=1)
        self.norm_out = nn.GroupNorm(8, model_ch)
        self.conv_out = nn.Conv2d(model_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t, lv=None, audio_ctx=None):
        B = x.shape[0]
        te = self.t_emb(timestep_embedding(t, self.t_emb_dim))
        if self.with_lv and lv is not None:
            te = te + self.lv_emb(lv)
        ctx = audio_ctx if audio_ctx is not None \
            else torch.zeros(B, 1, self.chs[0], device=x.device)

        h = self.conv_in(x)
        skips = []
        # encoder
        for li, level in enumerate(self.enc_levels):
            for blk in level:
                if isinstance(blk, CrossAttention):
                    h = blk(h, ctx)
                else:
                    h = blk(h, te)
            skips.append(h)
            if li < len(self.downs):
                h = self.downs[li](h)
        # middle
        for blk in self.mid:
            h = blk(h, te) if isinstance(blk, ResBlock) else blk(h)
        # decoder (深层 -> 浅层, 与 encoder skip 对齐)
        di = 0
        for li in range(len(self.chs) - 1, -1, -1):
            if li < len(self.chs) - 1:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = self.up_projs[di](h)
            skip = skips[li]
            if skip.shape[-2:] != h.shape[-2:]:
                skip = F.interpolate(skip, size=h.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for blk in self.dec_levels[di]:
                if isinstance(blk, CrossAttention):
                    h = blk(h, ctx)
                else:
                    h = blk(h, te)
            di += 1
        h = self.act(self.norm_out(self.out_proj(h)))
        return self.conv_out(h)


class DDPM(nn.Module):
    def __init__(self, unet, beta_start=1e-4, beta_end=0.02, timesteps=1000,
                 parameterization="eps"):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps
        self.parameterization = parameterization
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1 / alphas_cumprod - 1))

    def q_sample(self, x0, t, noise):
        return (self.sqrt_alphas_cumprod[t][:, None, None, None] * x0
                + self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise)

    def training_losses(self, x0, t, lv=None, audio_ctx=None):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.unet(xt, t, lv, audio_ctx)
        if self.parameterization == "eps":
            target = noise
        else:
            target = x0
        # 注意: 已回退普通 MSE。min-SNR 对 eps 预测压低高噪声步权重, 方向与需求相反。
        return F.mse_loss(pred, target, reduction="mean")

    @torch.no_grad()
    def sample(self, shape, lv=None, audio_ctx=None, steps=250, eta=0.0,
               cfg_scale=0.0, lv_null=None, audio_null=None, device="cuda"):
        """DDIM 采样 (steps < timesteps), 支持 classifier-free guidance。
        null 条件默认与训练一致: lv=0 + audio 全零 (而非 None)。
        """
        self.eval()
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.long, device=device)
        # 与训练 CFG drop 一致的 null 条件 (lv=0, audio 全零)
        if cfg_scale > 0:
            if lv_null is None:
                lv_null = torch.zeros(shape[0], dtype=torch.long, device=device)
            if audio_null is None:
                audio_null = torch.zeros_like(audio_ctx) if audio_ctx is not None else None
        for i, t in enumerate(ts):
            t_b = t.expand(shape[0])
            pred = self.unet(x, t_b, lv, audio_ctx)
            if cfg_scale > 0:
                null = self.unet(x, t_b, lv_null, audio_null)
                pred = null + cfg_scale * (pred - null)
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[ts[i + 1]] if i + 1 < steps else torch.ones_like(alpha)
            eps = pred if self.parameterization == "eps" else None
            x0_pred = self.sqrt_recip_alphas_cumprod[t] * x - self.sqrt_recipm1_alphas_cumprod[t] * pred
            x0_pred = x0_pred.clamp(-1, 1)
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha)) * torch.sqrt(1 - alpha / alpha_prev)
            c1 = torch.sqrt(alpha_prev) / torch.sqrt(alpha)
            c2 = torch.sqrt(1 - alpha_prev - sigma ** 2) - c1 * torch.sqrt(1 - alpha)
            x = c1 * x + c2 * pred + sigma * torch.randn_like(x)
            if eps is not None and False:
                pass
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    unet = UNet(with_audio=True, with_lv=True)
    ddpm = DDPM(unet)
    x0 = torch.randn(2, 16, 128, 32)
    t = torch.randint(0, 1000, (2,))
    lv = torch.tensor([12, 5])
    ctx = torch.randn(2, 128, 128)
    loss = ddpm.training_losses(x0, t, lv, ctx)
    print("训练损失:", loss.item())
    print(f"参数量: {sum(p.numel() for p in ddpm.parameters())/1e6:.2f}M")
