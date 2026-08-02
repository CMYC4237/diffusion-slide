# -*- coding: utf-8 -*-
"""DDP 正确性诊断: 验证 1) 数据被分片 2) 梯度 all-reduce 同步 3) 参数跨 rank 一致。
用法: CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 train/diag_ddp.py
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from vae import AutoencoderKL


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # 1) 数据分片: 每 rank 独立种子, 喂不同数据 (模拟 DistributedSampler)
    torch.manual_seed(1000 + rank)
    x = torch.randn(2, 9, 256, 256, device=device)

    # 2) 梯度同步: 标准 DDP
    torch.manual_seed(0)
    model = AutoencoderKL(middle=32, ch_mult=(1, 2)).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    loss = model(x)[0].sum()
    opt.zero_grad()
    loss.backward()

    g0 = model.module.encoder.conv_in.weight.grad.detach().clone()
    g_sum = g0.clone()
    dist.all_reduce(g_sum)
    # DDP 把梯度 all-reduce 成平均, 所以 g0 应等于 g_sum/world
    synced = torch.allclose(g0, g_sum / world, atol=1e-5)
    print(f"[rank{rank}] 梯度已 all-reduce 同步(每 rank 本地梯度=全局平均): {synced}", flush=True)

    # 3) 参数一致性: 一步 SGD 后两 rank 权重应完全一致
    opt.step()
    w = model.module.encoder.conv_in.weight.data.detach().clone()
    ws = w.clone()
    dist.all_reduce(ws)
    same = torch.allclose(w, ws / world)
    print(f"[rank{rank}] SGD 一步后参数跨 rank 一致: {same}", flush=True)

    dist.destroy_process_group()
    print(f"[rank{rank}] 诊断完成 (world={world})", flush=True)


if __name__ == "__main__":
    main()
