# -*- coding: utf-8 -*-
# 配套代码 5.1 —— 《视频异常检测原理及应用》第5章 深度学习理论基础
# 题注：VAE 训练损失示例
# 本文件由 _check/extract_companion_code.py 从 TeX_files/chapter-5-dl-basic.tex 自动抽取，
# 与书中印刷代码逐字一致；请勿直接在此修改，需改请改 LaTeX 源后重新抽取。

import torch
import torch.nn.functional as F

BETA = 1e-3          # 正则项权重

def vae_loss(x, x_recon, mu, log_var):
    """总损失 = 重建项 + BETA * KL 散度"""
    rec = F.mse_loss(x_recon, x, reduction="none").flatten(1).sum(1)
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    total = rec + BETA * kl
    return total.mean(), rec.mean().item(), kl.mean().item()
