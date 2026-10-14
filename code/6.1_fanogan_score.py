# -*- coding: utf-8 -*-
# 配套代码 6.1 —— 《视频异常事件检测原理及应用》第6章 深度学习视频异常检测方法
# 题注：f-AnoGAN 异常分数计算示例
# 本文件由 _check/extract_companion_code.py 从 TeX_files/chapter-6-dl-methods.tex 自动抽取，
# 与书中印刷代码逐字一致；请勿直接在此修改，需改请改 LaTeX 源后重新抽取。

import torch

@torch.no_grad()
def fanogan_score(x, encoder, generator, feat_net, lam=0.9):
    """A(x) = (1 - lam) * A_feat(x) + lam * A_res(x)"""
    z = encoder(x)
    x_res = generator(z)                                  # 一次重构
    x_res = 0.5 * (x_res + generator(encoder(x_res)))     # 回环精炼

    a_res = torch.norm(x_res - x, p=2, dim=(1, 2, 3))    # 残差项
    # flatten(1) 后按批维求范数，使 feat_net 输出 (N, C, H, W) 或 (N, D) 均可
    a_feat = torch.norm(feat_net(x_res).flatten(1) - feat_net(x).flatten(1), p=2, dim=1)
    return (1 - lam) * a_feat + lam * a_res
