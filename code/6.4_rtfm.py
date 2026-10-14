# -*- coding: utf-8 -*-
# 配套代码 6.4 —— 《视频异常检测原理及应用》第6章 深度学习视频异常检测方法
# 题注：RTFM 特征幅值对比损失示例
# 本文件由 _check/extract_companion_code.py 从 code_src/6.4_rtfm.py 自动抽取，
# 是书中代码清单 6.4 的完整可运行实现（书中仅摘录第 13-28 行的核心片段）。
# 请勿直接在此修改，需改请改 code_src/6.4_rtfm.py 后重新抽取。

# -*- coding: utf-8 -*-
"""RTFM 特征幅值对比学习（对应正文 6.4.2 节，Tian 等人, CVPR 2021）。

3D CNN 特征提取器与时间建模模块把片段 V 映射为特征序列 H，然后以特征
幅值作为可分性度量：正常片段幅值小、异常片段幅值大。训练目标为

    L_rtfm = (1/M) * sum_i [ r(V_n^i) - r(V_a^i) + m ]_+

其中 r(V) = ||H||_2 为特征幅值，M 为样本对数量，m 为间隔，[.]_+ 为 ReLU。
该损失仅需视频级标签，故属弱监督设定。完整实现可直接运行：
python 6.4_rtfm.py
"""
import torch
import torch.nn as nn


def feature_magnitude(H):
    """特征幅值 r(V) = ||H||_2，沿批维以外的维数求 L2 范数。

    H 形状 (M, T, d)，返回形状 (M,)。
    """
    return H.pow(2).sum(dim=(1, 2)).sqrt()


def rtfm_loss(mag_normal, mag_abnormal, margin=1.0):
    """特征幅值对比损失（正文式 6.x），形状 (M,) -> 标量。"""
    return torch.relu(mag_normal - mag_abnormal + margin).mean()


class FeatureExtractor(nn.Module):
    """3D CNN 特征提取器 F_theta 与时间建模模块（TCN 式）。"""

    def __init__(self, in_ch=3, width=8, code=32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(in_ch, width, 3, (1, 2, 2), 1), nn.ReLU(True),
            nn.Conv3d(width, code, 3, (1, 2, 2), 1), nn.ReLU(True),
        )
        self.temporal = nn.Conv1d(code, code, 3, 1, 1)

    def forward(self, clips):
        """clips: (B, T, C, H, W) -> H 特征序列 (B, T', code)。"""
        b, t = clips.shape[:2]
        f = self.backbone(clips.permute(0, 2, 1, 3, 4))     # (B, code, T', h, w)
        f = f.mean(dim=(3, 4))                               # 全局池化去空间维
        return self.temporal(f).permute(0, 2, 1)


def train_rtfm(model, normal_clips, abnormal_clips, epochs=40, lr=1e-3, margin=1.0):
    """用配对的正常/异常片段训练，使正常幅值比异常幅值至少小 margin。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        hn = model(normal_clips)
        ha = model(abnormal_clips)
        loss = rtfm_loss(feature_magnitude(hn), feature_magnitude(ha), margin)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses


@torch.no_grad()
def detect(model, clips):
    """推理：直接以特征幅值作为片段级异常分数，形状 (B,)。"""
    model.eval()
    return feature_magnitude(model(clips))


def _synthetic_pairs(n=8, t=8, c=3, s=32, seed=0):
    """合成配对片段：正常=静态背景+缓慢漂移亮斑；异常=额外出现的高频抖动方块。"""
    g = torch.Generator().manual_seed(seed)
    normal = torch.rand(n, t, c, s, s, generator=g) * 0.1
    abnormal = normal.clone()
    for i in range(n):
        for j in range(t):
            ox, oy = 2 + (j * 3 + i) % 10, 2 + (j * 5 + i) % 10
            normal[i, j, :, oy:oy + 5, ox:ox + 5] += 0.6
            if j % 2 == 0:
                abnormal[i, j, :, oy:oy + 5, ox:ox + 5] += 0.6
            else:
                # 抖动：同一物体在相邻帧大幅位移，制造异常运动
                px, py = 20 + (j % 2) * 6, 20 + ((j + 1) % 2) * 6
                abnormal[i, j, :, py:py + 5, px:px + 5] += 0.6
    return normal.clamp(0, 1), abnormal.clamp(0, 1)


if __name__ == '__main__':
    torch.manual_seed(0)
    normal, abnormal = _synthetic_pairs(8)
    model = FeatureExtractor()
    _, losses = train_rtfm(model, normal, abnormal, epochs=40, margin=0.5)

    s_normal = detect(model, normal)
    s_abnormal = detect(model, abnormal)
    print('训练前 loss        : %.4f' % losses[0])
    print('训练后 loss        : %.4f' % losses[-1])
    print('正常片段幅值       : %.4f' % s_normal.mean().item())
    print('异常片段幅值       : %.4f' % s_abnormal.mean().item())

    assert losses[-1] < losses[0], '损失应随训练下降'
    assert s_normal.mean().item() < s_abnormal.mean().item(), '正常幅值应小于异常幅值'
    margin = s_abnormal.mean().item() - s_normal.mean().item()
    print('PASS RTFM：幅值间隔 %.4f，损失下降 %.1f%%'
          % (margin, 100 * (1 - losses[-1] / max(losses[0], 1e-8))))
