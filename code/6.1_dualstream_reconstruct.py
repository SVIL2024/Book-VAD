# -*- coding: utf-8 -*-
# 配套代码 6.1 —— 《视频异常检测原理及应用》第6章 深度学习视频异常检测方法
# 题注：双流时空自编码器重构误差融合示例
# 本文件由 _check/extract_companion_code.py 从 code_src/6.1_dualstream_reconstruct.py 自动抽取，
# 是书中代码清单 6.1 的完整可运行实现（书中仅摘录第 12-34 行的核心片段）。
# 请勿直接在此修改，需改请改 code_src/6.1_dualstream_reconstruct.py 后重新抽取。

# -*- coding: utf-8 -*-
"""双流时空自编码器的重构误差融合（对应正文 6.2.1 节）。

外观流以原始帧为输入、运动流以帧差为输入，各自配备独立的编码器—解码器，
最终将两路重构误差加权融合为异常分数：

    S = lambda * ||x_t - x_hat_t||^2 + (1 - lambda) * ||d_t - d_hat_t||^2

其中 d_t 为运动表示、lambda 为平衡系数。仅用正常视频训练，测试时重构误差
大者为异常。完整实现可直接运行：python 6.1_dualstream_reconstruct.py
"""
import torch
import torch.nn as nn


def motion_repr(frames):
    """运动表示：相邻帧之差 d_t = x_t - x_{t-1}。

    输入 clips 形状 (B, T, C, H, W)，首帧以零位移对齐，返回 (B, T-1, C, H, W)。
    """
    return frames[:, 1:] - frames[:, :-1]


def fusion_score(x, x_hat, d, d_hat, lam=0.5):
    """双路重构误差加权融合，得到每个样本的异常分数（正文式 6.x）。

        S = lambda * ||x_t - x_hat_t||^2 + (1 - lambda) * ||d_t - d_hat_t||^2

    逐元素均方误差后沿特征维求和；返回值形状 (B,)。
    """
    app = (x - x_hat).pow(2).flatten(1).sum(1)
    mot = (d - d_hat).pow(2).flatten(1).sum(1)
    return lam * app + (1.0 - lam) * mot


class ConvAE(nn.Module):
    """卷积自编码器，编码器/解码器各自独立实例化。"""

    def __init__(self, ch=3, width=16, code=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(ch, width, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(width, width, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(width, code, 3, 2, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(code, width, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(width, width, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(width, ch, 4, 2, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DualStreamAE(nn.Module):
    """双流时空自编码器：外观流与运动流各一套编解码器。

    运动流的取值范围与外观流差异很大，归一化参数须分别设定，故此处
    保留各自的 scale 参数。
    """

    def __init__(self, ch=3, width=16, code=32):
        super().__init__()
        self.appearance = ConvAE(ch, width, code)
        self.motion = ConvAE(ch, width, code)
        self.app_scale = 1.0
        self.mot_scale = 5.0

    def forward(self, frames):
        """frames: (B, T, C, H, W) -> (x_hat, d_hat)。

        外观流逐帧重构；运动流对帧差序列重构后按位移回填到 T-1 帧。
        """
        b, t = frames.shape[:2]
        flat = frames.reshape(b * t, *frames.shape[2:])
        x_hat = self.appearance(flat).reshape(b, t, *frames.shape[2:])

        d = motion_repr(frames)
        d_hat = self.motion(d.reshape(-1, *d.shape[2:])).reshape(d.shape)
        return x_hat, d_hat

    def score(self, frames, lam=0.5):
        """返回形状 (B, T) 的逐帧异常分数。"""
        x_hat, d_hat = self.forward(frames)
        d = motion_repr(frames)
        # 运动表示只覆盖 T-1 帧，故外观项同样取 1..T-1 帧与之对齐
        app = (frames[:, 1:] - x_hat[:, 1:]).pow(2).flatten(2).sum(2)
        mot = (d - d_hat).pow(2).flatten(2).sum(2)
        s = lam * app + (1.0 - lam) * mot
        # 首帧没有对应运动表示，以其外观分数占位，保证逐帧输出长度仍为 T
        first = (frames[:, :1] - x_hat[:, :1]).pow(2).flatten(2).sum(2)
        return torch.cat([first, s], dim=1)


def train_on_normal(model, clips, epochs=30, lr=1e-3, batch=8):
    """仅用正常片段训练（正文 6.2.1 节的关键设定）。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = clips.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            c = clips[idx]
            x_hat, d_hat = model(c)
            loss = (c - x_hat).pow(2).mean() + (c[:, 1:] - c[:, :-1] - d_hat).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _synthetic_normal(n, t=8, c=3, s=32, seed=0):
    """合成正常片段：静态背景上一枚平滑漂移的亮斑。"""
    g = torch.Generator().manual_seed(seed)
    bg = torch.rand(n, t, c, s, s, generator=g) * 0.1
    for i in range(n):
        for j in range(t):
            x = 8 + j
            y = 8 + (j * i) % 8
            bg[i, j, :, y:y + 5, x:x + 5] += 0.6
    return bg.clamp(0, 1)


def _inject_anomaly(clip, seed=1):
    """在片段中植入训练集从未出现的高对比方块。"""
    out = clip.clone()
    out[:, :, :, 2:9, 2:9] = 1.0 - out[:, :, :, 2:9, 2:9]
    return out


if __name__ == '__main__':
    torch.manual_seed(0)
    normal = _synthetic_normal(12)
    model = train_on_normal(DualStreamAE(), normal, epochs=30)

    clean = _synthetic_normal(3, seed=7)
    dirty = _inject_anomaly(clean, seed=7)
    with torch.no_grad():
        s_clean = model.score(clean, lam=0.5).mean().item()
        s_dirty = model.score(dirty, lam=0.5).mean().item()

    print('正常片段平均分数   : %.4f' % s_clean)
    print('异常片段平均分数   : %.4f' % s_dirty)
    assert s_clean >= 0.0, '异常分数不应为负'
    assert s_dirty > s_clean * 1.2, '异常片段分数应明显高于正常片段'
    print('PASS 双流重构：异常片段 %.2fx 于正常片段' % (s_dirty / max(s_clean, 1e-8)))
