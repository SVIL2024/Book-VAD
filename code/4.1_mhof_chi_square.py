# -*- coding: utf-8 -*-
# 配套代码 4.1 —— 《视频异常检测原理及应用》第4章 传统视频异常检测方法
# 题注：光流直方图与卡方距离异常分数示例
# 本文件由 _check/extract_companion_code.py 从 TeX_files/chapter-4-ml.tex 自动抽取，
# 与书中印刷代码逐字一致；请勿直接在此修改，需改请改 LaTeX 源后重新抽取。

import cv2
import numpy as np

BIN_U, BIN_V, BIN_N = 8, 8, 16    # 水平/垂直速度分量与运动幅值的分箱数
U_SCALE, V_SCALE = 5.0, 5.0       # 速度分量归一化尺度（像素/帧）

def flow_histogram(prev, curr):
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    u, v = flow[..., 0], flow[..., 1]               # 水平/垂直速度分量
    mag = np.sqrt(u ** 2 + v ** 2)                 # 运动幅值

    h, w = mag.shape                                # 裁剪到分箱网格的整数倍
    h, w = h // BIN_N * BIN_N, w // BIN_N * BIN_N
    u, v, mag = u[:h, :w], v[:h, :w], mag[:h, :w]

    hu = np.clip((u / U_SCALE * (BIN_U - 1) + BIN_U / 2).astype(int), 0, BIN_U - 1)
    hv = np.clip((v / V_SCALE * (BIN_V - 1) + BIN_V / 2).astype(int), 0, BIN_V - 1)
    hn = np.clip((mag / 10.0 * (BIN_N - 1)).astype(int), 0, BIN_N - 1)

    # 直方图下标须由 (hu, hv) 唯一确定，故用 hu * BIN_V + hv 而非 hu + hv
    idx = hn * (BIN_U * BIN_V) + hu * BIN_V + hv
    hist = np.bincount(idx.ravel(),
                       minlength=BIN_N * BIN_U * BIN_V).reshape(BIN_N, BIN_U * BIN_V)
    return hist / max(hist.sum(), 1e-8)

def chi_square_distance(p, q):
    """两幅光流直方图之间的卡方距离"""
    return float(np.sum((p - q) ** 2 / (p + q + 1e-6)))
