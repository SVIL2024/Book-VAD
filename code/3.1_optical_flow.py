# -*- coding: utf-8 -*-
# 配套代码 3.1 —— 《视频异常事件检测原理及应用》第3章 视频数据处理基础
# 题注：光流计算示例
# 本文件由 _check/extract_companion_code.py 从 TeX_files/chapter-3-dataproc.tex 自动抽取，
# 与书中印刷代码逐字一致；请勿直接在此修改，需改请改 LaTeX 源后重新抽取。

import cv2
import numpy as np

prev = cv2.cvtColor(cv2.imread("frame_0001.jpg"), cv2.COLOR_BGR2GRAY)
curr = cv2.cvtColor(cv2.imread("frame_0002.jpg"), cv2.COLOR_BGR2GRAY)

# Farneback 多尺度变分光流：先金字塔粗到细，再逐层细化
flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                    pyr_scale=0.5, levels=3, winsize=15,
                                    iterations=3, poly_n=5, poly_sigma=1.1, flags=0)

u, v = flow[..., 0], flow[..., 1]
magnitude = np.sqrt(u ** 2 + v ** 2)      # 运动幅值
direction = np.arctan2(v, u)              # 运动方向，范围 (-pi, pi]
print("mean magnitude: %.4f" % magnitude.mean())
