# -*- coding: utf-8 -*-
# 配套代码 3.2 —— 《视频异常检测原理及应用》第3章 视频数据处理基础
# 题注：视频帧读取、缩放与归一化示例
# 本文件由 _check/extract_companion_code.py 从 TeX_files/chapter-3-dataproc.tex 自动抽取，
# 与书中印刷代码逐字一致；请勿直接在此修改，需改请改 LaTeX 源后重新抽取。

import cv2
import numpy as np

# 读取视频帧并缩放到统一分辨率
frame = cv2.imread("frame_0001.jpg", cv2.IMREAD_COLOR)
frame = cv2.resize(frame, (224, 224))

# 像素值从 [0, 255] 归一化到 [-1, 1]
frame = frame.astype(np.float32)
frame = (frame / 127.5) - 1.0
