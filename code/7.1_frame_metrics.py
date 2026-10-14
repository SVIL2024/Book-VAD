# -*- coding: utf-8 -*-
# 配套代码 7.1 —— 《视频异常检测原理及应用》第7章 视频异常检测基准测试体系
# 题注：帧级指标与事件合并示例
# 本文件由 _check/extract_companion_code.py 从 TeX_files/chapter-7-evaluation.tex 自动抽取，
# 与书中印刷代码逐字一致；请勿直接在此修改，需改请改 LaTeX 源后重新抽取。

import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

def frame_level_metrics(labels, scores):
    """帧级 AUC 与平均精度 AP"""
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {"AUC": roc_auc_score(labels, scores),
            "AP": average_precision_score(labels, scores)}

def merge_events(flags):
    """把连续异常帧合并为事件区间，避免同一事件被重复计数"""
    events, start = [], None
    for t, f in enumerate(flags):
        if f and start is None:
            start = t
        elif not f and start is not None:
            events.append((start, t - 1))
            start = None
    if start is not None:
        events.append((start, len(flags) - 1))
    return events
