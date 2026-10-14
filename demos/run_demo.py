# -*- coding: utf-8 -*-
"""端到端演示：合成一段"正常 + 异常"的监控视频，跑通书中第3、4、7章的完整流程。

不依赖任何外部数据集，克隆即可运行：

    python demos/run_demo.py

流程对应书中章节：
    合成帧生成        -> 第3章 视频数据处理基础
    稠密光流          -> 代码清单 3.1
    光流直方图        -> 代码清单 4.1（Adam et al. 2009 的 MHOF 思路）
    正常模型 + 卡方距离 -> 代码清单 4.1
    帧级 AUC / AP     -> 代码清单 7.1
    事件区间合并      -> 代码清单 7.1
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'code'))

import importlib.util


def load(fname):
    path = os.path.join(HERE, '..', 'code', fname)
    spec = importlib.util.spec_from_file_location(fname[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_video(n_normal=40, n_abnormal=10, size=(160, 120), seed=0):
    """合成视频：正常段为匀速平移的亮块，异常段为原地抖动的亮块。"""
    rng = np.random.default_rng(seed)
    h, w = size
    frames, labels = [], []
    x = 20
    for i in range(n_normal):
        img = rng.integers(30, 60, (h, w, 3), dtype=np.uint8)
        cv2_rect(img, x, 50, 24, 40)
        frames.append(img)
        labels.append(0)
        x += 2                                  # 正常：匀速平移
        if x > w - 40:
            x = 20
    for i in range(n_abnormal):
        img = rng.integers(30, 60, (h, w, 3), dtype=np.uint8)
        cv2_rect(img, w // 2 + rng.integers(-6, 7), 50, 24, 40)
        frames.append(img)
        labels.append(1)                        # 异常：原地抖动
    return frames, labels


def cv2_rect(img, x, y, rw, rh):
    import cv2
    cv2.rectangle(img, (int(x), int(y)), (int(x + rw), int(y + rh)),
                  (220, 220, 220), -1)


def main():
    import cv2

    mhof = load('4.1_mhof_chi_square.py')
    metrics = load('7.1_frame_metrics.py')

    frames, labels = make_video()
    n = len(frames)
    print('合成视频：%d 帧（正常 %d，异常 %d），尺寸 %s'
          % (n, n - 10, 10, frames[0].shape[:2]))

    # 代码清单 3.1：Farneback 稠密光流
    print('\n[1] 计算稠密光流（代码清单 3.1）')
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    flows = [cv2.calcOpticalFlowFarneback(grays[i - 1], grays[i], None,
                                          0.5, 3, 15, 3, 5, 1.1, 0)
             for i in range(1, n)]
    mags = [float(np.linalg.norm(f, axis=-1).mean()) for f in flows]
    print('    正常段平均幅值 %.3f，异常段平均幅值 %.3f'
          % (np.mean(mags[:30]), np.mean(mags[30:])))

    # 代码清单 4.1：光流直方图 + 正常模型 + 卡方距离
    print('\n[2] 正常模型与异常分数（代码清单 4.1）')
    split = 25
    normal_hist = []
    for i in range(1, split + 1):
        h = mhof.flow_histogram(frames[i - 1], frames[i])
        normal_hist.append(h)
    model = np.mean(normal_hist, axis=0)
    model = model / max(model.sum(), 1e-8)

    scores = []
    for i in range(1, n):
        hist = mhof.flow_histogram(frames[i - 1], frames[i])
        scores.append(mhof.chi_square_distance(hist, model))
    scores = np.asarray(scores)
    print('    正常段分数均值 %.4f，异常段分数均值 %.4f'
          % (scores[:split - 1].mean(), scores[split - 5:].mean()))

    # 代码清单 7.1：帧级指标 + 事件合并
    print('\n[3] 帧级评价指标（代码清单 7.1）')
    frame_labels = labels[1:]
    frame_scores = scores / max(scores.max(), 1e-8)
    m = metrics.frame_level_metrics(frame_labels, frame_scores)
    print('    帧级 AUC = %.4f    AP = %.4f' % (m['AUC'], m['AP']))

    flag = (frame_scores > np.percentile(frame_scores, 90)).astype(int)
    events = metrics.merge_events(flag)
    print('    按 90%% 分位阈值检出 %d 个事件区间：%s' % (len(events), events[:5]))

    print('\n提示：本演示只用合成数据说明流程与接口，并非论文性能复现；')
    print('      真实评测需自行准备 UCF-Crime / XD-Violence 等数据集。')


if __name__ == '__main__':
    main()