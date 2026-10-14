# -*- coding: utf-8 -*-
# 配套代码 7.2 —— 《视频异常事件检测原理及应用》第7章 视频异常检测基准测试体系
# 题注：随机种子敏感性分析示例
# 本文件由 _check/extract_companion_code.py 从 TeX_files/chapter-7-evaluation.tex 自动抽取，
# 与书中印刷代码逐字一致；请勿直接在此修改，需改请改 LaTeX 源后重新抽取。

import numpy as np
import torch, random

results = []
for seed in range(10):            # 10 个不同的随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    auc = train_and_evaluate()    # 完成一次完整的训练与评估
    results.append(auc)

results = np.array(results)
print("mean AUC: %.4f, std: %.4f" % (results.mean(), results.std()))
