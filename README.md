# 《视频异常事件检测原理及应用》配套代码

本仓库是图书 **《视频异常事件检测原理及应用》**（李石峰，2024）的配套电子资源，
收录书中全部代码清单的可运行版本，并提供端到端演示与回归测试。

## 与书籍版本的对应关系

| 项目 | 内容 |
|---|---|
| 对应书版 | 2024 年版（正文 171 页） |
| 书中代码清单 | 共 7 处，编号 3.1–3.2、4.1、5.1、6.1、7.1–7.2 |
| 本仓库文件 | `code/` 下 7 个 `.py`，文件名即清单编号 |

**代码一致性保证**：`code/` 下的文件由 `E:\paper\monograph\_check\extract_companion_code.py`
从 LaTeX 源文件 `TeX_files/*.tex` 自动抽取，与书中印刷内容**逐字一致**。
若需修改，请改 LaTeX 源后重新抽取，**不要直接编辑本目录**，否则会与书籍脱节。

## 目录结构

```
.
├── README.md              本文件
├── requirements.txt       依赖与版本
├── manifest.json          清单编号 / 文件名 / 题注的机器可读对照
├── code/                  书中 7 处代码清单（与书逐字一致）
│   ├── 3.1_optical_flow.py           第3章  稠密光流计算
│   ├── 3.2_frame_loading.py          第3章  帧读取、缩放与归一化
│   ├── 4.1_mhof_chi_square.py         第4章  光流直方图 + 卡方距离异常分数
│   ├── 5.1_vae_loss.py               第5章  VAE 训练损失
│   ├── 6.1_fanogan_score.py          第6章  f-AnoGAN 异常分数
│   ├── 7.1_frame_metrics.py          第7章  帧级指标与事件合并
│   └── 7.2_seed_sensitivity.py       第7章  随机种子敏感性分析
├── demos/
│   └── run_demo.py         端到端演示（合成视频，无需数据集）
├── tests/
│   └── test_smoke.py       回归测试（含两处已修缺陷的守护用例）
└── LICENSE                 MIT
```

## 快速开始

```bash
pip install -r requirements.txt

python demos/run_demo.py      # 端到端演示，合成数据即可跑通
python -m pytest tests -q     # 回归测试
```

`run_demo.py` 的输出示意：

```
[1] 计算稠密光流（代码清单 3.1）
    正常段平均幅值 0.800，异常段平均幅值 1.150
[2] 正常模型与异常分数（代码清单 4.1）
    正常段分数均值 0.0799，异常段分数均值 0.2390
[3] 帧级评价指标（代码清单 7.1）
    帧级 AUC = 0.9872    AP = 0.9430
    按 90% 分位阈值检出 4 个事件区间：[(40, 40), (42, 42), (44, 44), (47, 48)]
```

> 演示使用合成数据，目的是说明**流程与接口**，**不是论文性能复现**。
> 真实评测需自行准备 UCF-Crime、XD-Violence 等公开数据集。

## 两处已修缺陷（回归测试守护）

整理配套代码时发现书中两段代码存在实际错误，已在 LaTeX 源中修正，
`tests/test_smoke.py` 中有对应用例守护，防止回退。

**1. 代码清单 4.1：光流直方图分箱下标错误**

原实现用 `hu + hv == k` 作为方向箱下标，且 `hu`、`hv` 均由运动**角度**计算。
两处问题叠加导致：

- 64 个方向箱中只有 15 个可达，**77% 的箱永远为空**；
- 不同运动方向被错误合并，例如 `(hu,hv)=(1,4)` 与 `(4,1)` 落入同一箱；
- 最严重的是**纯水平流与纯垂直流的直方图完全相同**，卡方距离为 `0`，
  即该"异常分数"无法区分二者，方法在数学上失效。

修正：下标改为由 `(hu, hv)` 唯一确定的 `hu * BIN_V + hv`，
并改为对**水平/垂直速度分量**分箱（而非角度），同时向量化统计。

守护用例：`test_mhof_separates_horizontal_and_vertical`、`test_mhof_uses_all_direction_bins`。

**2. 代码清单 6.1：f-AnoGAN 异常分数维度不匹配**

原实现中 `a_res` 形状为 `(N,)`，而 `a_feat = torch.norm(..., dim=1)` 仅归约通道维、
得到 `(N, H, W)`，两者相加直接抛出 `RuntimeError`。
该写法隐式假定特征提取器输出已全局池化的二维张量，一旦输出 `(N, C, H, W)` 即崩溃。

修正：先 `flatten(1)` 再按批维求范数，使 `(N, C, H, W)` 与 `(N, D)` 两种输出均可。

守护用例：`test_fanogan_score_is_zero_for_identity_reconstruction`、`test_fanogan_lambda_extremes`。

## 已知局限

- 第 3、7 章的部分代码在函数外直接执行，需要真实文件或模型；
  仓库中通过 `demos/` 提供合成数据使其可运行，但**不构成端到端训练/评测流水线**。
- 第 5、6 章仅覆盖 VAE 损失与 f-AnoGAN 打分两个片段，
  **尚未覆盖书中第 6.10 节（弱监督与视觉—语言模型）等当前主流范式**。
- 本仓库只做方法演示，**不含任何数据集**，也未复现论文报告的性能数字。

## 许可

代码采用 MIT 许可，见 [LICENSE](LICENSE)。书籍正文版权归作者所有。

## 发布到 GitHub / Gitee（维护者操作）

仓库已初始化 git 并完成首次提交，**尚未推送到任何远端**。推���前请先把下述占位地址
（`your-account`）替换为你自己的账号。

```bash
# 1. GitHub
git remote add github   https://github.com/<你的账号>/vad-book-code.git
git push -u github main

# 2. Gitee（国内可达，建议同时推送，书中前言已注明两个入口）
git remote add gitee    https://gitee.com/<你的账号>/vad-book-code.git
git push -u gitee main

# 3. 打标签，与书籍版本对应
git tag -a v1.0 -m "对应《视频异常事件检测原理及应用》2024 年版"
git push github v1.0 && git push gitee v1.0
```

推送后请回填书籍前言中的两个占位链接：

- `TeX_files/chapter-0-preface.tex` 中
  `\url{https://github.com/your-account/vad-book-code}` 与
  `\url{https://gitee.com/your-account/vad-book-code}`

修改后需重跑 LaTeX 编译与 DOCX 流水线（见书籍仓库 `AGENTS.md`）。

### 发布前自检

```bash
python -m pytest tests -q     # 应为 9 passed
python demos/run_demo.py      # 应能完整跑完 3 个阶段
```

### 版本维护约定

- 书籍正文修订后，须重新运行 `extract_companion_code.py` 同步 `code/`，
  再提交并更新标签；**不要手工编辑 `code/` 下的文件**，否则会与书中印刷内容脱节。
- 每处修复建议同时补一条回归测试，避免缺陷重现。

