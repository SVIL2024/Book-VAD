# -*- coding: utf-8 -*-
# 配套代码 6.3 —— 《视频异常检测原理及应用》第6章 深度学习视频异常检测方法
# 题注：记忆库 Coreset 抽样与最近邻异常分数示例
# 本文件由 _check/extract_companion_code.py 从 code_src/6.2_memory_bank.py 自动抽取，
# 是书中代码清单 6.3 的完整可运行实现（书中仅摘录第 42-62 行的核心片段）。
# 请勿直接在此修改，需改请改 code_src/6.2_memory_bank.py 后重新抽取。

# -*- coding: utf-8 -*-
"""基于预训练特征的记忆库与最近邻异常分数（对应正文 6.3.4 节 VideoPatchCore）。

方法不训练任何参数：冻结的预训练编码器 E 提取通用特征，训练阶段通过
贪心 Coreset 抽样把记忆库压缩为最具代表性的原型子集，推理阶段以测试
特征到记忆库的最近邻距离作为异常分数：

    A(x) = min_{m_i in M} || E(x) - m_i ||_2

书中称此为三重记忆模块（空间/目标/时间），完整实现可直接运行：
python 6.2_memory_bank.py
"""
import torch


def coreset_greedy(features, n_prototypes, proj_dim=128, seed=0, device='cpu'):
    """贪心 Coreset 抽样：从 features 中选出 n_prototypes 个最具代表性的原型。

    每一步挑选"到已选集合最远"的样本，等价于近似最大化覆盖半径，
    从而用少量原型代表整个正常特征分布。
    """
    x = features.to(device).flatten(1).float()
    # 随机投影降维，O(N*k*d) -> O(N*k*proj_dim)，大特征集上必需
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(x.shape[1], proj_dim, generator=g).to(device)
    z = x @ p
    z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-8)

    idx = torch.empty(n_prototypes, dtype=torch.long)
    dist = torch.full((z.shape[0],), float('inf'), device=device)
    for i in range(n_prototypes):
        if i == 0:
            j = int(torch.argmax(z.norm(dim=1)))
        else:
            j = int(torch.argmax(dist))
        idx[i] = j
        # 距离度量用 1 - 余弦相似度，与预训练特征的语义空间一致
        dist = torch.minimum(dist, 1.0 - z @ z[j])
    return features[idx.to(features.device)]


class MemoryBank:
    """最近邻异常打分器。

    scores 越大越可能为异常。使用余弦距离（等价于归一化后的欧氏最近邻）
    而非原始欧氏距离，因为 CLIP 等对比预训练特征的方向才是语义主轴。
    """

    def __init__(self, features, n_prototypes=None, proj_dim=128, seed=0):
        if n_prototypes is None or n_prototypes >= features.shape[0]:
            self.memory = features.flatten(1).float()
        else:
            self.memory = coreset_greedy(features, n_prototypes, proj_dim, seed)
        self.memory = self.memory / self.memory.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def score(self, features):
        """返回形状 (B,) 的最近邻距离，即正文中的 A(x)。"""
        z = features.flatten(1).float()
        z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cos = z @ self.memory.t()
        return 1.0 - cos.max(dim=1).values

    @torch.no_grad()
    def update(self, features, keep_ratio=1.0):
        """可更新记忆：把高置信正常样本并入记忆库（正文 6.3.2 节 MNAD 式设定）。"""
        new = features.flatten(1).float()
        new = new / new.norm(dim=1, keepdim=True).clamp_min(1e-8)
        merged = torch.cat([self.memory, new], dim=0)
        if keep_ratio < 1.0:
            k = max(1, int(merged.shape[0] * keep_ratio))
            merged = coreset_greedy(merged, k)
            merged = merged / merged.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.memory = merged
        return self.memory.shape[0]


def _synthetic_features(n, dim=64, seed=0, spread=0.15):
    """合成特征：紧致簇代表正常模式，方向上的偏移代表异常。"""
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(dim, generator=g)
    c = c / c.norm()
    return c.unsqueeze(0) + spread * torch.randn(n, dim, generator=g)


if __name__ == '__main__':
    torch.manual_seed(0)
    dim = 64

    normal = _synthetic_features(400, dim, seed=0)
    bank = MemoryBank(normal, n_prototypes=64, seed=0)
    print('记忆库原型数      : %d / %d (Coreset 压缩)' % (bank.memory.shape[0], normal.shape[0]))

    # 库内样本：最近邻距离应接近 0
    probe = _synthetic_features(32, dim, seed=0)
    s_in = bank.score(probe)

    # 库外异常：沿正交方向偏移，与正常簇方向差异大
    g = torch.Generator().manual_seed(1)
    ortho = torch.randn(dim, generator=g)
    ortho = ortho - (ortho @ normal[0] / (normal[0] @ normal[0])) * normal[0]
    ortho = ortho / ortho.norm()
    s_out = bank.score(ortho.unsqueeze(0).repeat(32, 1))

    print('库内样本平均距离  : %.4f' % s_in.mean().item())
    print('库外异常平均距离  : %.4f' % s_out.mean().item())
    assert s_in.min().item() >= -1e-5, '最近邻距离不应为负'
    assert s_out.mean().item() > s_in.mean().item() * 2.0, '库外异常应明显更远'
    print('PASS 记忆库：库外异常距离为库内的 %.1f 倍'
          % (s_out.mean().item() / max(s_in.mean().item(), 1e-8)))
