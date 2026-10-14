# -*- coding: utf-8 -*-
"""Smoke tests for the companion code.

These run on synthetic data only -- no dataset download required -- so that
`pytest` works out of the box for anyone who clones the repository.

The MHOF tests are regression tests: an earlier draft binned the flow
histogram with `hu + hv`, which merged distinct directions and left 77% of the
bins unreachable. Horizontal and vertical motion then produced an identical
histogram, i.e. a chi-square anomaly score of exactly 0. These assertions fail
if that bug is ever reintroduced.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))


def load(module_file):
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), '..', 'code', module_file)
    spec = importlib.util.spec_from_file_location(module_file[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hist_from_flow(mod, u, v, mag):
    """Mirror of flow_histogram()'s binning, without needing real frames."""
    BIN_U, BIN_V, BIN_N = mod.BIN_U, mod.BIN_V, mod.BIN_N
    U_SCALE, V_SCALE = mod.U_SCALE, mod.V_SCALE
    hu = np.clip((u / U_SCALE * (BIN_U - 1) + BIN_U / 2).astype(int), 0, BIN_U - 1)
    hv = np.clip((v / V_SCALE * (BIN_V - 1) + BIN_V / 2).astype(int), 0, BIN_V - 1)
    hn = np.clip((mag / 10.0 * (BIN_N - 1)).astype(int), 0, BIN_N - 1)
    idx = hn * (BIN_U * BIN_V) + hu * BIN_V + hv
    hist = np.bincount(idx.ravel(),
                       minlength=BIN_N * BIN_U * BIN_V).reshape(BIN_N, BIN_U * BIN_V)
    return hist / max(hist.sum(), 1e-8)


# --------------------------------------------------------------- 4.1 MHOF
def test_mhof_separates_horizontal_and_vertical():
    mhof = load('4.1_mhof_chi_square.py')
    S = (64, 64)
    h_flow = (np.full(S, 4.0), np.zeros(S))
    v_flow = (np.zeros(S), np.full(S, 4.0))
    Hh = _hist_from_flow(mhof, *h_flow, np.abs(h_flow[0]))
    Hv = _hist_from_flow(mhof, *v_flow, np.abs(v_flow[0]))
    assert Hh.sum() == pytest.approx(1.0, abs=1e-6)
    assert Hv.sum() == pytest.approx(1.0, abs=1e-6)
    d = mhof.chi_square_distance(Hh, Hv)
    assert d > 1e-6, '水平流与垂直流的直方图不应完全相同（回归：hu+hv 分箱错误）'


def test_mhof_uses_all_direction_bins():
    """每个 (hu, hv) 组合都应可达，即下标由 hu * BIN_V + hv 唯一确定。"""
    mhof = load('4.1_mhof_chi_square.py')
    S = (64, 64)
    used = set()
    for hu in range(mhof.BIN_U):
        for hv in range(mhof.BIN_V):
            u = np.full(S, (hu - mhof.BIN_U / 2) * mhof.U_SCALE / (mhof.BIN_U - 1))
            v = np.full(S, (hv - mhof.BIN_V / 2) * mhof.V_SCALE / (mhof.BIN_V - 1))
            mag = np.sqrt(u ** 2 + v ** 2)
            H = _hist_from_flow(mhof, u, v, mag)
            used.add(int(np.argmax(H.sum(axis=0))))
    assert len(used) == mhof.BIN_U * mhof.BIN_V, \
        '只有 %d/%d 个方向箱可达，说明下标退化了' % (len(used), mhof.BIN_U * mhof.BIN_V)


def test_mhof_chi_square_properties():
    mhof = load('4.1_mhof_chi_square.py')
    rng = np.random.default_rng(0)
    S = (64, 64)
    p = _hist_from_flow(mhof, rng.normal(0, 2, S), rng.normal(0, 2, S),
                        np.ones(S))
    assert mhof.chi_square_distance(p, p) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------- 5.1 VAE
def test_vae_loss_shapes_and_nonnegative():
    torch = pytest.importorskip('torch')
    vae = load('5.1_vae_loss.py')
    torch.manual_seed(0)
    x = torch.randn(4, 3, 8, 8)
    recon = x + 0.1 * torch.randn_like(x)
    mu = torch.zeros(4, 2)
    log_var = torch.zeros(4, 2)
    total, rec, kl = vae.vae_loss(x, recon, mu, log_var)
    assert total.shape == () and rec >= 0 and kl >= 0
    assert torch.isfinite(total)


def test_vae_kl_is_zero_for_standard_normal_posterior():
    torch = pytest.importorskip('torch')
    vae = load('5.1_vae_loss.py')
    torch.manual_seed(0)
    x = torch.randn(4, 3, 8, 8)
    _, _, kl = vae.vae_loss(x, x.clone(), torch.zeros(4, 2), torch.zeros(4, 2))
    assert kl == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------- 6.2 f-AnoGAN
def test_fanogan_score_is_zero_for_identity_reconstruction():
    torch = pytest.importorskip('torch')
    fg = load('6.2_fanogan_score.py')

    class Id(torch.nn.Module):
        def forward(self, z):
            return z

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Conv2d(3, 3, 1)

        def forward(self, x):
            return self.f(x)

    torch.manual_seed(0)
    enc, gen, feat = Enc(), Id(), torch.nn.Conv2d(3, 3, 1)
    with torch.no_grad():
        feat.weight.fill_(0.1)
        x = torch.randn(2, 3, 8, 8)
        s = fg.fanogan_score(x, enc, gen, feat, lam=0.9)
    assert s.shape == (2,)
    assert torch.all(torch.isfinite(s))


def test_fanogan_lambda_extremes():
    torch = pytest.importorskip('torch')
    fg = load('6.2_fanogan_score.py')

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Conv2d(3, 3, 1)

        def forward(self, x):
            return self.f(x)

    torch.manual_seed(0)
    enc, feat = Enc(), torch.nn.Conv2d(3, 3, 1)
    with torch.no_grad():
        feat.weight.fill_(0.1)
        x = torch.randn(2, 3, 8, 8)
        s_low = fg.fanogan_score(x, enc, torch.nn.Identity(), feat, lam=0.0)
        s_high = fg.fanogan_score(x, enc, torch.nn.Identity(), feat, lam=1.0)
    assert s_low.shape == (2,) and s_high.shape == (2,)
    assert torch.all(s_low >= 0) and torch.all(s_high >= 0)


# --------------------------------------------------------------- 7.1 metrics
def test_frame_level_metrics_and_event_merge():
    ft = load('7.1_frame_metrics.py')
    labels = [0, 0, 1, 1, 0, 1, 1, 0]
    scores = [0.1, 0.2, 0.9, 0.8, 0.15, 0.95, 0.85, 0.05]
    m = ft.frame_level_metrics(labels, scores)
    assert 0.0 <= m['AUC'] <= 1.0 and 0.0 <= m['AP'] <= 1.0

    events = ft.merge_events([0, 1, 1, 0, 0, 1, 0])
    assert events == [(1, 2), (5, 5)]
    assert ft.merge_events([0, 0, 0]) == []
    assert ft.merge_events([1, 1, 1]) == [(0, 2)]


# --------------------------------------------------------------- 3.x / 7.2 IO
def test_frame_listing_and_optical_flow_on_synthetic_frames():
    """3.1/3.2/7.2 在顶层直接执行，这里用合成帧验证其计算路径。"""
    cv2 = pytest.importorskip('cv2')
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    shifted = np.roll(base, 3, axis=1)          # 纯水平位移
    g1 = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    assert flow.shape == g1.shape + (2,)
    u = flow[..., 0]
    assert u.mean() == pytest.approx(3.0, abs=0.5), '向右平移 3 像素，u 均值应约为 3'


# ---------------------------------------------------------- 6.1 双流重构
def test_dualstream_fusion_score_matches_formula():
    torch = pytest.importorskip('torch')
    ds = load('6.1_dualstream_reconstruct.py')
    torch.manual_seed(0)
    x, x_hat = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
    d, d_hat = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
    s = ds.fusion_score(x, x_hat, d, d_hat, lam=0.5)
    app = (x - x_hat).pow(2).flatten(1).sum(1)
    mot = (d - d_hat).pow(2).flatten(1).sum(1)
    expected = 0.5 * app + 0.5 * mot
    assert s.shape == (2,)
    assert torch.allclose(s, expected, rtol=1e-5)
    assert torch.all(s >= 0)


def test_dualstream_lambda_extremes_select_single_stream():
    torch = pytest.importorskip('torch')
    ds = load('6.1_dualstream_reconstruct.py')
    torch.manual_seed(0)
    x, x_hat = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
    d, d_hat = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
    app = (x - x_hat).pow(2).flatten(1).sum(1)
    mot = (d - d_hat).pow(2).flatten(1).sum(1)
    assert torch.allclose(ds.fusion_score(x, x_hat, d, d_hat, lam=0.0), mot, rtol=1e-5)
    assert torch.allclose(ds.fusion_score(x, x_hat, d, d_hat, lam=1.0), app, rtol=1e-5)


def test_dualstream_motion_repr_shrinks_time_axis():
    torch = pytest.importorskip('torch')
    ds = load('6.1_dualstream_reconstruct.py')
    d = ds.motion_repr(torch.randn(2, 5, 3, 8, 8))
    assert d.shape == (2, 4, 3, 8, 8)


# ---------------------------------------------------------- 6.3 记忆库
def test_memory_bank_member_scores_zero():
    torch = pytest.importorskip('torch')
    mb = load('6.3_memory_bank.py')
    torch.manual_seed(0)
    feats = torch.randn(50, 16)
    bank = mb.MemoryBank(feats)
    s = bank.score(feats)
    assert s.shape == (50,)
    assert torch.allclose(s, torch.zeros_like(s), atol=1e-5), '库内样本自身即最近邻，分数应为 0'


def test_memory_bank_scores_far_point_higher():
    torch = pytest.importorskip('torch')
    mb = load('6.3_memory_bank.py')
    torch.manual_seed(0)
    feats = torch.randn(50, 16)
    bank = mb.MemoryBank(feats)
    far = torch.randn(1, 16) * 20.0
    assert bank.score(far).item() > bank.score(feats[:1]).item()


def test_coreset_reduces_to_requested_prototype_count():
    torch = pytest.importorskip('torch')
    mb = load('6.3_memory_bank.py')
    torch.manual_seed(0)
    feats = torch.randn(200, 16)
    bank = mb.MemoryBank(feats, n_prototypes=10, seed=0)
    assert bank.memory.shape[0] == 10


# ---------------------------------------------------------- 6.4 RTFM
def test_rtfm_loss_zero_when_margin_already_satisfied():
    torch = pytest.importorskip('torch')
    rt = load('6.4_rtfm.py')
    loss = rt.rtfm_loss(torch.full((4,), 0.1), torch.full((4,), 1.0), margin=0.5)
    assert loss.item() == pytest.approx(0.0), '正常幅值已比异常低出 margin，损失应为 0'


def test_rtfm_loss_equals_slack_when_margin_unmet():
    torch = pytest.importorskip('torch')
    rt = load('6.4_rtfm.py')
    loss = rt.rtfm_loss(torch.tensor([1.0]), torch.tensor([1.0]), margin=0.5)
    assert loss.item() == pytest.approx(0.5)


def test_feature_magnitude_is_l2_norm_over_time_and_dim():
    torch = pytest.importorskip('torch')
    rt = load('6.4_rtfm.py')
    mag = rt.feature_magnitude(torch.ones(3, 2, 4))
    assert mag.shape == (3,)
    assert mag[0].item() == pytest.approx(8 ** 0.5)
