from math import comb

import numpy as np
import pandas as pd
import pytest

from hermes.ml import pipeline
from hermes.ml.changepoint import changepoint_features
from hermes.ml.cv import CombinatorialPurgedKFold, PurgedKFold
from hermes.ml.features import build_features
from hermes.ml.fracdiff import ffd_weights, frac_diff
from hermes.ml.labeling import bet_size_from_prob, cusum_filter, triple_barrier
from hermes.ml.regime import GaussianHMM
from hermes.ml.weights import average_uniqueness


def test_triple_barrier_labels():
    idx = pd.bdate_range("2020-01-01", periods=40)
    up = pd.Series(np.linspace(100, 140, 40), index=idx)
    vol = pd.Series(0.01, index=idx)
    out = triple_barrier(up, events=idx[:5], horizon=10, vol=vol)
    assert (out["barrier"] == 1).all() and (out["label"] == 1).all()
    down = pd.Series(np.linspace(140, 100, 40), index=idx)
    out = triple_barrier(down, events=idx[:5], horizon=10, vol=vol)
    assert (out["barrier"] == -1).all()
    # 做空方向的 meta-label：下跌時獲利
    side = pd.Series(-1.0, index=idx)
    out = triple_barrier(down, events=idx[:5], horizon=10, vol=vol, side=side)
    assert (out["ret"] > 0).all()


def test_triple_barrier_t1_after_event(prices):
    out = triple_barrier(prices["close"], horizon=10)
    assert (out["t1"] > out.index).all()
    pos = prices.index.get_indexer(out.index)
    end = prices.index.get_indexer(out["t1"])
    assert ((end - pos) <= 10).all()


def test_cusum_filter_sparse(prices):
    ev = cusum_filter(prices["close"], 0.05)
    assert 0 < len(ev) < len(prices) / 2


def test_uniqueness():
    idx = pd.bdate_range("2020-01-01", periods=10)
    t1 = pd.Series([idx[1], idx[3]], index=[idx[0], idx[2]])  # 不重疊
    assert np.allclose(average_uniqueness(idx, t1), 1.0)
    t1 = pd.Series([idx[3], idx[4]], index=[idx[0], idx[1]])  # 重疊三天
    u = average_uniqueness(idx, t1)
    assert u.iloc[0] == pytest.approx((1 + 0.5 + 0.5 + 0.5) / 4)


def _t1(n=300, h=5):
    idx = pd.bdate_range("2020-01-01", periods=n + h)
    ev = idx[:n]
    return pd.Series(idx[np.arange(n) + h], index=ev), idx


def test_purged_kfold_has_no_overlap():
    t1, idx = _t1()
    cv = PurgedKFold(5, embargo_pct=0.02)
    for train, test in cv.split(t1, idx):
        t_start, t_end = t1.index[test].min(), t1.iloc[test].max()
        tr = t1.iloc[train]
        overlap = (tr.index <= t_end) & (tr.values >= t_start)
        assert not overlap.any()
        assert len(np.intersect1d(train, test)) == 0


def test_cpcv_paths_cover_all_samples():
    t1, idx = _t1(240)
    cv = CombinatorialPurgedKFold(6, 2, embargo_pct=0.0)
    splits = list(cv.split(t1, idx))
    assert len(splits) == comb(6, 2)
    preds = [(combo, pd.Series(1.0, index=t1.index[test])) for _, test, combo in splits]
    paths = cv.assemble_paths(preds, t1)
    assert len(paths) == cv.n_paths == 5
    for p in paths:
        assert p.index.equals(t1.index) and p.notna().all()


def test_fracdiff_limits():
    s = pd.Series(np.cumsum(np.random.default_rng(0).normal(size=300)))
    assert np.allclose(frac_diff(s, 0.0), s)
    assert np.allclose(frac_diff(s, 1.0).dropna(), s.diff().dropna())
    w = ffd_weights(0.4)
    assert w[-1] == 1.0 and len(w) > 10


def test_bet_size_symmetric():
    p = pd.Series([0.1, 0.5, 0.9])
    s = bet_size_from_prob(p)
    assert s.iloc[1] == pytest.approx(0.0)
    assert s.iloc[0] == pytest.approx(-s.iloc[2])


def test_hmm_recovers_regimes():
    rng = np.random.default_rng(3)
    states = np.repeat([0, 1, 0, 1], 400)
    x = np.where(states == 1, rng.normal(1.0, 0.5, len(states)), rng.normal(-1.0, 1.5, len(states)))
    hmm = GaussianHMM(2).fit(x)
    hmm.sort_states(hmm.means_[:, 0])
    assert hmm.means_[0, 0] == pytest.approx(-1.0, abs=0.25)
    assert hmm.means_[1, 0] == pytest.approx(1.0, abs=0.15)
    acc = ((hmm.filter(x)[:, 1] > 0.5) == (states == 1)).mean()
    assert acc > 0.9


def test_bocpd_detects_shift():
    idx = pd.bdate_range("2018-01-01", periods=600)
    rng = np.random.default_rng(4)
    r = np.r_[rng.normal(0, 0.01, 400), rng.normal(0, 0.05, 200)]
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    cp = changepoint_features(close)
    before = cp["cp_severity"].iloc[300:395].mean()
    after = cp["cp_severity"].iloc[400:420].max()
    assert after > before + 0.3


def test_features_have_no_lookahead(prices):
    full = build_features(prices)
    part = build_features(prices.iloc[:500])
    a, b = full.iloc[499], part.iloc[-1]
    both = a.notna() & b.notna()
    assert np.allclose(a[both], b[both], rtol=1e-6, atol=1e-9)


def test_walk_forward_trains_only_on_resolved_labels(prices, monkeypatch):
    X = build_features(prices, changepoint=False)
    cfg = pipeline.MLConfig(model="logit", train_min=300, step=100, horizon=5)
    ds = pipeline.make_dataset(prices, X, cfg)
    seen = []

    class Recorder:
        def fit(self, Xtr, y, w=None):
            self.last = Xtr.index.max()
            seen.append(Xtr.index)
            return self

        def predict_proba(self, Xte):
            t1_max = ds.events.loc[seen[-1], "t1"].max()
            assert t1_max < Xte.index.min(), "訓練標籤延伸到測試期間"
            return np.full(len(Xte), 0.6)

    monkeypatch.setattr(pipeline, "make_model", lambda *a, **k: Recorder())
    proba = pipeline.walk_forward(ds, cfg)
    assert proba.notna().sum() > 0
    first = proba.first_valid_index()
    assert prices.index.get_loc(first) >= prices.index.get_loc(ds.events.index[0]) + cfg.train_min
    pos = pipeline.proba_to_position(proba, cfg)
    assert set(pos.dropna().unique()) <= {0.0, 1.0}
