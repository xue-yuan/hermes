"""機器學習流程：特徵 → triple-barrier 標記 → 唯一性權重 → purged CV 評估 → walk-forward 樣本外預測 → 部位。

整體方法依循 López de Prado (2018) *Advances in Financial Machine Learning*：
- 標記：triple-barrier（ch.3）、meta-labeling（ch.3.6）
- 權重：平均唯一性（ch.4）
- 驗證：Purged K-Fold + embargo（ch.7）、CPCV（ch.12）
- 特徵重要性：樣本外 MDA（ch.8）
- 部位：由機率推導下注大小並平均仍有效的下注（ch.10）
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from hermes.ml.cv import CombinatorialPurgedKFold, PurgedKFold
from hermes.ml.labeling import (
    bet_size_from_prob,
    cusum_filter,
    daily_vol,
    triple_barrier,
)
from hermes.ml.models import make_model
from hermes.ml.weights import average_uniqueness, sample_weights


@dataclass
class MLConfig:
    model: str = "hgb"
    horizon: int = 10
    pt: float = 1.0
    sl: float = 1.0
    cusum: bool = False
    train_min: int = 1000
    step: int = 63
    max_train: int | None = None
    embargo_pct: float = 0.01
    n_splits: int = 5
    long_only: bool = True
    sizing: str = (
        "threshold"  # threshold：機率 > threshold 即持有；afml：2Φ(z)−1 下注大小
    )
    threshold: float = 0.5
    bet_step: float = 0.1
    average_bets: bool = True
    seed: int = 0
    seq_len: int = 60


@dataclass
class Dataset:
    X: pd.DataFrame  # 所有交易日的特徵
    events: pd.DataFrame  # triple-barrier 結果（index=事件日）
    y: pd.Series  # 二元標籤
    w: pd.Series  # 樣本權重
    side: pd.Series | None = None  # meta-labeling 的主要方向


@dataclass
class MLResult:
    config: MLConfig
    dataset: Dataset
    cv: pd.DataFrame = field(default_factory=pd.DataFrame)
    proba: pd.Series = field(default_factory=pd.Series)
    position: pd.Series = field(default_factory=pd.Series)
    importance: pd.Series | None = None
    latest: dict = field(default_factory=dict)


def make_dataset(
    prices: pd.DataFrame, X: pd.DataFrame, cfg: MLConfig, side: pd.Series | None = None
) -> Dataset:
    close = prices["close"]
    vol = daily_vol(close)
    events = None
    if cfg.cusum:
        events = cusum_filter(close, vol.rolling(50).mean())
    bars = triple_barrier(
        close,
        events=events,
        horizon=cfg.horizon,
        pt=cfg.pt,
        sl=cfg.sl,
        vol=vol,
        side=side,
    )
    # 特徵需在暖機期之後才有意義：至少一半欄位有值
    enough = X.notna().mean(axis=1) > 0.8
    bars = bars[bars.index.isin(X.index[enough])]
    y = (
        (bars["ret"] > 0).astype(int)
        if side is not None
        else (bars["label"] > 0).astype(int)
    )
    w = sample_weights(close.index, bars["t1"]) if len(bars) else pd.Series(dtype=float)
    return Dataset(X=X, events=bars, y=y, w=w, side=side)


def _fit(cfg: MLConfig, ds: Dataset, train_idx: pd.Index):
    Xtr = ds.X.loc[train_idx]
    u = average_uniqueness(ds.X.index, ds.events.loc[train_idx, "t1"]).mean()
    model = make_model(
        cfg.model, seed=cfg.seed, avg_uniqueness=u, context=ds.X, seq_len=cfg.seq_len
    )
    return model.fit(Xtr, ds.y.loc[train_idx], ds.w.loc[train_idx])


def purged_cv(ds: Dataset, cfg: MLConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Purged K-Fold 交叉驗證：回傳每折指標與樣本外機率。"""
    ev = ds.events[ds.events["complete"]]
    t1 = ev["t1"]
    cv = PurgedKFold(cfg.n_splits, cfg.embargo_pct)
    rows, oos = [], []
    for k, (tr, te) in enumerate(cv.split(t1, ds.X.index)):
        tr_idx, te_idx = t1.index[tr], t1.index[te]
        model = _fit(cfg, ds, tr_idx)
        p = model.predict_proba(ds.X.loc[te_idx])
        yt, wt = ds.y.loc[te_idx], ds.w.loc[te_idx]
        rows.append(_scores(k + 1, yt, p, wt, len(tr_idx)))
        oos.append(pd.Series(p, index=te_idx))
    return pd.DataFrame(rows), pd.concat(oos).sort_index() if oos else pd.Series(
        dtype=float
    )


def _scores(fold, y, p, w, n_train) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    two = y.nunique() > 1
    return {
        "fold": fold,
        "n_train": n_train,
        "n_test": len(y),
        "base_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, p > 0.5, sample_weight=w)),
        "auc": float(roc_auc_score(y, p, sample_weight=w)) if two else float("nan"),
        "log_loss": float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
        "f1": float(f1_score(y, p > 0.5, zero_division=0)),
    }


def latest_position(p: float, cfg: MLConfig, side: float | None = None) -> float:
    """單一機率的建議部位（不做跨日平均）。"""
    one = replace(cfg, average_bets=False, bet_step=0.0)
    return float(
        proba_to_position(
            pd.Series([p]), one, None if side is None else pd.Series([side])
        ).iloc[0]
    )


def walk_forward(ds: Dataset, cfg: MLConfig, progress=None) -> pd.Series:
    """每 step 個交易日重新訓練；只用標籤在測試區塊開始前已結束的事件訓練（purge）。

    回傳每個交易日（暖機期後）的樣本外機率。
    """
    idx = ds.X.index
    ev = ds.events
    t1 = ev["t1"]
    first_event = idx.get_loc(ev.index[0]) if len(ev) else len(idx)
    start = first_event + cfg.train_min
    proba = pd.Series(np.nan, index=idx)
    blocks = list(range(start, len(idx), cfg.step))
    for k, s in enumerate(blocks):
        block_start = idx[s]
        train_mask = (t1 < block_start).to_numpy()
        if cfg.max_train is not None:
            train_mask &= ev.index >= idx[max(0, s - cfg.max_train)]
        train_idx = ev.index[train_mask]
        if len(train_idx) < 100:
            continue
        model = _fit(cfg, ds, train_idx)
        test_dates = idx[s : s + cfg.step]
        proba.loc[test_dates] = model.predict_proba(ds.X.loc[test_dates])
        if progress:
            progress(k + 1, len(blocks))
    return proba


def proba_to_position(
    proba: pd.Series, cfg: MLConfig, side: pd.Series | None = None
) -> pd.Series:
    """機率 → 部位。

    - threshold：p > threshold 持有 1 單位（允許放空時 p < 1−threshold 放空），
      與 Fischer & Krauss (2018)、Krauss et al. (2017) 依預測方向建立部位的做法一致。
    - afml：López de Prado 的 2Φ(z)−1 下注大小；原設計針對大量同時進行的下注，單一股票時部位會偏小。
    """
    p = proba.dropna()
    if cfg.sizing == "afml":
        size = bet_size_from_prob(p)
    elif cfg.sizing == "threshold":
        size = (p > cfg.threshold).astype(float) - (p < 1 - cfg.threshold).astype(float)
    else:
        raise ValueError("sizing 必須是 threshold 或 afml")
    size = size.reindex(proba.index)
    if side is not None:
        # meta-labeling：機率 < 0.5 代表不交易，只決定大小不決定方向
        size = size.clip(lower=0.0) * side.reindex(proba.index).fillna(0.0)
    elif cfg.long_only:
        size = size.clip(lower=0.0)
    if cfg.average_bets:
        size = size.rolling(cfg.horizon, min_periods=1).mean()
    if cfg.bet_step > 0:
        size = (size / cfg.bet_step).round() * cfg.bet_step
    return size.where(proba.notna())


def fit_latest(ds: Dataset, cfg: MLConfig) -> tuple[dict, pd.Series | None]:
    """以所有已完成標籤訓練最終模型，預測最新一日。"""
    ev = ds.events[ds.events["complete"]]
    model = _fit(cfg, ds, ev.index)
    last = ds.X.index[-1]
    p = float(model.predict_proba(ds.X.loc[[last]])[0])
    imp = model.feature_importance() if hasattr(model, "feature_importance") else None
    if imp is not None:
        imp = imp.sort_values(ascending=False)
    return {"date": last, "proba": p}, imp


def mda_importance(
    ds: Dataset, cfg: MLConfig, n_splits: int = 4, seed: int = 0
) -> pd.Series:
    """樣本外 Mean Decrease Accuracy（以加權 log loss 增量衡量；AFML Snippet 8.3）。"""
    rng = np.random.default_rng(seed)
    ev = ds.events[ds.events["complete"]]
    t1 = ev["t1"]
    cv = PurgedKFold(n_splits, cfg.embargo_pct)
    cols = list(ds.X.columns)
    gains = pd.DataFrame(0.0, index=range(n_splits), columns=cols)
    for k, (tr, te) in enumerate(cv.split(t1, ds.X.index)):
        tr_idx, te_idx = t1.index[tr], t1.index[te]
        model = _fit(cfg, ds, tr_idx)
        Xte = ds.X.loc[te_idx]
        yt, wt = ds.y.loc[te_idx], ds.w.loc[te_idx]
        base = log_loss(
            yt,
            np.clip(model.predict_proba(Xte), 1e-6, 1 - 1e-6),
            sample_weight=wt,
            labels=[0, 1],
        )
        for c in cols:
            Xp = Xte.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            ll = log_loss(
                yt,
                np.clip(model.predict_proba(Xp), 1e-6, 1 - 1e-6),
                sample_weight=wt,
                labels=[0, 1],
            )
            gains.loc[k, c] = ll - base
    return gains.mean().sort_values(ascending=False)


def cpcv_paths(
    ds: Dataset, cfg: MLConfig, n_groups: int = 6, n_test_groups: int = 2
) -> list[pd.Series]:
    """CPCV：回傳 φ 條樣本外機率路徑（只在事件日有值）。"""
    ev = ds.events[ds.events["complete"]]
    t1 = ev["t1"]
    cv = CombinatorialPurgedKFold(n_groups, n_test_groups, cfg.embargo_pct)
    preds = []
    for tr, te, combo in cv.split(t1, ds.X.index):
        model = _fit(cfg, ds, t1.index[tr])
        te_idx = t1.index[te]
        preds.append(
            (combo, pd.Series(model.predict_proba(ds.X.loc[te_idx]), index=te_idx))
        )
    return cv.assemble_paths(preds, t1)
