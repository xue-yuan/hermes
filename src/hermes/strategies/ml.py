"""機器學習策略：皆為 walk-forward 樣本外預測，不會使用未來資料訓練。"""

from __future__ import annotations

import pandas as pd

from hermes.ml.features import build_features
from hermes.ml.pipeline import MLConfig, make_dataset, proba_to_position, walk_forward
from hermes.strategies.base import StrategyContext, register


def _features(
    ctx: StrategyContext, exogenous: bool, fracdiff: float, changepoint: bool
) -> pd.DataFrame:
    extra = ctx.exogenous() if exogenous else None
    return build_features(
        ctx.prices,
        extra,
        fracdiff_d=fracdiff if fracdiff > 0 else None,
        changepoint=changepoint,
    )


@register(
    "ml",
    "機器學習方向預測",
    "機器學習",
    "以 triple-barrier 標籤訓練分類器（logit/rf/et/hgb/mlp/lstm/ensemble），"
    "walk-forward 樣本外機率經 bet sizing 轉成部位（López de Prado 2018）。",
    model="hgb",
    horizon=10,
    pt=1.0,
    sl=1.0,
    train_min=1000,
    step=63,
    sizing="threshold",
    threshold=0.5,
    exogenous=True,
    fracdiff=0.4,
    changepoint=True,
    seed=0,
)
def ml_strategy(
    ctx,
    model,
    horizon,
    pt,
    sl,
    train_min,
    step,
    sizing,
    threshold,
    exogenous,
    fracdiff,
    changepoint,
    seed,
):
    cfg = MLConfig(
        model=model,
        horizon=horizon,
        pt=pt,
        sl=sl,
        train_min=train_min,
        step=step,
        seed=seed,
        sizing=sizing,
        threshold=threshold,
    )
    X = _features(ctx, exogenous, fracdiff, changepoint)
    ds = make_dataset(ctx.prices, X, cfg)
    proba = walk_forward(ds, cfg, progress=ctx.progress)
    return proba_to_position(proba, cfg)


@register(
    "meta",
    "Meta-labeling",
    "機器學習",
    "主要策略（primary，strategies/ 裡的設定檔）決定方向，次級模型預測該訊號是否會獲利並決定部位大小"
    "（López de Prado 2018, ch.3.6）。",
    primary=None,
    model="rf",
    horizon=10,
    pt=1.0,
    sl=1.0,
    train_min=750,
    step=63,
    sizing="threshold",
    threshold=0.5,
    exogenous=True,
    fracdiff=0.4,
    changepoint=True,
    seed=0,
)
def meta_strategy(
    ctx,
    primary,
    model,
    horizon,
    pt,
    sl,
    train_min,
    step,
    sizing,
    threshold,
    exogenous,
    fracdiff,
    changepoint,
    seed,
):
    if not primary or ctx.rule_position is None:
        raise ValueError(
            "meta 要指定主要策略：-p primary=策略設定檔名稱（strategies/ 裡的檔名）"
        )
    side = (
        ctx.rule_position(str(primary))
        .fillna(0.0)
        .apply(lambda v: 1.0 if v > 0 else (-1.0 if v < 0 else 0.0))
    )
    cfg = MLConfig(
        model=model,
        horizon=horizon,
        pt=pt,
        sl=sl,
        train_min=train_min,
        step=step,
        seed=seed,
        sizing=sizing,
        threshold=threshold,
    )
    X = _features(ctx, exogenous, fracdiff, changepoint)
    ds = make_dataset(ctx.prices, X, cfg, side=side)
    proba = walk_forward(ds, cfg, progress=ctx.progress)
    return proba_to_position(proba, cfg, side=side)


@register(
    "dmn",
    "Deep Momentum Network",
    "機器學習",
    "Lim, Zohren & Roberts (2019)：LSTM 直接最大化 Sharpe ratio，輸出部位並以波動度目標化；"
    "可加入 Wood et al. (2021) 的變點特徵。需要 torch。",
    hidden=20,
    seq_len=63,
    dropout=0.2,
    vol_target=0.20,
    cost_bps=10.0,
    changepoint=True,
    long_only=True,
    seeds=1,
    train_min=1260,
    step=252,
)
def dmn_strategy(
    ctx,
    hidden,
    seq_len,
    dropout,
    vol_target,
    cost_bps,
    changepoint,
    long_only,
    seeds,
    train_min,
    step,
):
    from hermes.ml.deep import DMNConfig, dmn_walk_forward

    cfg = DMNConfig(
        hidden=hidden,
        seq_len=seq_len,
        dropout=dropout,
        vol_target=vol_target,
        cost_bps=cost_bps,
        use_changepoint=changepoint,
        long_only=long_only,
        seeds=seeds,
    )
    out = dmn_walk_forward(
        ctx.prices, cfg, train_min=train_min, step=step, progress=ctx.progress
    )
    return out["position"]


@register(
    "hmm",
    "HMM 市場狀態",
    "機器學習",
    "高斯隱馬可夫模型辨識市場狀態（Hamilton 1989），以前向濾波的「平均報酬最高狀態」機率決定持股。",
    states=2,
    train_min=1000,
    step=63,
    threshold=0.5,
    continuous=False,
    with_vol=False,
)
def hmm_strategy(ctx, states, train_min, step, threshold, continuous, with_vol):
    from hermes.ml.regime import hmm_walk_forward

    out = hmm_walk_forward(
        ctx.prices, n_states=states, train_min=train_min, step=step, with_vol=with_vol
    )
    p = out["p_bull"]
    pos = p if continuous else (p > threshold).astype(float).where(p.notna())
    return pos
