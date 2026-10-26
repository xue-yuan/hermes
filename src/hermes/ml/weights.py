"""樣本權重：平均唯一性（AFML ch.4）。

標籤區間重疊時，樣本並非獨立；以每個樣本在其標籤區間內的平均「1/同時存在標籤數」作為權重，
避免模型被大量重疊樣本過度影響。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def concurrency(index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """每個交易日有多少個標籤區間同時涵蓋。"""
    pos = index.get_indexer(t1.index)
    end = index.get_indexer(t1.values)
    count = np.zeros(len(index) + 1)
    for s, e in zip(pos, end):
        if s < 0 or e < 0:
            continue
        count[s] += 1
        count[e + 1] -= 1
    return pd.Series(np.cumsum(count)[:-1], index=index)


def average_uniqueness(index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    conc = concurrency(index, t1).to_numpy()
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1), 0.0)
    csum = np.concatenate([[0.0], np.cumsum(inv)])
    pos = index.get_indexer(t1.index)
    end = index.get_indexer(t1.values)
    out = np.full(len(t1), np.nan)
    for k, (s, e) in enumerate(zip(pos, end)):
        if s < 0 or e < 0:
            continue
        out[k] = (csum[e + 1] - csum[s]) / (e - s + 1)
    return pd.Series(out, index=t1.index)


def sample_weights(
    index: pd.DatetimeIndex, t1: pd.Series, returns: pd.Series | None = None
) -> pd.Series:
    """平均唯一性權重（若給 returns，則依 AFML Snippet 4.10 以報酬歸因加權），正規化為平均 1。"""
    if returns is None:
        w = average_uniqueness(index, t1)
    else:
        conc = concurrency(index, t1)
        logret = np.log1p(returns.reindex(index).fillna(0.0)) / conc.replace(0, np.nan)
        csum = np.concatenate([[0.0], np.cumsum(logret.fillna(0.0).to_numpy())])
        pos = index.get_indexer(t1.index)
        end = index.get_indexer(t1.values)
        w = pd.Series(
            [abs(csum[e + 1] - csum[s]) for s, e in zip(pos, end)], index=t1.index
        )
    w = w.fillna(w.mean())
    return w * len(w) / w.sum()
