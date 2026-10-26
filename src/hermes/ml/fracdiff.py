"""分數差分（Fractional Differentiation, AFML ch.5；Hosking 1981）。

整數差分（報酬率）雖然平穩，卻抹去了價格的長期記憶；分數差分以 0<d<1 的階數，
在「通過 ADF 平穩性檢定」的前提下保留最多記憶，適合作為機器學習特徵。
這裡實作固定窗寬版本（FFD）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ffd_weights(d: float, threshold: float = 1e-4, max_size: int = 2000) -> np.ndarray:
    """w_k = −w_{k−1}·(d−k+1)/k，直到 |w_k| < threshold。回傳由舊到新的權重。"""
    w = [1.0]
    k = 1
    while k < max_size:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])


def frac_diff(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    w = ffd_weights(d, threshold)
    width = len(w)
    x = series.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= width:
        conv = np.convolve(x, w[::-1], mode="valid")
        out[width - 1 :] = conv
    return pd.Series(out, index=series.index)


def min_ffd_order(
    series: pd.Series,
    pvalue: float = 0.05,
    threshold: float = 1e-4,
    grid: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """找出能通過 ADF 檢定（p < pvalue）的最小 d。需要 statsmodels。"""
    from statsmodels.tsa.stattools import adfuller

    grid = np.linspace(0.0, 1.0, 11) if grid is None else grid
    rows = []
    best = 1.0
    for d in grid:
        fd = frac_diff(series, d, threshold).dropna()
        if len(fd) < 100:
            continue
        stat, p, *_ = adfuller(fd, maxlag=1, regression="c", autolag=None)
        corr = float(np.corrcoef(series.loc[fd.index], fd)[0, 1])
        rows.append((d, stat, p, corr))
        if p < pvalue and d < best:
            best = d
    return float(best), pd.DataFrame(rows, columns=["d", "adf_stat", "pvalue", "corr"])
