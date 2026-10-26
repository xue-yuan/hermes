"""標記方法（López de Prado, 2018, *Advances in Financial Machine Learning*, ch.2–3）。

- CUSUM filter：只在累積變動超過門檻時產生事件，減少高度重疊的樣本。
- Triple-barrier：以停利、停損、到期三道柵欄中最先觸及者決定標籤。
- Meta-labeling：給定主要模型的方向（side），標記該筆交易是否獲利，讓次級模型學習「要不要下注、下多少」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_vol(close: pd.Series, span: int = 50) -> pd.Series:
    """以 EWM 標準差估計日報酬波動度。"""
    return np.log(close).diff().ewm(span=span, min_periods=span // 2).std()


def cusum_filter(close: pd.Series, threshold: float | pd.Series) -> pd.DatetimeIndex:
    """對稱 CUSUM 濾波器（AFML Snippet 2.4）。threshold 可為常數或與 close 對齊的序列。"""
    r = np.log(close).diff().dropna()
    thr = (
        threshold.reindex(r.index)
        if isinstance(threshold, pd.Series)
        else pd.Series(threshold, index=r.index)
    )
    events = []
    s_pos = s_neg = 0.0
    for t, x in r.items():
        h = thr.loc[t]
        if not np.isfinite(h):
            continue
        s_pos = max(0.0, s_pos + x)
        s_neg = min(0.0, s_neg + x)
        if s_neg < -h:
            s_neg = 0.0
            events.append(t)
        elif s_pos > h:
            s_pos = 0.0
            events.append(t)
    return pd.DatetimeIndex(events)


def triple_barrier(
    close: pd.Series,
    events: pd.DatetimeIndex | None = None,
    horizon: int = 10,
    pt: float = 1.0,
    sl: float = 1.0,
    vol: pd.Series | None = None,
    side: pd.Series | None = None,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """Triple-barrier 標記。

    柵欄寬度 = 日波動度 × √horizon × 倍數（pt、sl）；垂直柵欄為 horizon 個交易日後。
    回傳欄位：t1（標籤結束日）、ret（持有報酬，若有 side 則乘上方向）、
    barrier（+1 停利、-1 停損、0 到期）、label（+1/-1；到期時取報酬正負號）。
    """
    close = close.dropna()
    if vol is None:
        vol = daily_vol(close)
    if events is None:
        events = close.index
    events = events.intersection(vol.dropna().index)
    if side is not None:
        side = side.reindex(events)
        events = events[(side.fillna(0) != 0).to_numpy()]

    idx = close.index
    pos = idx.get_indexer(events)
    values = close.to_numpy()
    vols = vol.reindex(events).to_numpy()
    rows = []
    for e, i, v in zip(events, pos, vols):
        target = v * np.sqrt(horizon)
        if not np.isfinite(target) or target <= min_ret:
            continue
        end = min(i + horizon, len(idx) - 1)
        if end <= i:
            continue
        s = 1.0 if side is None else float(side.loc[e])
        path = values[i + 1 : end + 1] / values[i] - 1.0
        path = path * s
        upper = pt * target if pt > 0 else np.inf
        lower = -sl * target if sl > 0 else -np.inf
        hit_up = np.flatnonzero(path >= upper)
        hit_dn = np.flatnonzero(path <= lower)
        first_up = hit_up[0] if hit_up.size else np.inf
        first_dn = hit_dn[0] if hit_dn.size else np.inf
        if first_up == np.inf and first_dn == np.inf:
            k, barrier = len(path) - 1, 0
        elif first_up <= first_dn:
            k, barrier = int(first_up), 1
        else:
            k, barrier = int(first_dn), -1
        ret = float(path[k])
        complete = barrier != 0 or (i + horizon) <= len(idx) - 1
        label = barrier if barrier != 0 else (1 if ret > 0 else -1)
        rows.append((e, idx[i + 1 + k], ret, barrier, label, target, s, complete))
    out = pd.DataFrame(
        rows,
        columns=[
            "event",
            "t1",
            "ret",
            "barrier",
            "label",
            "target",
            "side",
            "complete",
        ],
    )
    return out.set_index("event")


def meta_labels(barriers: pd.DataFrame) -> pd.Series:
    """meta-label：主要模型方向下的交易是否獲利（1/0）。"""
    return (barriers["ret"] > 0).astype(int)


def bet_size_from_prob(prob: pd.Series, step: float = 0.0) -> pd.Series:
    """由預測機率決定下注大小（AFML ch.10）：z=(p−½)/√(p(1−p))，m=2Φ(z)−1。

    回傳值介於 −1～1；step>0 時離散化以降低換手。
    """
    from scipy.stats import norm

    p = prob.clip(1e-6, 1 - 1e-6)
    z = (p - 0.5) / np.sqrt(p * (1 - p))
    size = 2 * norm.cdf(z) - 1
    if step > 0:
        size = (size / step).round() * step
    return pd.Series(size, index=prob.index)
