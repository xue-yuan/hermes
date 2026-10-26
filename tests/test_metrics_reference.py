"""回測績效數字的獨立對照。

- 每個顯示給使用者看的績效數字，都用照定義、逐筆計算的簡單寫法對照；少一個測試就失敗。
- PSR 的公式另外用模擬驗證：真實 Sharpe 等於門檻時，PSR 超過 95% 的機率應該是 5%。
"""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from hermes.backtest import metrics as m
from hermes.backtest.engine import extract_trades
from hermes.cli.console import METRIC_ROWS


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _trades(returns, position):
    """連續同方向的部位算一筆交易；賣出那天（隔夜跳空和賣出成本）也算在這筆。"""
    out, entry, side = [], None, 0
    for i, w in enumerate(position):
        s = 0 if abs(w) < 1e-10 else (1 if w > 0 else -1)
        if entry is not None and s != side:
            out.append((math.prod(1 + r for r in returns[entry: i + 1]) - 1, i - entry))
            entry = None
        if entry is None and s != 0:
            entry, side = i, s
    if entry is not None:
        out.append((math.prod(1 + r for r in returns[entry:]) - 1, len(position) - entry))
    return out


def reference_metrics(returns: list[float], dates: list[date], position: list[float], bench: list[float]) -> dict:
    n = len(returns)
    years = (dates[-1] - dates[0]).days / 365.25
    per_year = n / years
    equity, peak, dd = [], 1.0, []
    v = 1.0
    for r in returns:
        v *= 1 + r
        peak = max(peak, v)
        equity.append(v)
        dd.append(v / peak - 1)
    longest = run = 0
    for d in dd:
        run = run + 1 if d < 0 else 0
        longest = max(longest, run)
    mu, sd = _mean(returns), _std(returns)
    downside = math.sqrt(_mean([min(r, 0.0) ** 2 for r in returns]))
    # PSR（Bailey & López de Prado 2012）：偏態、峰度用母體動差
    m2 = _mean([(r - mu) ** 2 for r in returns])
    skew = _mean([(r - mu) ** 3 for r in returns]) / m2 ** 1.5
    kurt = _mean([(r - mu) ** 4 for r in returns]) / m2 ** 2
    sr = mu / sd
    z = sr * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    trades = _trades(returns, position)
    rets = [t[0] for t in trades]
    gains, losses = sum(r for r in rets if r > 0), -sum(r for r in rets if r < 0)
    mb = _mean(bench)
    beta = sum((r - mu) * (b - mb) for r, b in zip(returns, bench)) / sum((b - mb) ** 2 for b in bench)
    active = [r - b for r, b in zip(returns, bench)]
    cagr = equity[-1] ** (1 / years) - 1
    return {
        "total_return": equity[-1] - 1,
        "cagr": cagr,
        "ann_vol": sd * math.sqrt(per_year),
        "sharpe": mu / sd * math.sqrt(per_year),
        "sortino": mu / downside * math.sqrt(per_year),
        "max_drawdown": min(dd),
        "mdd_duration": longest,
        "psr": 0.5 * (1 + math.erf(z / math.sqrt(2))),
        "calmar": cagr / abs(min(dd)),
        "exposure": sum(1 for w in position if abs(w) > 1e-9) / n,
        "turnover": sum(abs(position[i] - position[i - 1]) for i in range(1, n)) / years,
        "trades": len(trades),
        "win_rate": sum(1 for r in rets if r > 0) / len(rets),
        "avg_trade": _mean(rets),
        "profit_factor": gains / losses,
        "avg_hold": _mean([t[1] for t in trades]),
        "beta": beta,
        "alpha": (mu - beta * mb) * per_year,
        "info_ratio": _mean(active) / _std(active) * math.sqrt(per_year),
    }


def _case(seed: int):
    """隨機的交易日曆（平日扣掉休市日）、報酬、0／1 部位，以及基準報酬。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", "2024-06-28")
    days = days.delete(rng.choice(np.arange(1, len(days) - 1), size=60, replace=False))
    n = len(days)
    position = pd.Series((rng.random(n) < 0.03).cumsum() % 2, index=days, dtype=float)  # 隨機進出約 20 次
    bench = pd.Series(rng.normal(0.0004, 0.012, n), index=days)
    returns = (position * (bench + rng.normal(0.0002, 0.006, n))).where(position > 0, 0.0)
    return returns, position, bench


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_metric_matches_definition(seed):
    returns, position, bench = _case(seed)
    trades = extract_trades(returns, position)
    ours = m.summarize(returns, position, trades, bench)
    ref = reference_metrics(returns.tolist(), [d.date() for d in returns.index], position.tolist(), bench.tolist())
    assert set(ours) == set(ref), f"績效數字和對照不一致：多了 {set(ours) - set(ref)}，少了 {set(ref) - set(ours)}"
    assert ref["trades"] >= 10
    for key in ref:
        assert ours[key] == pytest.approx(ref[key], rel=1e-9, abs=1e-12), key


def test_displayed_metrics_all_have_reference():
    """畫面上列出的每個績效數字（METRIC_ROWS）都要在 reference_metrics 裡有對照。"""
    returns, position, bench = _case(1)
    ref = reference_metrics(returns.tolist(), [d.date() for d in returns.index], position.tolist(), bench.tolist())
    shown = {key for key, _, _ in METRIC_ROWS}
    assert shown <= set(ref), f"沒有對照的績效數字：{shown - set(ref)}"


@pytest.mark.parametrize("dist", ["normal", "skewed"])
def test_psr_is_calibrated(dist):
    """真實 Sharpe 剛好等於門檻時，PSR 應該像均勻分布：超過 95% 的比例約 5%。

    偏態的情況用負的對數常態報酬（左尾長），檢查公式裡的偏態、峰度修正：
    拿掉修正的話，超過 95% 的比例會變成約 10%。
    """
    rng = np.random.default_rng(11)
    t, sims = 1000, 2000
    if dist == "normal":
        true_sr = 0.05
        draws = rng.normal(true_sr, 1.0, (sims, t))
    else:
        sigma, true_sr = 0.75, 0.15
        draws = -rng.lognormal(0.0, sigma, (sims, t))
        mean, var = -math.exp(sigma ** 2 / 2), (math.exp(sigma ** 2) - 1) * math.exp(sigma ** 2)
        draws += true_sr * math.sqrt(var) - mean  # 平移到真實 Sharpe = true_sr
    psr = np.array([m.probabilistic_sharpe_ratio(pd.Series(row), true_sr) for row in draws])
    assert abs((psr > 0.95).mean() - 0.05) < 0.015
    assert abs((psr < 0.5).mean() - 0.5) < 0.04
