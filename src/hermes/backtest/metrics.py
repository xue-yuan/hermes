"""績效指標，包含 Probabilistic / Deflated Sharpe Ratio。

參考：
- Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier", Journal of Risk.
- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for Selection Bias,
  Backtest Overfitting and Non-Normality", Journal of Portfolio Management.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329
DAYS_PER_YEAR = 365.25  # 日曆天數，含閏年


def years_between(index: pd.DatetimeIndex) -> float:
    """第一天到最後一天經過的日曆年數。年化不假設一年有幾個交易日。"""
    if len(index) < 2:
        return float("nan")
    return (index[-1] - index[0]).days / DAYS_PER_YEAR


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """這段期間實際平均每年有幾個交易日，用來把每日的波動度、Sharpe 換算成年化。"""
    years = years_between(index)
    return len(index) / years if years > 0 else float("nan")


def cagr(returns: pd.Series) -> float:
    """年化報酬：期初淨值為 1，用第一天到最後一天經過的日曆時間換算。returns 的 index 是日期。"""
    returns = returns.dropna()
    years = years_between(returns.index)
    growth = float((1.0 + returns).prod())
    if not years > 0 or growth < 0:
        return float("nan")
    return growth ** (1.0 / years) - 1.0


def ann_vol(returns: pd.Series, periods: float) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe(returns: pd.Series, periods: float, rf: float = 0.0) -> float:
    excess = returns - rf / periods
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def sortino(returns: pd.Series, periods: float, rf: float = 0.0) -> float:
    excess = returns - rf / periods
    downside = np.sqrt((excess.clip(upper=0.0) ** 2).mean())
    return (
        float(excess.mean() / downside * np.sqrt(periods))
        if downside > 0
        else float("nan")
    )


def drawdown(equity: pd.Series) -> pd.Series:
    """淨值離之前的最高點跌了多少。期初的本金（淨值 1）也算最高點，第一天就賠錢也是回撤。"""
    return equity / equity.cummax().clip(lower=1.0) - 1.0


def max_drawdown(equity: pd.Series) -> dict:
    dd = drawdown(equity)
    if dd.empty:
        return {"mdd": float("nan"), "duration": 0}
    # 最長水下期間（交易日）
    underwater = (dd < 0).astype(int)
    groups = (underwater.diff() != 0).cumsum()
    longest = int(underwater.groupby(groups).sum().max()) if underwater.any() else 0
    return {"mdd": float(dd.min()), "duration": longest}


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR：真實 Sharpe 大於 sr_benchmark 的機率。Sharpe 皆為「每期」（未年化）數值。

    PSR = Φ[ (SR̂ − SR*)·√(T−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) ]，γ₄ 為（非超額）峰度。
    """
    r = returns.dropna()
    t = len(r)
    if t < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(t - 1) / np.sqrt(denom)))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """N 次獨立、真實 Sharpe 為 0 的試驗中，最大 Sharpe 的期望值（每期單位）。"""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    returns: pd.Series, trial_sharpes: list[float] | np.ndarray
) -> float:
    """DSR：以多次試驗（參數組合）的 Sharpe 變異數修正選擇偏誤後的 PSR。

    trial_sharpes 為所有試驗的「每期」Sharpe（含被選中的那一個）。
    """
    trial_sharpes = np.asarray([s for s in trial_sharpes if np.isfinite(s)])
    n = len(trial_sharpes)
    if n < 2:
        return probabilistic_sharpe_ratio(returns, 0.0)
    sr0 = expected_max_sharpe(n, float(np.var(trial_sharpes, ddof=1)))
    return probabilistic_sharpe_ratio(returns, sr0)


def per_period_sharpe(returns: pd.Series) -> float:
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd) if sd > 0 else float("nan")


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {
            "trades": 0,
            "win_rate": float("nan"),
            "avg_trade": float("nan"),
            "profit_factor": float("nan"),
            "avg_hold": float("nan"),
        }
    rets = trades["return"]
    gains, losses = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "trades": len(trades),
        "win_rate": float((rets > 0).mean()),
        "avg_trade": float(rets.mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "avg_hold": float(trades["bars"].mean()),
    }


def summarize(
    returns: pd.Series,
    position: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | None = None,
    rf: float = 0.0,
) -> dict:
    """returns 的 index 是日期；年化用這段期間經過的日曆時間和實際的交易日數。"""
    returns = returns.fillna(0.0)
    equity = (1.0 + returns).cumprod()
    mdd = max_drawdown(equity)
    years, periods = years_between(returns.index), periods_per_year(returns.index)
    out = {
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else float("nan"),
        "cagr": cagr(returns),
        "ann_vol": ann_vol(returns, periods),
        "sharpe": sharpe(returns, periods, rf),
        "sortino": sortino(returns, periods, rf),
        "max_drawdown": mdd["mdd"],
        "mdd_duration": mdd["duration"],
        "psr": probabilistic_sharpe_ratio(returns),
    }
    out["calmar"] = (
        out["cagr"] / abs(out["max_drawdown"])
        if out["max_drawdown"] < 0
        else float("nan")
    )
    if position is not None:
        out["exposure"] = float((position.abs() > 1e-9).mean())
        # 期初空手（和回測引擎相同），第一天就有部位也是一次買進
        change = position.diff().fillna(position.iloc[0]) if len(position) else position
        out["turnover"] = (
            float(change.abs().sum() / years) if years > 0 else float("nan")
        )
    if trades is not None:
        out.update(trade_stats(trades))
    if benchmark_returns is not None:
        b = benchmark_returns.reindex(returns.index).fillna(0.0)
        var_b = b.var(ddof=1)
        beta = float(returns.cov(b) / var_b) if var_b > 0 else float("nan")
        alpha = (
            float((returns.mean() - beta * b.mean()) * periods)
            if np.isfinite(beta)
            else float("nan")
        )
        active = returns - b
        te = active.std(ddof=1)
        out["beta"] = beta
        out["alpha"] = alpha
        out["info_ratio"] = (
            float(active.mean() / te * np.sqrt(periods)) if te > 0 else float("nan")
        )
    return out
