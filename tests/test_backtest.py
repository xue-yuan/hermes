import numpy as np
import pandas as pd
import pytest

from hermes.backtest import CostModel, run_backtest
from hermes.backtest import metrics as m


def test_buy_hold_without_costs_matches_price_path(prices):
    target = pd.Series(1.0, index=prices.index)
    res = run_backtest(prices, target, CostModel.zero())
    # t=0 收盤產生訊號，t=1 開盤買進
    expected = prices["close"].iloc[-1] / prices["open"].iloc[1]
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-10)


def test_signal_only_affects_following_days(prices):
    target = pd.Series(0.0, index=prices.index)
    target.iloc[100] = 1.0
    res = run_backtest(prices, target, CostModel.zero())
    nz = res.returns[res.returns != 0]
    assert nz.index.min() == prices.index[101]
    # 第 101 日開盤買進、第 102 日開盤賣出
    expected = (prices["close"].iloc[101] / prices["open"].iloc[101]) * (prices["open"].iloc[102] / prices["close"].iloc[101])
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-10)


def test_round_trip_costs():
    idx = pd.bdate_range("2020-01-01", periods=5)
    flat = pd.DataFrame({"open": 100.0, "close": 100.0, "high": 100.0, "low": 100.0, "volume": 1.0}, index=idx)
    target = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0], index=idx)
    costs = CostModel()
    res = run_backtest(flat, target, costs)
    total_cost = 1 - res.equity.iloc[-1]
    assert total_cost == pytest.approx(costs.buy_rate + costs.sell_rate * (1 - costs.buy_rate), rel=1e-9)
    assert costs.sell_rate - costs.buy_rate == pytest.approx(0.003)


def test_close_execution(prices):
    target = pd.Series(1.0, index=prices.index)
    res = run_backtest(prices, target, CostModel.zero(), execution="close")
    assert res.equity.iloc[-1] == pytest.approx(prices["close"].iloc[-1] / prices["close"].iloc[0], rel=1e-10)


def test_trade_extraction(prices):
    target = pd.Series(0.0, index=prices.index)
    target.iloc[10:20] = 1.0
    target.iloc[50:60] = 1.0
    res = run_backtest(prices, target, CostModel.zero())
    assert len(res.trades) == 2
    assert (res.trades["bars"] == 10).all()


def test_drawdown_and_sharpe():
    eq = pd.Series([1.0, 1.2, 0.6, 0.9, 1.3])
    assert m.max_drawdown(eq)["mdd"] == pytest.approx(-0.5)
    r = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 5000))
    assert m.sharpe(r, 244) == pytest.approx(r.mean() / r.std() * np.sqrt(244))


def _calendar(start: str, end: str, drop: int, seed: int = 0) -> pd.DatetimeIndex:
    """平日扣掉 drop 個隨機的休市日，頭尾兩天保留。"""
    days = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    holidays = rng.choice(np.arange(1, len(days) - 1), size=drop, replace=False)
    return days.delete(holidays)


def test_cagr_uses_calendar_time():
    """兩年漲 21%，年化就是約 10%，和這段期間有幾個交易日無關。

    2021-01-04 到 2023-01-04 共 730 天，手算 1.21 ^ (365.25 / 730) − 1 = 10.0072%。
    原本「一年 252 個交易日」的算法：交易日越少，年化報酬算出來越高。
    """
    for drop in (0, 30, 60):
        idx = _calendar("2021-01-04", "2023-01-04", drop)
        r = pd.Series(1.21 ** (1 / len(idx)) - 1, index=idx)
        assert m.summarize(r)["cagr"] == pytest.approx(0.100072, abs=1e-6)


def test_annualization_uses_actual_trading_days():
    """每年的交易日數用這段期間實際的數字。和逐年數交易日、取平均的結果對照。"""
    idx = _calendar("2020-01-02", "2025-12-31", drop=6 * 17, seed=3)
    counted = pd.Series(1, index=idx).groupby(idx.year).sum().mean()
    assert 240 < counted < 250
    assert m.periods_per_year(idx) == pytest.approx(counted, rel=2e-3)
    r = pd.Series(np.random.default_rng(4).normal(0.0005, 0.02, len(idx)), index=idx)
    out = m.summarize(r)
    assert out["ann_vol"] == pytest.approx(r.std() * np.sqrt(counted), rel=1e-3)
    assert out["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(counted), rel=1e-3)


def test_psr_and_dsr():
    rng = np.random.default_rng(2)
    good = pd.Series(rng.normal(0.001, 0.01, 2500))
    assert m.probabilistic_sharpe_ratio(good) > 0.99
    noise = pd.Series(rng.normal(0.0, 0.01, 2500))
    assert m.probabilistic_sharpe_ratio(noise) < 0.99
    trials = rng.normal(0.0, 0.02, 100)
    assert m.deflated_sharpe_ratio(good, trials) <= m.probabilistic_sharpe_ratio(good)
    assert m.expected_max_sharpe(1000, 0.01) > m.expected_max_sharpe(10, 0.01) > 0
