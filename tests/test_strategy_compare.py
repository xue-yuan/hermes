"""內建策略的回測、多個策略並排比較、籌碼摘要：services 組合的結果要和直接呼叫底層模組相同。"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_prices
from hermes.backtest import CostModel, run_backtest
from hermes.chips import chip_summary
from hermes.data import Cache, DataLoader
from hermes.services import (
    SORT_KEYS,
    builtin_backtest,
    chip_report,
    compare_strategies,
    is_rule_strategy,
    rule_backtest,
    signal_history,
)
from hermes.strategies import StrategyConfigError
from test_rule_backtest import _write_stock

COSTS = CostModel(discount=0.5, slippage=0.001, tax=0.003)
# 門檻不同的突破策略；never 永遠不會買進，報酬全是 0，Sharpe 算不出來
THRESHOLDS = {"low": (95, 90), "mid": (105, 98), "high": (115, 108)}
NEVER = "name: 永遠不買\nbuy: {left: close, op: gt, right: 1000000}\nsell: {left: close, op: lt, right: 0}\n"


@pytest.fixture
def case(tmp_path):
    home, strategies = tmp_path / "home", tmp_path / "strategies"
    base = make_prices(n=300, seed=11)
    days = pd.bdate_range("2022-01-03", periods=len(base))
    bars = [tuple(base[c].iloc[i] for c in ("open", "high", "low", "close")) for i in range(len(base))]
    _write_stock(home, "1234", bars, days)
    strategies.mkdir()
    for name, (buy, sell) in THRESHOLDS.items():
        (strategies / f"{name}.yaml").write_text(
            f"name: 突破 {buy}\nbuy: {{left: close, op: gt, right: {buy}}}\nsell: {{left: close, op: lt, right: {sell}}}\n",
            encoding="utf-8")
    (strategies / "never.yaml").write_text(NEVER, encoding="utf-8")
    return strategies, DataLoader(Cache(home))


def test_builtin_backtest_matches_engine(case):
    strategies, loader = case
    bt = builtin_backtest("1234", "buy_hold", costs=COSTS, loader=loader)
    prices = loader.prices("1234")
    direct = run_backtest(prices, pd.Series(1.0, index=prices.index), COSTS, "open", "買進持有", {}, prices.index[0])
    pd.testing.assert_series_equal(bt.result.returns, direct.returns)
    assert bt.label == "買進持有" and bt.result.metrics == direct.metrics


@pytest.mark.parametrize("sort", SORT_KEYS)
def test_compare_sorts_best_first(case, sort):
    strategies, loader = case
    refs = [*THRESHOLDS, "never"]
    cmp = compare_strategies("1234", refs, costs=COSTS, sort=sort, loader=loader, directory=strategies)

    # 對照：每個策略各自回測，從大家都有訊號的第一天開始
    start = max(signal_history("1234", r, loader=loader, directory=strategies).result.warmup_end for r in refs)
    alone = {rule_backtest(signal_history("1234", r, loader=loader, directory=strategies), start, None, COSTS,
                           "open", loader).result.name: r for r in refs}
    values = {name: rb_metric for name, rb_metric in
              ((r.name, r.metrics[sort]) for r in cmp.results)}
    assert set(values) == set(alone)
    shown = [values[r.name] for r in cmp.results]
    finite = [v for v in shown if np.isfinite(v)]
    assert finite == sorted(finite, reverse=True)  # 最大回撤是負數：越接近 0 越前面
    assert all(np.isfinite(v) for v in shown[:len(finite)])  # 算不出來的排最後
    if sort == "max_drawdown":
        assert cmp.results[0].name == "永遠不買" and shown[0] == 0.0  # 沒有交易就沒有回撤
    if sort == "sharpe":
        assert cmp.results[-1].name == "永遠不買" and np.isnan(shown[-1])


def test_compare_includes_buy_and_hold(case):
    strategies, loader = case
    cmp = compare_strategies("1234", ["mid"], costs=COSTS, loader=loader, directory=strategies)
    prices = loader.prices("1234")
    start = cmp.results[0].returns.index[0]
    direct = run_backtest(prices, pd.Series(1.0, index=prices.index), COSTS, "open", "買進持有", {}, start)
    pd.testing.assert_series_equal(cmp.benchmark.returns, direct.returns)


def test_compare_rejects_bad_input(case):
    strategies, loader = case
    with pytest.raises(ValueError, match="排序依據"):
        compare_strategies("1234", ["mid"], sort="profit", loader=loader, directory=strategies)
    with pytest.raises(ValueError, match="請指定"):
        compare_strategies("1234", [], loader=loader, directory=strategies)


def test_rule_strategy_name_must_not_shadow_builtin(case):
    strategies, _ = case
    assert is_rule_strategy("mid", strategies) and not is_rule_strategy("buy_hold", strategies)
    (strategies / "buy_hold.yaml").write_text(NEVER, encoding="utf-8")
    with pytest.raises(StrategyConfigError, match="同時是策略設定檔和內建策略"):
        is_rule_strategy("buy_hold", strategies)


@pytest.mark.parametrize(("closes", "expected"), [
    ([100] * 30 + [101], (True, "buy")),  # 最後一天剛站上 100.5
    ([100] * 30 + [101, 101], (True, None)),
    ([100] * 30 + [101, 99], (False, "sell")),
    ([100] * 31, (False, None)),
])
def test_latest_action(tmp_path, closes, expected):
    home, strategies = tmp_path / "home", tmp_path / "strategies"
    days = pd.bdate_range("2026-01-05", periods=len(closes))
    _write_stock(home, "1234", [(c, c, c, c) for c in closes], days)
    strategies.mkdir()
    (strategies / "s.yaml").write_text("name: s\nbuy: {left: close, op: gt, right: 100.5}\n"
                                       "sell: {left: close, op: lt, right: 99.5}\n", encoding="utf-8")
    loader = DataLoader(Cache(home))
    rb = rule_backtest(signal_history("1234", "s", loader=loader, directory=strategies), days[0], loader=loader)
    assert rb.latest() == (days[-1], *expected)


class _FakeLoader:
    """chip_report 只讀這四種資料。"""

    def __init__(self):
        days = pd.bdate_range("2026-01-05", periods=30)
        rng = np.random.default_rng(3)
        self.days = days
        self.inst = pd.DataFrame({k: rng.integers(-5e6, 5e6, len(days)).astype(float)
                                  for k in ("foreign", "trust", "dealer", "total")}, index=days)
        self.hold = pd.DataFrame({"foreign_ratio": rng.uniform(70, 75, len(days))}, index=days)
        self.marg = pd.DataFrame({"margin_balance": rng.uniform(1e4, 2e4, len(days)),
                                  "margin_limit": 1e6, "short_balance": rng.uniform(100, 200, len(days)),
                                  "short_margin_ratio": rng.uniform(0.01, 0.02, len(days))}, index=days)

    def prices(self, stock_id, adjust=True):
        return pd.DataFrame({"close": 1.0}, index=self.days)

    def institutional(self, stock_id):
        return self.inst

    def shareholding(self, stock_id):
        return self.hold

    def margin(self, stock_id):
        return self.marg


def test_chip_report_matches_summary():
    loader = _FakeLoader()
    got = chip_report("1234", [5, 20], 7, loader)
    want = chip_summary(loader.days, loader.inst, loader.hold, loader.marg, (5, 20), 7)
    pd.testing.assert_frame_equal(got.flows, want.flows)
    pd.testing.assert_frame_equal(got.recent, want.recent)
    assert got.streaks == want.streaks and got.as_of == want.as_of
    assert got.foreign_ratio == want.foreign_ratio and got.margin == want.margin
    with pytest.raises(ValueError, match="正整數"):
        chip_report("1234", [0], 7, loader)
