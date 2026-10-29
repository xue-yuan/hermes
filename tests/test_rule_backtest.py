"""設定檔策略的回測。

對照方法：照定義逐日模擬持股和現金，只用實際成交價和官方除權息紀錄。
除權息當天開盤前，持股乘上「前一日收盤 ÷ 參考價」（股利用參考價再投入），
和回測引擎「用還原價格算報酬」是兩種不同的寫法，結果必須相同。
"""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from conftest import make_prices
from hermes.backtest import CostModel
from hermes.cli.main import app
from hermes.data import Cache, DataLoader
from hermes.services import reference_prices, rule_backtest, signal_history
from hermes.strategies import StrategyConfigError, parse_strategy, run_rules
from test_engine_oracle import _random_strategy

STRATEGY = """
name: 突破 105 買、跌破 100 賣
buy:  {left: close, op: gt, right: 105}
sell: {left: close, op: lt, right: 100}
"""


def _write_stock(home, stock_id: str, bars: list[tuple[float, float, float, float]], days: pd.DatetimeIndex,
                 dividends=(), reductions=(), splits=()) -> None:
    """bars：每天的（開, 高, 低, 收）；dividends 等：（日期, 前一日收盤, 參考價）。"""
    cache = Cache(home)
    cache.write("TaiwanStockPrice", stock_id, pd.DataFrame([
        {"date": f"{d:%Y-%m-%d}", "stock_id": stock_id, "Trading_Volume": 1000, "Trading_money": 1000 * c,
         "open": o, "max": h, "min": lo, "close": c, "spread": 0.0, "Trading_turnover": 10}
        for d, (o, h, lo, c) in zip(days, bars)]))
    cache.write("TaiwanStockDividendResult", stock_id, pd.DataFrame(
        [{"date": f"{d:%Y-%m-%d}", "stock_id": stock_id, "before_price": b, "after_price": r,
          "stock_or_cache_dividend": "息"} for d, b, r in dividends],
        columns=["date", "stock_id", "before_price", "after_price", "stock_or_cache_dividend"]))
    cache.write("TaiwanStockCapitalReductionReferencePrice", stock_id, pd.DataFrame(
        [{"date": f"{d:%Y-%m-%d}", "stock_id": stock_id, "ClosingPriceonTheLastTradingDay": b,
          "PostReductionReferencePrice": r} for d, b, r in reductions],
        columns=["date", "stock_id", "ClosingPriceonTheLastTradingDay", "PostReductionReferencePrice"]))
    cache.write("TaiwanStockSplitPrice", stock_id, pd.DataFrame(
        [{"date": f"{d:%Y-%m-%d}", "stock_id": stock_id, "before_price": b, "after_price": r} for d, b, r in splits],
        columns=["date", "stock_id", "before_price", "after_price"]))
    cache.write("TaiwanStockInfo", stock_id, pd.DataFrame([{"stock_id": stock_id, "stock_name": "測試", "type": "twse",
                                                            "industry_category": ""}]))


# ---------------------------------------------------------------------------
# 手算的例子
#
# 第 2 天收 106 > 105 → 第 3 天開盤 107 買進。第 5 天除息 5 元（前收 110、參考價 105）。
# 第 7 天收 99 < 100 → 第 8 天開盤 98.5 賣出。
# 不計成本的報酬 = 98.5 / 107 × 110 / 105 − 1 = −3.56%（沒算股利會是 −7.94%）。
# 用還原價格判斷：除息前的收盤都乘上 105 / 110，最高只到 105，不大於 105，所以完全沒有買進。
DAYS = pd.bdate_range("2026-03-02", periods=10)
OPENS = [100, 100, 101, 107, 108, 104, 104, 102, 98.5, 98]
CLOSES = [100, 100, 106, 108, 110, 104, 103, 99, 98, 98]


@pytest.fixture
def dividend_case(tmp_path):
    home, strategies = tmp_path / "home", tmp_path / "strategies"
    bars = [(o, max(o, c), min(o, c), c) for o, c in zip(OPENS, CLOSES)]
    _write_stock(home, "1234", bars, DAYS, dividends=[(DAYS[5], 110.0, 105.0)])
    strategies.mkdir()
    (strategies / "breakout.yaml").write_text(STRATEGY, encoding="utf-8")
    return home, strategies, DataLoader(Cache(home))


def test_hand_computed_trade_includes_dividend(dividend_case):
    _, strategies, loader = dividend_case
    rb = rule_backtest(signal_history("1234", "breakout", loader=loader, directory=strategies),
                       costs=CostModel.zero(), loader=loader)
    assert rb.start == DAYS[0]
    [t] = rb.trades().to_dict("records")
    assert (t["buy_signal"], t["entry"], t["sell_signal"], t["exit"]) == (DAYS[2], DAYS[3], DAYS[7], DAYS[8])
    assert (t["entry_price"], t["exit_price"], t["bars"]) == (107.0, 98.5, 5)
    assert t["return"] == pytest.approx(98.5 / 107 * 110 / 105 - 1, rel=1e-12)
    assert rb.result.equity.iloc[-1] == pytest.approx(98.5 / 107 * 110 / 105, rel=1e-12)
    assert not t["buy_at_limit"] and not t["sell_at_limit"]


def test_adjusted_mode_uses_adjusted_signals(dividend_case):
    """還原模式：用還原價格判斷完全沒有買進（見上面的手算），一直空手。"""
    _, strategies, loader = dividend_case
    rb = rule_backtest(signal_history("1234", "breakout", adjust=True, loader=loader, directory=strategies),
                       costs=CostModel.zero(), loader=loader)
    assert rb.adjusted and rb.result.trades.empty and rb.trades().empty
    assert rb.result.equity.iloc[-1] == 1.0
    assert (rb.latest().bought, rb.latest().action) == (False, None)
    # 成交價格和漲跌停一律看實際成交價
    assert rb.raw_prices["close"].equals(loader.prices("1234", adjust=False)["close"])


def test_costs_follow_engine_convention(dividend_case):
    """買進成本從買進當天的淨值扣除，賣出成本從賣出金額扣除（backtest/engine.py）。"""
    _, strategies, loader = dividend_case
    costs = CostModel()
    rb = rule_backtest(signal_history("1234", "breakout", loader=loader, directory=strategies),
                       costs=costs, loader=loader)
    adj_close3 = 108 * 105 / 110  # 除息前的價格要乘上 105 / 110
    expected = (108 / 107 - costs.buy_rate) * (98.5 / adj_close3) * (1 - costs.sell_rate) - 1
    assert rb.trades()["return"].iloc[0] == pytest.approx(expected, rel=1e-12)


def test_reference_price_uses_official_record(dividend_case):
    _, _, loader = dividend_case
    ref = reference_prices(loader.prices("1234", adjust=False), loader, "1234")
    assert np.isnan(ref.iloc[0])
    assert ref[DAYS[5]] == 105.0  # 除息當天用官方參考價，不是前一天收盤 110
    assert ref[DAYS[6]] == 104.0


def test_close_execution_trades_on_signal_day(dividend_case):
    _, strategies, loader = dividend_case
    rb = rule_backtest(signal_history("1234", "breakout", loader=loader, directory=strategies),
                       costs=CostModel.zero(), execution="close", loader=loader)
    [t] = rb.trades().to_dict("records")
    assert (t["buy_signal"], t["entry"], t["entry_price"]) == (DAYS[2], DAYS[2], 106.0)
    assert (t["sell_signal"], t["exit"], t["exit_price"]) == (DAYS[7], DAYS[7], 99.0)
    assert t["return"] == pytest.approx(99 / 106 * 110 / 105 - 1, rel=1e-12)


def test_limit_prices_are_flagged(tmp_path):
    """第 1 天收 106 → 第 2 天開盤 116.5 是漲停價（106 × 1.1 = 116.6，向下取到 0.5）；
    第 3 天收 99 → 第 4 天開盤 89.1 是跌停價（99 × 0.9 = 89.1）。"""
    days = pd.bdate_range("2026-03-02", periods=5)
    bars = [(100, 100, 100, 100), (100, 106, 100, 106), (116.5, 116.5, 110, 110), (110, 110, 99, 99),
            (89.1, 90, 89.1, 90)]
    _write_stock(tmp_path, "1234", bars, days)
    loader = DataLoader(Cache(tmp_path))
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    (strategies / "breakout.yaml").write_text(STRATEGY, encoding="utf-8")
    rb = rule_backtest(signal_history("1234", "breakout", loader=loader, directory=strategies), loader=loader)
    [t] = rb.trades().to_dict("records")
    assert (t["entry_price"], t["exit_price"]) == (116.5, 89.1)
    assert t["buy_at_limit"] and t["sell_at_limit"]


def test_position_holds_until_sell():
    idx = pd.bdate_range("2026-01-05", periods=8)
    prices = pd.DataFrame({c: 10.0 for c in ("open", "high", "low", "close", "volume")}, index=idx)
    res = run_rules(parse_strategy({"name": "t", "buy": {"left": "close", "op": "gt", "right": 0}}), prices)
    res.buy = pd.Series([False, True, False, True, True, False, True, True], index=idx)
    res.sell = pd.Series([False, False, False, True, False, True, True, False], index=idx)
    # 第 3 天買賣都成立：持有中就賣出；第 6 天買賣都成立：空手就不買
    assert res.position().tolist() == [0, 1, 1, 0, 1, 0, 0, 1]
    pos = res.position(idx[2])
    assert pos.iloc[:2].isna().all()
    assert pos.iloc[2:].tolist() == [0, 0, 1, 0, 0, 1]  # 從第 2 天空手開始，第 1 天的買進不算


def test_strategy_without_sell_cannot_be_backtested(dividend_case):
    _, strategies, loader = dividend_case
    (strategies / "buy_only.yaml").write_text("name: 只有買進\nbuy: {left: close, op: gt, right: 105}\n",
                                              encoding="utf-8")
    with pytest.raises(StrategyConfigError, match="沒有賣出條件"):
        rule_backtest(signal_history("1234", "buy_only", loader=loader, directory=strategies), loader=loader)


# ---------------------------------------------------------------------------
# 隨機策略 × 有除權息、減資、分割的資料：照定義逐日模擬，和回測引擎比對


def _event_case(home, n: int = 320, seed: int = 5) -> dict[pd.Timestamp, float]:
    """回傳 {事件日: 前一日收盤 ÷ 參考價}。事件日起的價格乘上比例，模擬除權息、減資、分割後的價格。"""
    base = make_prices(n=n, seed=seed)
    days = pd.bdate_range("2021-01-04", periods=n)
    ratios = {days[90]: 0.96, days[170]: 1.25, days[250]: 0.5}  # 參考價 ÷ 前一日收盤
    scale = np.ones(n)
    for d, r in ratios.items():
        scale[days >= d] *= r
    bars = [tuple(base[c].iloc[i] * scale[i] for c in ("open", "high", "low", "close")) for i in range(n)]
    closes = [b[3] for b in bars]
    event = lambda d: (d, closes[days.get_loc(d) - 1], closes[days.get_loc(d) - 1] * ratios[d])  # noqa: E731
    _write_stock(home, "1234", bars, days, dividends=[event(days[90])], reductions=[event(days[170])],
                 splits=[event(days[250])])
    return {d: 1 / r for d, r in ratios.items()}


def _simulate(raw: pd.DataFrame, buy: pd.Series, sell: pd.Series, events: dict, start, end, execution: str):
    """逐日模擬持股和現金（不計成本）。回傳每天的淨值，以及每筆交易的（買進日, 賣出日）。"""
    days = raw.index[(raw.index >= start) & (raw.index <= (end or raw.index[-1]))]
    cash, shares, holding, entry = 1.0, 0.0, False, None
    equity, trades = [], []
    for i, d in enumerate(days):
        if holding and d in events:
            shares *= events[d]  # 股利用參考價再投入
        signal_day = days[i - 1] if execution == "open" else d
        price = raw.at[d, execution]
        if execution == "close" or i > 0:
            if holding and sell[signal_day]:
                cash, shares, holding = shares * price, 0.0, False
                trades.append((entry, d))
            elif not holding and buy[signal_day] and not sell[signal_day]:
                shares, cash, holding, entry = cash / price, 0.0, True, d
        equity.append(shares * raw.at[d, "close"] if holding else cash)
    if holding:
        trades.append((entry, None))
    return pd.Series(equity, index=days), trades


def _engine_trades(trades: pd.DataFrame) -> list:
    return [(t["entry"], None if pd.isna(t["exit"]) else t["exit"]) for _, t in trades.iterrows()]


@pytest.fixture(scope="module")
def event_case(tmp_path_factory):
    home = tmp_path_factory.mktemp("events")
    events = _event_case(home)
    return DataLoader(Cache(home)), events


def test_event_case_adjustment_is_consistent(event_case):
    loader, events = event_case
    adj = loader.prices("1234", adjust=True)
    assert loader.unexplained_gaps("1234").empty
    assert len(loader.adjustments("1234")) == 3
    for d in events:
        i = adj.index.get_loc(d)
        assert adj["close"].iloc[i] / adj["close"].iloc[i - 1] == pytest.approx(
            adj["raw_close"].iloc[i] * events[d] / adj["raw_close"].iloc[i - 1], rel=1e-12)


@pytest.mark.parametrize("adjusted", [False, True])
@pytest.mark.parametrize("execution", ["open", "close"])
@pytest.mark.parametrize("seed", range(12))
def test_backtest_matches_daily_simulation(event_case, tmp_path, seed, execution, adjusted):
    """還原模式（adjusted）：訊號用還原價格判斷。逐日模擬的訊號自己用 run_rules 算，不用回測引擎拿到的。"""
    loader, events = event_case
    raw = loader.prices("1234", adjust=False)
    strategy = _random_strategy(seed, raw)
    (tmp_path / "s.yaml").write_text(strategy.model_dump_json(exclude_none=True), encoding="utf-8")
    hist = signal_history("1234", str(tmp_path / "s.yaml"), adjust=adjusted, loader=loader)
    signals = run_rules(strategy, loader.prices("1234", adjust=adjusted))
    windows = [(None, None), (raw.index[120], raw.index[280])]
    for start, end in windows:
        rb = rule_backtest(hist, start, end, CostModel.zero(), execution, loader)
        equity, trades = _simulate(raw, signals.buy, signals.sell, events, rb.start, end, execution)
        assert rb.result.equity.index.equals(equity.index)
        np.testing.assert_allclose(rb.result.equity.to_numpy(), equity.to_numpy(), rtol=1e-10)
        assert _engine_trades(rb.result.trades) == trades


def test_random_backtests_trade_through_events(event_case, tmp_path):
    """確認對照有意義：隨機策略有足夠的交易，而且有持股跨過除權息、減資、分割。"""
    loader, events = event_case
    raw = loader.prices("1234", adjust=False)
    n_trades, crossed = 0, set()
    for seed in range(12):
        strategy = _random_strategy(seed, raw)
        (tmp_path / "s.yaml").write_text(strategy.model_dump_json(exclude_none=True), encoding="utf-8")
        rb = rule_backtest(signal_history("1234", str(tmp_path / "s.yaml"), loader=loader),
                           costs=CostModel.zero(), loader=loader)
        held = rb.result.position
        n_trades += len(rb.result.trades)
        crossed |= {d for d in events if held.get(d, 0) and held.shift(1).get(d, 0)}
    assert n_trades >= 30
    assert crossed == set(events)


# ---------------------------------------------------------------------------
# CLI


def test_backtest_command_with_rule_strategy(dividend_case, monkeypatch):
    home, strategies, _ = dividend_case
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STRATEGIES", str(strategies))
    result = CliRunner().invoke(app, ["backtest", "1234", "-s", "breakout", "--no-plot"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    for text in ("設定檔策略回測", "2026-03-04", "2026-03-05", "107", "98.5", "持有天數",
                 "價格：訊號用實際成交價判斷；報酬包含股利", "策略目前還沒買進；下一個交易日開盤：不用動",
                 "一開始還沒買進"):
        assert text in result.output
    assert "還原" not in result.output

    result = CliRunner().invoke(app, ["backtest", "1234", "-s", "breakout", "-p", "x=1", "--no-plot"])
    assert result.exit_code == 1 and "不能用 -p" in result.output

    result = CliRunner().invoke(app, ["backtest", "1234", "-s", "breakout", "--adjust", "--no-plot"],
                                env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    for text in ("價格：訊號用還原價格判斷；報酬包含股利", "這段期間沒有任何交易"):
        assert text in result.output
    assert "實際成交價" not in result.output
    result = CliRunner().invoke(app, ["backtest", "1234", "-s", "buy_hold", "--adjust", "--no-plot"])
    assert result.exit_code == 1 and "--adjust 只用在策略設定檔" in result.output


def test_cli_rejects_bad_dates_and_counts_in_chinese(dividend_case, monkeypatch):
    """日期格式不對、起始日晚於結束日、筆數小於 1：顯示中文錯誤，不是程式例外或英文訊息。"""
    home, strategies, _ = dividend_case
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STRATEGIES", str(strategies))
    cases = [
        (["backtest", "1234", "-s", "breakout", "--start", "2026-13-01"], "--start 要是 YYYY-MM-DD 格式的日期"),
        (["backtest", "1234", "-s", "breakout", "--start", "2026-03-09", "--end", "2026-03-02"],
         "--start 不能晚於 --end"),
        (["backtest", "1234", "-s", "breakout", "--trades", "0"], "--trades 要是正整數"),
        (["compare", "1234", "-s", "breakout", "--end", "abc"], "--end 要是 YYYY-MM-DD 格式的日期"),
        (["compare", "1234", "-s", "breakout", "--top", "0"], "--top 要是正整數"),
        (["signals", "1234", "-s", "breakout", "--start", "3/2"], "--start 要是 YYYY-MM-DD 格式的日期"),
        (["signals", "1234", "-s", "breakout", "--start", "2026-04-01"], "--start 不能晚於結束日 2026-03-13"),
        (["tech", "1234", "--date", "abc"], "--date 要是 YYYY-MM-DD 格式的日期"),
        (["show", "1234", "--last", "0"], "--last 要是正整數"),
        (["fetch", "1234", "--start", "abc"], "--start 要是 YYYY-MM-DD 格式的日期"),
        (["verify", "1234", "--months", "0"], "--months 要是正整數"),
    ]
    for args, message in cases:
        result = CliRunner().invoke(app, [*args, "--no-plot"] if args[0] in ("backtest", "compare", "signals")
                                    else args, env={"COLUMNS": "200"})
        assert result.exit_code == 1 and message in result.output, (args, result.output)


def test_signals_default_start_is_a_year_before_end(dividend_case, monkeypatch):
    """只給結束日時，顯示結束日往前一年，不會因為預設的起始日（最新一天往前一年）晚於結束日而報錯。"""
    home, strategies, _ = dividend_case
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STRATEGIES", str(strategies))
    result = CliRunner().invoke(app, ["signals", "1234", "-s", "breakout", "--end", "2026-03-06", "--no-plot"],
                                env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "2025-03-06 ～ 2026-03-06" in result.output


def test_rule_strategy_name_must_not_shadow_python_strategy(dividend_case, monkeypatch):
    home, strategies, _ = dividend_case
    (strategies / "hmm.yaml").write_text(STRATEGY, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STRATEGIES", str(strategies))
    result = CliRunner().invoke(app, ["backtest", "1234", "-s", "hmm", "--no-plot"], env={"COLUMNS": "200"})
    assert result.exit_code == 1 and "同時是策略設定檔和內建策略" in result.output


def test_compare_command_with_rule_strategies(dividend_case, monkeypatch):
    home, strategies, _ = dividend_case
    (strategies / "breakout_close.yaml").write_text(STRATEGY.replace("突破 105 買", "收盤突破 105 買"),
                                                    encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STRATEGIES", str(strategies))
    result = CliRunner().invoke(app, ["compare", "1234", "-s", "breakout", "-s", "breakout_close", "--no-plot"],
                                env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    for text in ("突破 105 買、跌破 100 賣", "收盤突破 105 買", "買進持有", "訊號用實際成交價判斷，報酬包含股利"):
        assert text in result.output
    assert "⚠" not in result.output and "還原" not in result.output
