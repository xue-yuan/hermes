from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from hermes.data import Cache, DataLoader
from hermes.services import (
    StrategyNotFoundError,
    list_rule_strategies,
    resolve_strategy_path,
    signal_history,
)
from hermes.strategies import StrategyConfigError, describe, load_strategy, parse_strategy, streak
from hermes.technical import CATALOG

REPO_STRATEGIES = Path(__file__).resolve().parents[1] / "strategies"

STATE_RULE = """
name: 收盤在 10 元之上
indicators:
  ma2: {type: sma, period: 2}
buy:  {left: close, op: gt, right: 10}
sell: {left: close, op: crosses_below, right: ma2}
"""


def test_streak_counts_consecutive_days():
    flags = pd.Series([False, True, True, False, True, True, True])
    assert streak(flags).tolist() == [0, 1, 2, 0, 1, 2, 3]


def test_describe_rule():
    s = parse_strategy({"name": "t", "indicators": {"cci": {"type": "cci", "period": 20}, "ma60": {"type": "sma", "period": 60}},
                        "buy": {"all": [{"left": "cci", "op": "crosses_above", "right": -100},
                                        {"left": "close", "op": "gt", "right": "ma60"}]},
                        "sell": {"any": [{"left": "cci", "op": "crosses_below", "right": 100},
                                         {"left": "close", "op": "lt", "right": "ma60", "days": 3}]}})
    assert describe(s.buy) == "cci 由下往上穿越 -100 且 close > ma60"
    assert describe(s.sell) == "cci 由上往下穿越 100 或 close < ma60（連續 3 天）"
    nested = parse_strategy({"name": "t", "buy": {"all": [
        {"left": "close", "op": "gt", "right": 1},
        {"any": [{"left": "close", "op": "lt", "right": 5}, {"left": "volume", "op": "gte", "right": 100}]}]}})
    assert describe(nested.buy) == "close > 1 且 （close < 5 或 volume ≥ 100）"


def test_repo_strategy_files_are_valid():
    for path, item in list_rule_strategies(REPO_STRATEGIES):
        assert not isinstance(item, StrategyConfigError), f"{path.name}: {item}"


def test_resolve_strategy_path(tmp_path):
    (tmp_path / "a.yaml").write_text(STATE_RULE, encoding="utf-8")
    assert resolve_strategy_path("a", tmp_path) == tmp_path / "a.yaml"
    assert resolve_strategy_path(str(tmp_path / "a.yaml"), Path("/nonexistent")) == tmp_path / "a.yaml"
    with pytest.raises(StrategyNotFoundError, match="裡有：a"):
        resolve_strategy_path("b", tmp_path)


def test_list_rule_strategies_reports_bad_files(tmp_path):
    (tmp_path / "good.yaml").write_text(STATE_RULE, encoding="utf-8")
    (tmp_path / "bad.yaml").write_text("name: x\nbuy: {left: close, op: bigger, right: 1}\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("不是策略", encoding="utf-8")
    items = dict((p.stem, item) for p, item in list_rule_strategies(tmp_path))
    assert set(items) == {"good", "bad"}
    assert isinstance(items["bad"], StrategyConfigError) and "buy.op" in str(items["bad"])


def _cache_with_prices(tmp_path, closes):
    days = pd.bdate_range("2026-01-05", periods=len(closes)).strftime("%Y-%m-%d")
    rows = [{"date": d, "stock_id": "9999", "Trading_Volume": 1000, "Trading_money": 1000 * c, "open": c,
             "max": c, "min": c, "close": c, "spread": 0.0, "Trading_turnover": 10} for d, c in zip(days, closes)]
    cache = Cache(tmp_path / "home")
    cache.write("TaiwanStockPrice", "9999", pd.DataFrame(rows))
    cache.write("TaiwanStockDividendResult", "9999", pd.DataFrame())
    return cache


def test_signal_history_table(tmp_path):
    (tmp_path / "s.yaml").write_text(STATE_RULE, encoding="utf-8")
    cache = _cache_with_prices(tmp_path, [9, 11, 12, 13, 9, 11, 8])
    # 測試資料的跳空超過漲跌幅限制，還原權息會把它當成除權息調整，所以用實際成交價
    hist = signal_history("9999", "s", adjust=False, loader=DataLoader(cache), directory=tmp_path)
    assert hist.indicator_columns == ["ma2"] and not hist.adjusted

    # 收盤 9、11、12、13、9、11、8（1/5～1/13）；ma2 為 —、10、11.5、12.5、11、10、9.5
    signals = hist.table()
    assert list(signals.index.strftime("%m-%d")) == ["01-06", "01-07", "01-08", "01-09", "01-12", "01-13"]
    assert signals["buy_streak"].tolist() == [1, 2, 3, 0, 1, 0]
    assert signals["sell_streak"].tolist() == [0, 0, 0, 1, 0, 1]
    assert len(hist.table(every_day=True)) == 7
    assert hist.last_trigger("buy") == pd.Timestamp("2026-01-12")
    assert hist.last_trigger("sell") == pd.Timestamp("2026-01-13")
    # 顯示區間只影響列出的日子，計算用全部歷史
    assert hist.table(start=date(2026, 1, 7), every_day=True)["buy_streak"].iloc[0] == 2


def test_plot_signals_writes_png(tmp_path, prices):
    from hermes.cli.plots import plot_signals
    from hermes.strategies import run_rules

    s = load_strategy(REPO_STRATEGIES / "connors_rsi2.yaml")
    res = run_rules(s, prices)
    values = res.values.iloc[-250:]
    path = plot_signals(values, res.buy_streak.iloc[-250:], res.sell_streak.iloc[-250:], s, "測試",
                        tmp_path / "signals.png")
    assert path.exists() and path.stat().st_size > 0


def test_plot_signals_draws_volume_threshold(tmp_path, prices, monkeypatch):
    """條件裡有成交量的門檻時，成交量另外一個子圖，畫出門檻。"""
    from hermes.cli import plots
    from hermes.strategies import run_rules

    s = parse_strategy({"name": "量大", "buy": {"left": "volume", "op": "gt", "right": 5e6},
                        "sell": {"left": "close", "op": "lt", "right": 1}})
    res = run_rules(s, prices)
    figs = []
    monkeypatch.setattr(plots, "_finish", lambda fig, path: figs.append(fig) or path)
    plots.plot_signals(res.values, res.buy_streak, res.sell_streak, s, "測試", tmp_path / "volume.png")
    ax = figs[0].axes[1]
    assert ax.get_ylabel() == "volume"
    assert [line.get_label() for line in ax.get_lines()] == ["volume", "門檻 5e+06"]


def test_overlay_lines():
    assert CATALOG["sma"].on_price_chart()
    assert not CATALOG["cci"].on_price_chart()
    assert CATALOG["bollinger"].on_price_chart("upper") and not CATALOG["bollinger"].on_price_chart("pct_b")
