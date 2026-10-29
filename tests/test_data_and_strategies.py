import numpy as np
import pandas as pd
import pytest

from hermes.data.errors import MissingDataError
from hermes.data.loader import DataLoader
from hermes.data.store import Cache
from hermes.strategies import REGISTRY, StrategyContext, get_strategy, parse_params
from hermes.strategies.base import stateful


def _offline_loader(tmp_path, price_rows, dividend_rows):
    cache = Cache(tmp_path)
    cache.write("TaiwanStockPrice", "9999", pd.DataFrame(price_rows))
    cache.write("TaiwanStockDividendResult", "9999", pd.DataFrame(dividend_rows))
    return DataLoader(cache=cache)


def test_loader_raises_when_data_not_downloaded(tmp_path):
    with pytest.raises(MissingDataError, match="hermes fetch 1234"):
        DataLoader(cache=Cache(tmp_path)).prices("1234")


def _price_row(d, o, c):
    return {"date": d, "stock_id": "9999", "Trading_Volume": 1000, "Trading_money": 1000 * c, "open": o,
            "max": max(o, c), "min": min(o, c), "close": c, "spread": 0.0, "Trading_turnover": 10}


def test_adjustment_uses_only_official_records(tmp_path):
    """只用官方紀錄還原，不估算；沒有紀錄的跳空保持原樣，列在 unexplained_gaps。"""
    days = pd.bdate_range("2021-03-01", periods=12).strftime("%Y-%m-%d")
    closes = [100, 101, 102, 103, 104, 105, 52, 53, 54, 52, 53, 54]
    opens = [100, 100, 101, 102, 103, 104, 52.5, 52, 53, 51, 52, 53]
    rows = [_price_row(d, o, c) for d, o, c in zip(days, opens, closes)]
    # 第 6 天跳空 −50% 但沒有官方紀錄；第 9 天有官方除息紀錄
    divs = [{"date": days[9], "stock_id": "9999", "before_price": 54.0, "after_price": 51.0,
             "stock_or_cache_dividend": "息"}]
    loader = _offline_loader(tmp_path, rows, divs)
    p = loader.prices("9999")
    assert p["adj_factor"].iloc[0] == pytest.approx(51 / 54)
    assert p["adj_factor"].iloc[-1] == 1.0
    assert p["close"].pct_change().iloc[9] == pytest.approx(52 / 51 - 1)
    assert p["close"].pct_change().iloc[6] == pytest.approx(52 / 105 - 1)  # 沒有紀錄，不調整
    assert p["raw_close"].iloc[0] == 100
    gaps = loader.unexplained_gaps("9999")
    assert list(gaps["date"].dt.strftime("%Y-%m-%d")) == [days[6]]
    assert list(loader.adjustments("9999")["source"]) == ["除權息"]


def test_only_no_trade_days_is_missing_data(tmp_path):
    """FinMind 把沒有成交的日子記成全 0；全部都是這種日子時，等於沒有股價資料，要明確報錯。"""
    days = pd.bdate_range("2021-03-01", periods=3).strftime("%Y-%m-%d")
    loader = _offline_loader(tmp_path, [_price_row(d, 0, 0) for d in days], [])
    with pytest.raises(MissingDataError, match="都是沒有成交的日子"):
        loader.prices("9999")


def test_codes_with_letters_are_adjusted_too(tmp_path):
    """代號有英文字母的上市股票（例如 00632R）一樣照官方紀錄還原。"""
    cache = Cache(tmp_path)
    days = pd.bdate_range("2021-03-01", periods=4).strftime("%Y-%m-%d")
    cache.write("TaiwanStockPrice", "00632R", pd.DataFrame(
        [_price_row(d, 10, c) | {"stock_id": "00632R"} for d, c in zip(days, [10, 10, 9.5, 9.6])]))
    cache.write("TaiwanStockDividendResult", "00632R", pd.DataFrame([
        {"date": days[2], "stock_id": "00632R", "before_price": 10.0, "after_price": 9.5,
         "stock_or_cache_dividend": "息"}]))
    p = DataLoader(cache=cache).prices("00632R")
    assert p["adj_factor"].iloc[0] == pytest.approx(0.95)
    assert p["close"].pct_change().iloc[2] == pytest.approx(0.0)  # 除息的跳空已經還原


def test_cache_keeps_data_from_history_start(tmp_path):
    cache = Cache(tmp_path)
    cache.write("TaiwanStockPrice", "9999", pd.DataFrame([_price_row("2019-12-31", 10, 10),
                                                          _price_row("2020-01-02", 10, 11)]))
    assert list(cache.read("TaiwanStockPrice", "9999")["date"]) == ["2020-01-02"]
    info = pd.DataFrame([{"stock_id": "9999", "stock_name": "測試", "date": "2019-01-01"}])
    cache.write("TaiwanStockInfo", "9999", info)
    assert len(cache.read("TaiwanStockInfo", "9999")) == 1  # 基本資料的 date 是更新日期，不篩選


def test_buy_hold_and_registry(prices):
    pos = get_strategy("buy_hold").run(StrategyContext(prices=prices))
    assert (pos == 1.0).all() and pos.index.equals(prices.index)
    # 只剩買進持有和機器學習策略；技術面的 Python 策略已經拿掉，改用策略設定檔
    assert {s.category for s in REGISTRY.values()} == {"基準", "機器學習"}


def test_parse_params_coercion():
    spec = get_strategy("hmm")
    assert parse_params(spec, ["states=3", "continuous=true"]) == {"states": 3, "continuous": True}
    with pytest.raises(ValueError):
        parse_params(spec, ["nope=1"])


def test_meta_requires_rule_strategy_as_primary(prices):
    with pytest.raises(ValueError, match="primary"):
        get_strategy("meta").run(StrategyContext(prices=prices))
    asked = []

    def rule_position(ref):
        asked.append(ref)
        return pd.Series((np.arange(len(prices)) // 20) % 2, index=prices.index, dtype=float)  # 每 20 天進出一次

    ctx = StrategyContext(prices=prices, rule_position=rule_position)
    pos = get_strategy("meta").run(ctx, primary="my_rule", train_min=300, step=100, exogenous=False,
                                   changepoint=False)
    assert asked == ["my_rule"]
    assert pos.index.equals(prices.index)


def test_stateful_entry_exit():
    entry = pd.Series([False, True, False, False, True, False])
    exit_ = pd.Series([False, False, False, True, False, False])
    assert stateful(entry, exit_).tolist() == [0, 1, 1, 0, 1, 1]


def test_hmm_strategy_runs(prices):
    pos = get_strategy("hmm").run(StrategyContext(prices=prices), train_min=300, step=100)
    assert pos.first_valid_index() is not None
    assert set(np.unique(pos.dropna())) <= {0.0, 1.0}
