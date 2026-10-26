"""條件使用籌碼、本益比。"""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from hermes.cli.main import app
from hermes.data import Cache, DataLoader
from hermes.services import load_fields, rule_backtest, screen, signal_history
from hermes.strategies import StrategyConfigError, parse_strategy, run_rules

DAYS = pd.bdate_range("2026-03-02", periods=8)
CLOSES = [100.0] * 8
FOREIGN = [1000, 2000, -500, 1500, 1200, 900, 300, -500]  # 外資買賣超（張）


def _write(home, stock_id="1234", skip_margin_day=None, with_chips=True):
    cache = Cache(home)
    days = [f"{d:%Y-%m-%d}" for d in DAYS]
    cache.write("TaiwanStockPrice", stock_id, pd.DataFrame([
        {"date": d, "stock_id": stock_id, "Trading_Volume": 1000, "Trading_money": 1000 * c, "open": c, "max": c,
         "min": c, "close": c, "spread": 0.0, "Trading_turnover": 10} for d, c in zip(days, CLOSES)]))
    for ds in ("TaiwanStockDividendResult", "TaiwanStockCapitalReductionReferencePrice", "TaiwanStockSplitPrice"):
        cache.write(ds, stock_id, pd.DataFrame())
    cache.write("TaiwanStockInfo", stock_id, pd.DataFrame([{"stock_id": stock_id, "stock_name": "測試", "type": "twse",
                                                            "industry_category": ""}]))
    if not with_chips:
        return DataLoader(cache)
    # 三大法人原始資料（股）：外資 = Foreign_Investor + Foreign_Dealer_Self
    rows = []
    for d, f in zip(days, FOREIGN):
        shares = f * 1000
        rows += [{"date": d, "stock_id": stock_id, "name": "Foreign_Investor", "buy": max(shares, 0) + 500,
                  "sell": max(-shares, 0) + 500},
                 {"date": d, "stock_id": stock_id, "name": "Foreign_Dealer_Self", "buy": 0, "sell": 0},
                 {"date": d, "stock_id": stock_id, "name": "Investment_Trust", "buy": 3000, "sell": 1000},
                 {"date": d, "stock_id": stock_id, "name": "Dealer_self", "buy": 0, "sell": 250},
                 {"date": d, "stock_id": stock_id, "name": "Dealer_Hedging", "buy": 100, "sell": 0}]
    cache.write("TaiwanStockInstitutionalInvestorsBuySell", stock_id, pd.DataFrame(rows))
    cache.write("TaiwanStockMarginPurchaseShortSale", stock_id, pd.DataFrame([
        {"date": d, "stock_id": stock_id, "MarginPurchaseTodayBalance": 30000 + i, "ShortSaleTodayBalance": 15,
         "MarginPurchaseLimit": 1_000_000} for i, d in enumerate(days) if d != skip_margin_day]))
    cache.write("TaiwanStockShareholding", stock_id, pd.DataFrame([
        {"date": d, "stock_id": stock_id, "ForeignInvestmentSharesRatio": 68.5, "NumberOfSharesIssued": 1e9}
        for d in days]))
    cache.write("TaiwanStockPER", stock_id, pd.DataFrame([
        {"date": d, "stock_id": stock_id, "PER": 0.0 if i == 0 else 20.0, "PBR": 2.0, "dividend_yield": 0.0}
        for i, d in enumerate(days)]))
    return DataLoader(cache)


def test_load_fields_units_and_missing_days(tmp_path):
    loader = _write(tmp_path, skip_margin_day=f"{DAYS[3]:%Y-%m-%d}")
    names = {"foreign_net", "trust_net", "dealer_net", "inst_net", "margin_balance", "foreign_ratio", "per",
             "dividend_yield"}
    fields, warnings = load_fields(loader, "1234", names, loader.prices("1234").index)
    assert fields["foreign_net"].tolist() == FOREIGN  # 股換算成張
    assert fields["trust_net"].iloc[0] == 2.0 and fields["dealer_net"].iloc[0] == -0.15
    assert fields["inst_net"].iloc[0] == pytest.approx(1000 + 2.0 - 0.15)
    assert np.isnan(fields["margin_balance"].iloc[3])  # 缺資料不補值
    assert fields["margin_balance"].iloc[4] == 30004
    assert np.isnan(fields["per"].iloc[0]) and fields["per"].iloc[1] == 20.0  # 官方寫「-」
    assert (fields["dividend_yield"] == 0.0).all()  # 沒有配息是 0，不是空值
    assert warnings == [f"融資融券有 1 個交易日沒有資料（最近的：{DAYS[3]:%Y-%m-%d}），這些日子用到它的條件不會成立"]
    assert load_fields(loader, "1234", set(), DAYS) == (None, [])


def test_chip_condition_and_missing_day(tmp_path):
    loader = _write(tmp_path, skip_margin_day=f"{DAYS[3]:%Y-%m-%d}")
    s = parse_strategy({"name": "外資連 2 天買超", "buy": {"left": "foreign_net", "op": "gt", "right": 0, "days": 2},
                        "sell": {"left": "margin_balance", "op": "gt", "right": 0}})
    fields, _ = load_fields(loader, "1234", s.data_fields, DAYS)
    res = run_rules(s, loader.prices("1234", adjust=False), fields)
    assert res.buy.astype(int).tolist() == [0, 1, 0, 0, 1, 1, 1, 0]
    assert res.sell.astype(int).tolist() == [1, 1, 1, 0, 1, 1, 1, 1]  # 沒有融資資料的那天不成立
    assert list(res.values.columns[5:]) == ["foreign_net", "margin_balance"]


def test_strategy_without_field_data_raises():
    s = parse_strategy({"name": "t", "buy": {"left": "foreign_net", "op": "gt", "right": 0}})
    idx = pd.bdate_range("2026-01-05", periods=3)
    prices = pd.DataFrame({c: 10.0 for c in ("open", "high", "low", "close", "volume")}, index=idx)
    with pytest.raises(ValueError, match="用到 foreign_net"):
        run_rules(s, prices)


def test_config_rules_for_fields():
    with pytest.raises(StrategyConfigError, match="不能和價格欄位、籌碼或本益比欄位相同"):
        parse_strategy({"name": "t", "indicators": {"per": {"type": "sma", "period": 5}},
                        "buy": {"left": "close", "op": "gt", "right": "per"}})
    with pytest.raises(StrategyConfigError, match="hermes indicators 查看"):
        parse_strategy({"name": "t", "buy": {"left": "foreign", "op": "gt", "right": 0}})
    s = parse_strategy({"name": "t", "buy": {"all": [
        {"left": "foreign_net", "op": "gt", "right": 0},
        {"left": {"ref": "trust_net", "ago": 1}, "op": "gt", "right": 0}]}})
    assert s.data_fields == {"foreign_net", "trust_net"}
    assert s.same_day_fields == {"foreign_net"}
    # 形成中的條件用到的欄位也要讀進來；它不影響買賣，所以不算「收盤成交時還不知道」，暖機也不等它
    f = parse_strategy({"name": "t", "buy": {"left": {"ref": "trust_net", "ago": 1}, "op": "gt", "right": 0},
                        "forming": {"left": "foreign_net", "op": "gt", "right": 0}})
    assert f.data_fields == {"foreign_net", "trust_net"}
    assert f.same_day_fields == set()
    idx = pd.bdate_range("2026-01-05", periods=4)
    prices = pd.DataFrame({c: 10.0 for c in ("open", "high", "low", "close", "volume")}, index=idx)
    fields = pd.DataFrame({"foreign_net": [np.nan, np.nan, 5.0, -1.0], "trust_net": 1.0}, index=idx)
    res = run_rules(f, prices, fields)
    assert res.forming.tolist() == [False, False, True, False]
    assert res.forming_only == ("foreign_net",)
    assert res.warmup_end == idx[1]  # 只等前 1 天的投信買賣超


def test_backtest_on_close_cannot_use_same_day_chips(tmp_path):
    loader = _write(tmp_path)
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    (strategies / "chips.yaml").write_text(
        "name: 外資買超\nbuy: {left: foreign_net, op: gt, right: 0}\nsell: {left: foreign_net, op: lt, right: 0}\n",
        encoding="utf-8")
    (strategies / "lagged.yaml").write_text(
        "name: 前一天外資買超\nbuy: {left: {ref: foreign_net, ago: 1}, op: gt, right: 0}\n"
        "sell: {left: {ref: foreign_net, ago: 1}, op: lt, right: 0}\n", encoding="utf-8")
    cmp = signal_history("1234", "chips", loader=loader, directory=strategies)
    with pytest.raises(StrategyConfigError, match="收盤後才公布"):
        rule_backtest(cmp, execution="close", loader=loader)
    assert not rule_backtest(cmp, execution="open", loader=loader).result.trades.empty
    lagged = signal_history("1234", "lagged", loader=loader, directory=strategies)
    assert not rule_backtest(lagged, execution="close", loader=loader).result.trades.empty


def test_screen_reports_missing_chip_data_per_strategy(tmp_path):
    loader = _write(tmp_path, with_chips=False)
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    (strategies / "chips.yaml").write_text("name: 外資買超\nbuy: {left: foreign_net, op: gt, right: 0}\n",
                                           encoding="utf-8")
    (strategies / "price.yaml").write_text("name: 收盤大於 90\nbuy: {left: close, op: gt, right: 90}\n",
                                           encoding="utf-8")
    report = screen(["1234"], loader=loader, directory=strategies)
    assert [r.strategy_id for r in report.rows] == ["price"]
    assert len(report.errors) == 1 and "chips" in report.errors[0][0]


def test_indicators_command_lists_fields():
    result = CliRunner().invoke(app, ["indicators"], env={"COLUMNS": "200"})
    assert "foreign_net" in result.output and "外資買賣超" in result.output
