"""指標組和 hermes tech。

指標的算法由 tests/test_indicator_reference.py 對照；這裡檢查指標組的讀取、名稱，以及取出的是哪一天、哪種價格的值。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from conftest import make_prices
from hermes.cli.main import app
from hermes.data import Cache, DataError, DataLoader
from hermes.services import IndicatorSetNotFoundError, indicator_snapshot
from hermes.strategies import StrategyConfigError, load_indicator_set, load_strategy
from hermes.technical import CATALOG

COMMON = Path(__file__).resolve().parents[1] / "indicators" / "common.yaml"


@pytest.fixture
def setup(tmp_path):
    """300 個交易日，第 200 天除息（參考價 = 前一天收盤 − 2）。"""
    p = make_prices(n=300, seed=3)
    p.index = pd.bdate_range("2024-01-01", periods=300)  # 資料從 2020 年起才讀得到
    days = p.index.strftime("%Y-%m-%d")
    cache = Cache(tmp_path / "home")
    cache.write("TaiwanStockPrice", "1234", pd.DataFrame([
        {"date": d, "stock_id": "1234", "Trading_Volume": r.volume, "Trading_money": r.close * r.volume,
         "open": r.open, "max": r.high, "min": r.low, "close": r.close, "spread": 0.0, "Trading_turnover": 10}
        for d, r in zip(days, p.itertuples())]))
    before = float(p["close"].iloc[199])
    cache.write("TaiwanStockDividendResult", "1234", pd.DataFrame([
        {"date": days[200], "stock_id": "1234", "before_price": before, "after_price": before - 2,
         "stock_or_cache_dividend": "息"}]))
    cache.write("TaiwanStockCapitalReductionReferencePrice", "1234", pd.DataFrame())
    cache.write("TaiwanStockSplitPrice", "1234", pd.DataFrame())
    cache.write("TaiwanStockInfo", "1234", pd.DataFrame([{"stock_id": "1234", "stock_name": "測試", "type": "twse",
                                                        "industry_category": ""}]))
    sets = tmp_path / "indicators"
    sets.mkdir()
    (sets / "common.yaml").write_text(COMMON.read_text(encoding="utf-8"), encoding="utf-8")
    (sets / "mine.yaml").write_text("name: 自訂\nindicators:\n  rsi_mine: {type: rsi, period: 10}\n"
                                    "  rsi14: {type: rsi, period: 14}\n", encoding="utf-8")
    return tmp_path, sets, DataLoader(cache)


def test_common_set_is_valid():
    s = load_indicator_set(COMMON)
    assert len(s.indicators) >= 30
    assert {cfg.type for cfg in s.indicators.values()} <= set(CATALOG)
    # 每個參數都寫在檔案裡：指標目錄沒有預設值，讀得進來就代表都寫了
    for cfg in s.indicators.values():
        CATALOG[cfg.type].resolve_params(cfg.params)


def test_snapshot_takes_the_right_day_and_prices(setup):
    _, sets, loader = setup
    raw, adjusted = loader.prices("1234", adjust=False), loader.prices("1234", adjust=True)
    for day in (None, raw.index[250]):
        for prices, mode in ((raw, False), (adjusted, True)):
            snap = indicator_snapshot("1234", "common", day, loader, sets, adjusted=mode)
            when = raw.index[-1] if day is None else day
            assert snap.date == when and snap.close == prices.at[when, "close"] and snap.adjusted == mode
            for name, cfg in snap.indicator_set.indicators.items():
                spec = CATALOG[cfg.type]
                out = spec.compute(prices, **spec.resolve_params(cfg.params))
                for line in spec.outputs or (None,):
                    expected = (out[line] if line else out).loc[when]
                    got = snap.values[f"{name}.{line}" if line else name]
                    assert (np.isnan(expected) and np.isnan(got)) or got == expected, (name, line, mode)
    # 除息前的均線，用還原價格算會比較低
    raw_snap = indicator_snapshot("1234", "common", None, loader, sets)
    adj_snap = indicator_snapshot("1234", "common", None, loader, sets, adjusted=True)
    assert adj_snap.values["ma240"] < raw_snap.values["ma240"]


def test_same_type_with_different_names(setup):
    _, sets, loader = setup
    snap = indicator_snapshot("1234", "mine", None, loader, sets)
    assert snap.values["rsi_mine"] != snap.values["rsi14"]
    common = indicator_snapshot("1234", "common", None, loader, sets)
    assert snap.values["rsi14"] == common.values["rsi14"]


def test_errors(setup):
    _, sets, loader = setup
    with pytest.raises(IndicatorSetNotFoundError, match="common、mine"):
        indicator_snapshot("1234", "nope", None, loader, sets)
    with pytest.raises(DataError, match="沒有 1234 在 2030-01-01"):
        indicator_snapshot("1234", "common", "2030-01-01", loader, sets)


def test_duplicate_names_are_rejected(tmp_path):
    """YAML 預設會讓重複的名稱默默蓋掉前一個；指標組和策略設定檔都要直接報錯。"""
    path = tmp_path / "dup.yaml"
    path.write_text("name: t\nindicators:\n  rsi: {type: rsi, period: 14}\n  rsi: {type: rsi, period: 6}\n",
                    encoding="utf-8")
    with pytest.raises(StrategyConfigError, match="rsi 重複"):
        load_indicator_set(path)
    path.write_text("name: t\nindicators:\n  ma: {type: sma, period: 5}\n  ma: {type: sma, period: 20}\n"
                    "buy: {left: close, op: gt, right: ma}\n", encoding="utf-8")
    with pytest.raises(StrategyConfigError, match="ma 重複"):
        load_strategy(path)


def test_missing_params_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: t\nindicators:\n  rsi: {type: rsi}\n", encoding="utf-8")
    with pytest.raises(StrategyConfigError, match="需要指定 period"):
        load_indicator_set(path)


def test_tech_command(setup, monkeypatch):
    tmp_path, sets, _ = setup
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_INDICATORS", str(sets))
    result = CliRunner().invoke(app, ["tech", "1234"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    for text in ("常用指標", "kd.k", "macd.hist", "（實際成交價）", "RSI（14 日）"):
        assert text in result.output
    assert "還原" not in result.output
    result = CliRunner().invoke(app, ["tech", "1234", "-i", "mine"], env={"COLUMNS": "200"})
    assert result.exit_code == 0 and "rsi_mine" in result.output
    result = CliRunner().invoke(app, ["tech", "1234", "-i", "nope"], env={"COLUMNS": "200"})
    assert result.exit_code == 1 and "找不到指標組" in result.output
