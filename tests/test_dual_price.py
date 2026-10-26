"""判斷用哪種價格：預設用實際成交價，還原模式用還原價格；每個模式只算、只顯示自己那種價格。

情境：前 10 天收 100，第 11 天除息 5 元（參考價 95），之後都收 95。策略：收盤 < 10 日均線。
- 實際成交價：今天的 10 日均線還包含 6 天的 100 =（600 + 380）/ 10 = 98，收盤 95 < 98 → 成立。
  明天收 c：均線 =（500 + 380 + c）/ 10，c < 均線 ⟺ c < 97.78 → 最高 97.7（100 元以下每檔 0.1）。
- 還原價格：過去的 100 都還原成 95，均線 = 95，收盤 95 不小於 95 → 不成立。明天要 c < 95 → 最高 94.9。
"""

import pandas as pd
import pytest
from typer.testing import CliRunner

from hermes.cli.main import app
from hermes.data import Cache, DataLoader
from hermes.services import screen, signal_history

STRATEGY = "name: 跌破十日線\nindicators:\n  ma10: {type: sma, period: 10}\nbuy: {left: close, op: lt, right: ma10}\n"


@pytest.fixture
def setup(tmp_path):
    days = pd.bdate_range("2026-03-02", periods=14).strftime("%Y-%m-%d")
    closes = [100.0] * 10 + [95.0] * 4
    cache = Cache(tmp_path / "home")
    cache.write("TaiwanStockPrice", "1234", pd.DataFrame([
        {"date": d, "stock_id": "1234", "Trading_Volume": 1000, "Trading_money": 1000 * c, "open": c, "max": c,
         "min": c, "close": c, "spread": 0.0, "Trading_turnover": 10} for d, c in zip(days, closes)]))
    cache.write("TaiwanStockDividendResult", "1234", pd.DataFrame([
        {"date": days[10], "stock_id": "1234", "before_price": 100.0, "after_price": 95.0,
         "stock_or_cache_dividend": "息"}]))
    cache.write("TaiwanStockCapitalReductionReferencePrice", "1234", pd.DataFrame())
    cache.write("TaiwanStockSplitPrice", "1234", pd.DataFrame())
    cache.write("TaiwanStockInfo", "1234", pd.DataFrame([{"stock_id": "1234", "stock_name": "測試", "type": "twse",
                                                        "industry_category": ""}]))
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    (strategies / "ma10.yaml").write_text(STRATEGY, encoding="utf-8")
    return tmp_path / "home", strategies, DataLoader(cache)


def test_actual_prices_by_default(setup):
    _, strategies, loader = setup
    hist = signal_history("1234", "ma10", loader=loader, directory=strategies)
    table = hist.table(every_day=True)
    assert not hist.adjusted and hist.result.values["ma10"].iloc[-1] == pytest.approx(98.0)
    assert table["buy"].iloc[-1] and table["buy_streak"].iloc[-1] == 4  # 除息當天起連續 4 天
    assert table["close"].iloc[0] == 100.0
    assert not any(c.startswith(("other_", "adj_")) or c == "differs" for c in table.columns)


def test_adjusted_mode(setup):
    _, strategies, loader = setup
    hist = signal_history("1234", "ma10", adjust=True, loader=loader, directory=strategies)
    table = hist.table(every_day=True)
    assert hist.adjusted and hist.result.values["ma10"].iloc[-1] == pytest.approx(95.0)
    assert not table["buy"].any()
    assert table["close"].iloc[0] == pytest.approx(95.0)  # 除息前的 100 還原成 95


def test_screen_modes(setup):
    _, strategies, loader = setup
    [row] = screen(["1234"], loader=loader, directory=strategies).rows
    assert row.streak == {"buy": 4, "sell": 0}
    report = screen(["1234"], loader=loader, directory=strategies, adjusted=True)
    [row] = report.rows
    assert report.adjusted and row.streak == {"buy": 0, "sell": 0}
    assert report.triggered_today() == [] and report.ongoing() == []


def test_cli_shows_only_the_chosen_price(setup, monkeypatch, tmp_path):
    home, strategies, _ = setup
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STRATEGIES", str(strategies))
    sets = tmp_path / "indicators"
    sets.mkdir()
    (sets / "ma.yaml").write_text("name: 均線\nindicators:\n  ma10: {type: sma, period: 10}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_INDICATORS", str(sets))
    env = {"COLUMNS": "200"}

    def run(*args: str) -> str:
        out = CliRunner().invoke(app, list(args), env=env)
        assert out.exit_code == 0, out.output
        return out.output

    out = run("screen", "1234")
    assert "價格：實際成交價" in out and "③" not in out and "4 天" in out and "還原" not in out
    out = run("screen", "1234", "--adjust")
    assert "價格：還原價格" in out and "② 沒有持續成立中的條件" in out and "實際成交價" not in out

    out = run("signals", "1234", "-s", "ma10", "--no-plot")
    assert "價格：實際成交價" in out and "買（第 4 天）" in out and "還原" not in out
    out = run("signals", "1234", "-s", "ma10", "--adjust", "--no-plot")
    assert "價格：還原價格" in out and "這段期間沒有任何訊號" in out and "實際成交價" not in out

    out = run("tech", "1234", "-i", "ma")
    assert "收盤 95.00（實際成交價）" in out and "98.00" in out and "還原" not in out
    out = run("tech", "1234", "-i", "ma", "--adjust")
    assert "收盤 95.00（還原價格）" in out and "95.00" in out and "98.00" not in out and "實際成交價" not in out
