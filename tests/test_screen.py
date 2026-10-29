import pandas as pd
import pytest
from typer.testing import CliRunner

from hermes.cli.main import app
from hermes.data import Cache, DataLoader
from hermes.services import gap_warnings, screen

EVENT = "name: 事件\nbuy:  {left: close, op: crosses_above, right: 102}\nsell: {left: close, op: crosses_below, right: 98}\n"
STATE = "name: 狀態\nbuy: {left: close, op: gt, right: 102}\n"


def _write_prices(cache, sid, name, closes, start="2026-01-05"):
    days = pd.bdate_range(start, periods=len(closes)).strftime("%Y-%m-%d")
    cache.write("TaiwanStockPrice", sid, pd.DataFrame([
        {"date": d, "stock_id": sid, "Trading_Volume": 1000, "Trading_money": 1000 * c, "open": c, "max": c,
         "min": c, "close": c, "spread": 0.0, "Trading_turnover": 10} for d, c in zip(days, closes)]))
    cache.write("TaiwanStockInfo", sid, pd.DataFrame([{"stock_id": sid, "stock_name": name, "type": "twse",
                                                       "industry_category": ""}]))


@pytest.fixture
def setup(tmp_path):
    home, strategies = tmp_path / "home", tmp_path / "strategies"
    strategies.mkdir()
    (strategies / "event.yaml").write_text(EVENT, encoding="utf-8")
    (strategies / "state.yaml").write_text(STATE, encoding="utf-8")
    cache = Cache(home)
    _write_prices(cache, "1111", "甲", [100.0] * 10 + [105.0])  # 最後一天剛站上 102
    _write_prices(cache, "2222", "乙", [100.0] * 7 + [104.0, 105.0, 106.0])  # 已在 102 之上 3 天，且少一天資料
    return home, strategies, DataLoader(cache)


def test_screen_sections(setup):
    _, strategies, loader = setup
    report = screen(["1111", "2222", "3333"], loader=loader, directory=strategies)
    by = {(r.stock_id, r.strategy_id): r for r in report.rows}
    assert set(by) == {("1111", "event"), ("1111", "state"), ("2222", "event"), ("2222", "state")}

    assert {(r.stock_id, r.strategy_id, side) for r, side in report.triggered_today()} == {
        ("1111", "event", "buy"), ("1111", "state", "buy")}
    assert [(r.stock_id, r.strategy_id, side, n) for r, side, n in report.ongoing()] == [("2222", "state", "buy", 3)]

    assert list(report.stale()) == ["2222"]
    assert [who for who, _ in report.errors] == ["股票 3333"]


def test_screen_reports_bad_strategy_file(setup):
    _, strategies, loader = setup
    (strategies / "bad.yaml").write_text("name: x\nbuy: {left: close, op: bigger, right: 1}\n", encoding="utf-8")
    report = screen(["1111"], loader=loader, directory=strategies)
    assert {r.strategy_id for r in report.rows} == {"event", "state"}
    assert [who for who, _ in report.errors] == ["策略 bad"]


def test_screen_command(setup, monkeypatch):
    home, strategies, _ = setup
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STRATEGIES", str(strategies))
    result = CliRunner().invoke(app, ["screen", "1111", "2222"], env={"COLUMNS": "160"})
    assert result.exit_code == 0, result.output
    for text in ("① 新觸發（2026-01-19）", "② 條件仍成立", "2222 的資料只到 2026-01-16"):
        assert text in result.output


def test_screen_warns_unexplained_gap(setup, monkeypatch):
    """沒有官方紀錄的跳空會讓還原價格不正確：還原模式才提醒（每則提醒都是（對象, 訊息）），CLI 能正常列出。
    實際成交價模式用不到還原價格，不提醒。"""
    home, strategies, loader = setup
    _write_prices(loader.cache, "4444", "丁", [100.0] * 8 + [130.0] * 3)
    assert screen(["4444"], loader=loader, directory=strategies).warnings == []
    report = screen(["4444"], loader=loader, directory=strategies, adjusted=True)
    assert report.warnings == [("股票 4444", "2026-01-15 跳空 +30.0% 超過漲跌幅限制，"
                                              "但沒有官方除權息／減資／分割紀錄，還原價格可能不正確")]
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STRATEGIES", str(strategies))
    result = CliRunner().invoke(app, ["screen", "4444", "--adjust"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "2026-01-15 跳空 +30.0%" in result.output
    assert "跳空" not in CliRunner().invoke(app, ["screen", "4444"], env={"COLUMNS": "200"}).output


def test_backtest_warns_unexplained_gap_in_both_modes(setup):
    """回測報酬一律用還原價格算，所以兩種模式都提醒；實際成交價模式寫「回測報酬」。"""
    _, _, loader = setup
    _write_prices(loader.cache, "4444", "丁", [100.0] * 8 + [130.0] * 3)
    [raw] = gap_warnings(loader, "4444", False, returns=True)
    assert raw.endswith("回測報酬可能不正確")
    [adj] = gap_warnings(loader, "4444", True, returns=True)
    assert adj.endswith("還原價格可能不正確")
    assert gap_warnings(loader, "4444", False) == []


FORMING = ("name: 形成中\nbuy: {left: close, op: crosses_above, right: 105.5}\n"
           "sell: {left: close, op: gt, right: 104.5}\n"
           "forming: {left: close, op: gt, right: {ref: close, ago: 8}}\n")


@pytest.fixture
def forming_dir(setup, tmp_path):
    d = tmp_path / "forming"
    d.mkdir()
    (d / "forming.yaml").write_text(FORMING, encoding="utf-8")
    return d


def test_screen_forming(setup, forming_dir):
    """形成中：條件今天成立、今天沒有觸發買進；天數是連續成立的天數；數值是今天的值。"""
    _, _, loader = setup
    report = screen(["1111", "2222"], loader=loader, directory=forming_dir)
    by = {r.stock_id: r for r in report.rows}
    # 1111：最後一天 105 比 8 天前的 100 高，第 1 天；收盤沒有超過 105.5，沒有買進。
    #       賣出條件（> 104.5）今天也成立，照樣列在形成中，只有觸發買進才不列
    # 2222：105、106 比 8 天前的 100 高，連續 2 天（104 那天還沒有 8 天前的資料）；
    #       但今天 106 穿越 105.5 觸發買進，不列在形成中
    assert (by["1111"].forming_streak, by["2222"].forming_streak) == (1, 2)
    assert by["2222"].streak["buy"] == 1 and by["1111"].streak == {"buy": 0, "sell": 1}
    assert [(r.stock_id, days) for r, days in report.forming()] == [("1111", 1)]
    assert by["1111"].values[["close", "close（前8天）"]].tolist() == [105.0, 100.0]
    # 沒有寫形成中條件的策略：天數一律是 0
    plain = screen(["1111", "2222"], loader=loader, directory=setup[1])
    assert plain.forming() == [] and {r.forming_streak for r in plain.rows} == {0}


def test_screen_command_forming(setup, forming_dir, monkeypatch):
    home, strategies, _ = setup
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STRATEGIES", str(forming_dir))
    out = CliRunner().invoke(app, ["screen", "1111", "2222"], env={"COLUMNS": "200"}).output
    assert "③ 形成中" in out
    row = next(line for line in out.split("③ 形成中")[1].splitlines() if "1111" in line)
    assert "close 105、close（前8天） 100" in row and "1 天" in row
    # 寫了形成中條件、但今天沒有成立
    assert "③ 沒有形成中的型態" in CliRunner().invoke(app, ["screen", "2222"], env={"COLUMNS": "200"}).output
    # 策略都沒寫形成中條件：不顯示這一段
    monkeypatch.setenv("HERMES_STRATEGIES", str(strategies))
    assert "③" not in CliRunner().invoke(app, ["screen", "1111", "2222"], env={"COLUMNS": "200"}).output
