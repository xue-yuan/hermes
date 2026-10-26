"""成交量還原：股數改變、但沒有證交所股數比例時的處理（有比例的情況見 test_share_ratios.py）。

照 CRSP 的做法，成交量只在股數改變時才要調整；沒有官方的股數比例就不估算。所以：
- 沒有比例的除權、減資、分割以前，還原成交量是空值，之後的不用調整；只有「息」的除息不影響成交量。
- 用到成交量的指標，窗口裡有空值就沒有值；OBV 是累加值，之後都沒有值。照定義用不到成交量的日子
  （OBV 收盤沒漲跌、MFI 典型價格沒變）不受影響。
- 實際成交價不受影響。還原模式下，策略用到的值算不出來時，用到它的條件不成立，並顯示警告。
- 還原模式的 tech 顯示「無法計算」，verify 列為注意，機器學習不跑（它會把空值補成中位數）。
"""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from conftest import make_prices
from hermes.backtest import CostModel
from hermes.cli.main import app
from hermes.data import Cache, DataLoader
from hermes.ml.features import exogenous_features, price_features
from hermes.services import indicator_snapshot, rule_backtest, screen, signal_history
from hermes.services.verify import check_share_changes
from hermes.strategies import StrategyConfigError
from hermes.technical import CATALOG

N, EVENT = 300, 150  # 第 150 天有事件
VOLUME_INDICATORS = {"obv": {}, "mfi": {"period": 14}, "cmf": {"period": 20}, "vwap": {"period": 20},
                     "volume_ratio": {"period": 20}}

STRATEGIES = {
    # 用到成交量
    "vol": "name: 量比\nindicators:\n  vr: {type: volume_ratio, period: 20}\n"
           "buy: {left: vr, op: gt, right: 1.2}\nsell: {left: vr, op: lt, right: 0.8}\n",
    "obv": "name: OBV 上升\nindicators:\n  obv: {type: obv}\n"
           "buy: {left: obv, op: gt, right: {ref: obv, ago: 1}}\nsell: {left: obv, op: lt, right: {ref: obv, ago: 1}}\n",
    "raw_volume": "name: 成交量\nbuy: {left: volume, op: gt, right: 3000000}\nsell: {left: volume, op: lt, right: 2000000}\n",
    # 定義了 OBV 但條件沒用到：暖機天數也看它，所以一樣不比對
    "ma_obv": "name: 站上月線（看 OBV）\nindicators:\n  ma20: {type: sma, period: 20}\n  obv: {type: obv}\n"
              "buy: {left: close, op: gt, right: ma20}\nsell: {left: close, op: lt, right: ma20}\n",
    # 沒用到成交量
    "ma": "name: 站上月線\nindicators:\n  ma20: {type: sma, period: 20}\n"
          "buy: {left: close, op: gt, right: ma20}\nsell: {left: close, op: lt, right: ma20}\n",
}
USES_VOLUME = {"vol", "obv", "raw_volume", "ma_obv"}


def _write(home, dividends=(), reductions=(), splits=()):
    """dividends：（第幾天, 權／息）；reductions、splits：第幾天。參考價都設成前一天收盤的 0.9。"""
    p = make_prices(n=N, seed=7)
    days = pd.bdate_range("2024-01-01", periods=N)
    d = days.strftime("%Y-%m-%d")
    cache = Cache(home)
    cache.write("TaiwanStockPrice", "1234", pd.DataFrame([
        {"date": d[i], "stock_id": "1234", "Trading_Volume": r.volume, "Trading_money": r.close * r.volume,
         "open": r.open, "max": r.high, "min": r.low, "close": r.close, "spread": 0.0, "Trading_turnover": 10}
        for i, r in enumerate(p.itertuples())]))
    before = lambda i: float(p["close"].iloc[i - 1])  # noqa: E731
    cache.write("TaiwanStockDividendResult", "1234", pd.DataFrame(
        [{"date": d[i], "stock_id": "1234", "before_price": before(i), "after_price": before(i) * 0.9,
          **({"stock_or_cache_dividend": flag} if flag is not None else {})} for i, flag in dividends]))
    cache.write("TaiwanStockCapitalReductionReferencePrice", "1234", pd.DataFrame(
        [{"date": d[i], "stock_id": "1234", "ClosingPriceonTheLastTradingDay": before(i),
          "PostReductionReferencePrice": before(i) * 0.9} for i in reductions]))
    cache.write("TaiwanStockSplitPrice", "1234", pd.DataFrame(
        [{"date": d[i], "stock_id": "1234", "before_price": before(i), "after_price": before(i) * 0.9}
         for i in splits]))
    cache.write("TaiwanStockInfo", "1234", pd.DataFrame([{"stock_id": "1234", "stock_name": "測試", "type": "twse",
                                                        "industry_category": ""}]))
    return DataLoader(cache), days


@pytest.fixture
def rights(tmp_path):
    """第 150 天除權（配股或現金增資）。"""
    loader, days = _write(tmp_path / "home", dividends=[(EVENT, "權息")])
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    for name, text in STRATEGIES.items():
        (strategies / f"{name}.yaml").write_text(text, encoding="utf-8")
    return tmp_path / "home", strategies, loader, days


# ---------------------------------------------------------------------------
# 讀取：哪些事件會讓還原成交量變成空值


@pytest.mark.parametrize("events, expected", [
    ({"dividends": [(EVENT, "息")]}, []),
    ({"dividends": [(EVENT, "權")]}, ["除權"]),
    ({"dividends": [(EVENT, "權息")]}, ["除權"]),
    ({"dividends": [(EVENT, None)]}, ["除權"]),  # 沒有權／息欄位：分不出來，當成可能改變
    ({"dividends": [(EVENT, "看不懂")]}, ["除權"]),
    ({"reductions": [EVENT]}, ["減資"]),
    ({"splits": [EVENT]}, ["分割／面額變更"]),
])
def test_which_events_may_change_shares(tmp_path, events, expected):
    loader, days = _write(tmp_path, **events)
    changes = loader.share_changes("1234")
    assert changes["source"].tolist() == expected
    raw, adj = loader.prices("1234", adjust=False), loader.prices("1234", adjust=True)
    # 價格照舊還原：事件以前乘上 0.9
    np.testing.assert_allclose(adj["close"].iloc[:EVENT], raw["close"].iloc[:EVENT] * 0.9, rtol=1e-12)
    assert adj["close"].iloc[EVENT:].equals(raw["close"].iloc[EVENT:])
    # 成交量：股數可能改變、又沒有比例時，事件以前是空值；事件當天起和實際成交量相同
    if expected:
        assert adj["volume"].iloc[:EVENT].isna().all()
    else:
        assert adj["volume"].iloc[:EVENT].equals(raw["volume"].iloc[:EVENT])
    assert adj["volume"].iloc[EVENT:].equals(raw["volume"].iloc[EVENT:])
    # 成交金額、筆數不調整；實際成交價不受影響
    assert adj["value"].equals(raw["value"]) and adj["trades"].equals(raw["trades"])
    assert raw["volume"].notna().all()


def test_volume_is_missing_before_the_last_event(tmp_path):
    loader, days = _write(tmp_path, dividends=[(50, "權"), (200, "息")], reductions=[120])
    assert loader.share_changes("1234")["date"].tolist() == [days[50], days[120]]
    adj = loader.prices("1234", adjust=True)
    assert adj["volume"].iloc[:120].isna().all() and adj["volume"].iloc[120:].notna().all()


# ---------------------------------------------------------------------------
# 指標：照定義，窗口裡有一天成交量是空值就沒有值


@pytest.mark.parametrize("name", VOLUME_INDICATORS)
def test_volume_indicators_with_missing_volume(name):
    clean = make_prices(n=N, seed=11)
    missing = clean.copy()
    missing.loc[missing.index[:EVENT], "volume"] = np.nan
    spec, params = CATALOG[name], VOLUME_INDICATORS[name]
    full, partial = spec.compute(clean, **params), spec.compute(missing, **params)
    if name == "obv":
        assert partial.iloc[1:].isna().all()  # 累加值：前面有空值，之後都算不出來；第一天沒有漲跌，加 0
        return
    n = params["period"]
    first = EVENT + n - 1  # 窗口剛好全部在事件以後的第一天
    assert partial.iloc[:first].isna().all()
    # 窗口裡沒有空值的日子，和用完整資料算的完全相同
    pd.testing.assert_series_equal(partial.iloc[first:], full.iloc[first:])


def test_obv_after_missing_volume_in_the_middle():
    p = make_prices(n=60, seed=2)
    p.loc[p.index[30], "volume"] = np.nan
    out = CATALOG["obv"].compute(p)
    sign = np.sign(p["close"].diff()).fillna(0.0)
    expected = [sum(sign.iloc[k] * p["volume"].iloc[k] for k in range(1, t + 1)) for t in range(30)]
    np.testing.assert_allclose(out.iloc[:30], expected, rtol=1e-12)
    assert out.iloc[30:].isna().all()


def _flat_day_case(days: int = 12) -> pd.DataFrame:
    """第 5 天收盤（典型價格）和前一天相同，那天的成交量是空值。"""
    idx = pd.bdate_range("2024-01-01", periods=days)
    tp = [10, 11, 12, 11, 12, 12, 13, 12, 13, 14, 13, 14][:days]
    vol = [1000.0, 2000, 1500, 3000, 2500, np.nan, 1200, 1800, 2200, 1700, 2600, 1900][:days]
    return pd.DataFrame({"high": tp, "low": tp, "close": tp, "volume": vol}, index=idx, dtype=float)


def test_days_that_do_not_use_volume_are_not_affected():
    """照定義：OBV 收盤沒漲跌那天加 0，MFI 典型價格沒變那天不算資金流；這天的成交量用不到。"""
    p = _flat_day_case()
    filled = p.fillna({"volume": 123456.0})  # 換成任何數字，結果都要相同
    for name, params in (("obv", {}), ("mfi", {"period": 3})):
        out, ref = CATALOG[name].compute(p, **params), CATALOG[name].compute(filled, **params)
        pd.testing.assert_series_equal(out, ref)
        assert out.iloc[-1] == out.iloc[-1]  # 有值


# ---------------------------------------------------------------------------
# 訊號：實際成交價不受影響；還原模式下算不出來的那段沒有訊號，要提醒


def test_signal_warnings_only_in_adjusted_mode(rights):
    _, strategies, loader, days = rights
    for name in STRATEGIES:
        raw = signal_history("1234", name, loader=loader, directory=strategies)
        assert raw.warnings == [] and raw.result.values.notna().iloc[-1].all()
        adj = signal_history("1234", name, adjust=True, loader=loader, directory=strategies)
        if name in USES_VOLUME:
            [warning] = adj.warnings
            assert f"{days[EVENT]:%Y-%m-%d} 除權" in warning and "這天以前的指標算不出來" in warning
            if name != "ma_obv":  # ma_obv 的條件只用月線，OBV 只影響回測從哪天開始
                assert not adj.result.buy.iloc[:EVENT].any() and not adj.result.sell.iloc[:EVENT].any()
        else:
            assert adj.warnings == []


def test_screen_warnings_only_in_adjusted_mode(rights):
    _, strategies, loader, _ = rights
    assert screen(["1234"], loader=loader, directory=strategies).warnings == []
    report = screen(["1234"], loader=loader, directory=strategies, adjusted=True)
    assert {who for who, _ in report.warnings} == {f"股票 1234（策略 {n}）" for n in USES_VOLUME}
    assert len(report.rows) == len(STRATEGIES)


DARVAS = ("name: 箱型\nindicators:\n  box: {type: darvas, new_high_days: 10, confirm_days: 3}\n"
          "buy: {left: high, op: gt, right: {ref: box.top, ago: 1}}\nsell: {left: low, op: lt, right: {ref: box.stop, ago: 1}}\n")


def test_pattern_without_value_is_not_missing_data(rights, tmp_path):
    """Darvas 箱沒有值是「當時沒有箱子」：還原價格沒有箱子的日子，不當成算不出來。"""
    _, strategies, loader, _ = rights
    (strategies / "darvas.yaml").write_text(DARVAS, encoding="utf-8")
    raw = signal_history("1234", "darvas", loader=loader, directory=strategies).result.values["box.top"]
    hist = signal_history("1234", "darvas", adjust=True, loader=loader, directory=strategies)
    adj = hist.result.values["box.top"]
    assert (raw.notna() & adj.isna()).any()  # 確實有實際成交價有箱子、還原價格沒有的日子
    assert hist.warnings == []
    # 回測從暖機結束開始：第 10 天起前一天的箱子才算得出來，不是等到第一個箱子
    rb = rule_backtest(hist, costs=CostModel.zero(), loader=loader)
    assert rb.start == adj.index[10] < adj.first_valid_index()

    sets = tmp_path / "indicators"
    sets.mkdir()
    (sets / "box.yaml").write_text("name: 箱型\nindicators:\n  box: {type: darvas, new_high_days: 10, confirm_days: 3}\n",
                                   encoding="utf-8")
    day = (raw.notna() & adj.isna()).idxmax()
    snap = indicator_snapshot("1234", "box", day, loader, sets, adjusted=True)
    assert snap.values["box.top"] != snap.values["box.top"]  # 還原價格那天沒有箱子
    assert not snap.unavailable.any() and snap.warnings == []


def test_backtest_start_depends_on_mode(rights):
    """實際成交價的成交量都在，從實際成交價的暖機結束開始；還原模式要等還原價格的成交量算得出來。"""
    _, strategies, loader, days = rights
    for name in ("obv", "ma_obv"):
        raw = signal_history("1234", name, loader=loader, directory=strategies)
        assert rule_backtest(raw, costs=CostModel.zero(), loader=loader).start == raw.result.warmup_end < days[EVENT]
    adj = signal_history("1234", "vol", adjust=True, loader=loader, directory=strategies)
    rb = rule_backtest(adj, costs=CostModel.zero(), loader=loader)
    assert rb.start == adj.result.warmup_end > days[EVENT]
    # OBV 是累加值，除權以前算不出來，之後也一直沒有值，沒辦法回測
    obv = signal_history("1234", "obv", adjust=True, loader=loader, directory=strategies)
    with pytest.raises(StrategyConfigError, match="資料天數不夠"):
        rule_backtest(obv, costs=CostModel.zero(), loader=loader)


def test_cli_messages(rights, monkeypatch):
    home, strategies, _, days = rights
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STRATEGIES", str(strategies))
    env = {"COLUMNS": "250"}
    out = CliRunner().invoke(app, ["backtest", "1234", "-s", "vol", "--no-plot"], env=env)
    assert out.exit_code == 0, out.output
    assert "⚠" not in out.output and "還原" not in out.output
    out = CliRunner().invoke(app, ["backtest", "1234", "-s", "vol", "--adjust", "--no-plot"], env=env)
    assert out.exit_code == 0, out.output
    assert f"{days[EVENT]:%Y-%m-%d} 除權" in out.output and "這天以前的指標算不出來" in out.output
    out = CliRunner().invoke(app, ["screen", "1234", "--adjust"], env=env)
    assert out.exit_code == 0, out.output
    assert out.output.count("這天以前的指標算不出來") == len(USES_VOLUME)


# ---------------------------------------------------------------------------
# tech、show、verify、機器學習


def test_tech_marks_unavailable_values(rights, tmp_path, monkeypatch):
    home, _, loader, days = rights
    sets = tmp_path / "indicators"
    sets.mkdir()
    (sets / "v.yaml").write_text("name: 量\nindicators:\n  obv: {type: obv}\n  vr: {type: volume_ratio, period: 20}\n"
                                 "  ma20: {type: sma, period: 20}\n", encoding="utf-8")
    raw = indicator_snapshot("1234", "v", None, loader, sets)
    assert not raw.unavailable.any() and raw.warnings == [] and raw.values.notna().all()
    snap = indicator_snapshot("1234", "v", None, loader, sets, adjusted=True)
    assert snap.unavailable.to_dict() == {"obv": True, "vr": False, "ma20": False}
    [warning] = snap.warnings
    assert f"{days[EVENT]:%Y-%m-%d} 除權" in warning
    # 除權以前的日子，量比也算不出來
    assert indicator_snapshot("1234", "v", days[EVENT - 1], loader, sets, adjusted=True).unavailable["vr"]

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INDICATORS", str(sets))
    out = CliRunner().invoke(app, ["tech", "1234", "-i", "v", "--adjust"], env={"COLUMNS": "200"})
    assert out.exit_code == 0, out.output
    assert "無法計算" in out.output and "除權" in out.output
    out = CliRunner().invoke(app, ["tech", "1234", "-i", "v"], env={"COLUMNS": "200"})
    assert "無法計算" not in out.output and "除權" not in out.output


def test_tech_without_share_changes_has_no_warning(tmp_path):
    loader, _ = _write(tmp_path, dividends=[(EVENT, "息")])
    sets = tmp_path / "indicators"
    sets.mkdir()
    (sets / "v.yaml").write_text("name: 量\nindicators:\n  obv: {type: obv}\n", encoding="utf-8")
    snap = indicator_snapshot("1234", "v", None, loader, sets, adjusted=True)
    assert not snap.unavailable.any() and snap.warnings == []


def test_show_adjusted_volume(rights, monkeypatch):
    home, _, _, _ = rights
    monkeypatch.setenv("HERMES_HOME", str(home))
    out = CliRunner().invoke(app, ["show", "1234", "--adjust", "--last", f"{N - EVENT + 2}"], env={"COLUMNS": "200"})
    assert out.exit_code == 0, out.output
    assert "無法計算" in out.output and "除權" in out.output
    out = CliRunner().invoke(app, ["show", "1234", "--adjust", "--last", "5"], env={"COLUMNS": "200"})
    assert "無法計算" not in out.output


def test_verify_lists_share_changes(tmp_path):
    loader, days = _write(tmp_path, dividends=[(50, "權"), (100, "息")], splits=[200])
    check = check_share_changes("1234", loader)
    assert check.status == "warn"
    assert check.problems == [f"{days[50]:%Y-%m-%d} 除權：沒有證交所的股數比例",
                              f"{days[200]:%Y-%m-%d} 分割／面額變更：沒有證交所的股數比例"]
    loader, _ = _write(tmp_path / "only_cash", dividends=[(100, "息")])
    assert check_share_changes("1234", loader).status == "ok"


def test_ml_refuses_missing_volume(rights):
    _, _, loader, days = rights
    adjusted = loader.prices("1234", adjust=True)
    for build in (lambda p: price_features(p), lambda p: exogenous_features(loader, "1234", p)):
        with pytest.raises(ValueError, match=f"{days[EVENT - 1]:%Y-%m-%d} 為止算不出來"):
            build(adjusted)
    price_features(loader.prices("1234", adjust=False))  # 實際成交價照常
