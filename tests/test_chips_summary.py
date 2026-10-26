"""籌碼面摘要（hermes chips）的獨立對照：照定義逐日計算。

- N 日合計、和 N 日前相比，都以交易日計算；區間內有一天沒有紀錄就是空值，不補值。
- 還沒公布的日子（資料的最後一天之後）不算缺資料。
"""

import math

import numpy as np
import pandas as pd
import pytest

from hermes.chips import chip_summary

DAYS = pd.bdate_range("2026-01-05", periods=40)
MISSING = DAYS[25]  # 三大法人這天沒有紀錄


def _data():
    rng = np.random.default_rng(5)
    inst = pd.DataFrame({k: rng.integers(-3_000_000, 3_000_000, len(DAYS)).astype(float)
                         for k in ("foreign", "trust", "dealer")}, index=DAYS)
    inst.iloc[-4:, 0] = [2000.0, 1000.0, 3000.0, 5000.0]  # 外資最後連買 4 天
    inst.iloc[-5, 0] = -1000.0
    inst["total"] = inst.sum(axis=1)
    inst = inst.drop(MISSING)
    holding = pd.DataFrame({"foreign_ratio": 70 + rng.normal(0, 0.1, len(DAYS)).cumsum(),
                            "shares_issued": 1e9}, index=DAYS)
    margin_days = DAYS[:-1]  # 融資融券最後一天還沒公布
    margin = pd.DataFrame({"margin_balance": 10_000 + rng.integers(-300, 300, len(margin_days)).cumsum(),
                           "short_balance": 500 + rng.integers(-20, 20, len(margin_days)).cumsum(),
                           "margin_limit": 50_000.0}, index=margin_days).astype(float)
    margin["short_margin_ratio"] = margin["short_balance"] / margin["margin_balance"]
    return inst, holding, margin


def _ref_sum(series: dict, days, n):
    w = list(days[-n:])
    if len(w) < n or any(d not in series for d in w):
        return math.nan
    return sum(series[d] for d in w)


def _ref_change(series: dict, days, n):
    if len(days) <= n or days[-1] not in series or days[-1 - n] not in series:
        return math.nan
    return series[days[-1]] - series[days[-1 - n]]


def _ref_streak(series: dict, days):
    sign = lambda v: (v > 0) - (v < 0)  # noqa: E731
    if days[-1] not in series or series[days[-1]] == 0:
        return 0
    s, n = sign(series[days[-1]]), 0
    for d in reversed(days):
        if d not in series or sign(series[d]) != s:
            break
        n += 1
    return n * s


def test_summary_matches_definition():
    inst, holding, margin = _data()
    rep = chip_summary(DAYS, inst, holding, margin, windows=(5, 20, 3))
    days = list(DAYS)
    lots = {k: {d: v / 1000 for d, v in inst[k].items()} for k in ("foreign", "trust", "dealer", "total")}
    names = {"foreign": "外資", "trust": "投信", "dealer": "自營商", "total": "三大法人合計"}
    assert list(rep.flows.columns) == ["5 日", "20 日", "3 日"]
    for key, who in names.items():
        for n in (5, 20, 3):
            ref = _ref_sum(lots[key], days, n)
            ours = rep.flows.loc[who, f"{n} 日"]
            assert (math.isnan(ref) and math.isnan(ours)) or ours == pytest.approx(ref), (who, n)
    # 20 日的區間包含沒有紀錄的那天，所以是空值；5 日不包含
    assert math.isnan(rep.flows.loc["外資", "20 日"]) and not math.isnan(rep.flows.loc["外資", "5 日"])
    for key in ("foreign", "trust", "dealer"):
        assert rep.streaks[names[key]] == _ref_streak(lots[key], days)
    assert rep.streaks["外資"] == 4

    fr = holding["foreign_ratio"].to_dict()
    assert rep.foreign_ratio["now"] == fr[days[-1]]
    for n in (5, 20, 3):
        assert rep.foreign_ratio[f"chg_{n}"] == pytest.approx(_ref_change(fr, days, n))

    mdays = [d for d in days if d <= margin.index.max()]
    mb = margin["margin_balance"].to_dict()
    assert rep.margin["date"] == mdays[-1] != rep.as_of
    assert rep.margin["margin_balance"] == mb[mdays[-1]]
    assert rep.margin["margin_usage"] == pytest.approx(mb[mdays[-1]] / 50_000)
    assert rep.margin["short_margin_ratio"] == pytest.approx(margin["short_balance"].iloc[-1] / mb[mdays[-1]])
    for n in (5, 20, 3):
        assert rep.margin[f"margin_chg_{n}"] == pytest.approx(_ref_change(mb, mdays, n))


def test_no_windows_means_no_aggregates():
    inst, holding, margin = _data()
    rep = chip_summary(DAYS, inst, holding, margin)
    assert rep.flows.empty
    assert set(rep.foreign_ratio) == {"date", "now"}
    assert not any(k.startswith("margin_chg_") for k in rep.margin)
    assert rep.streaks["外資"] == 4 and len(rep.recent) == 10


def test_streak_stops_at_missing_day():
    inst, holding, margin = _data()
    rep = chip_summary(DAYS.append(pd.DatetimeIndex([DAYS[-1] + pd.offsets.BDay()])), inst.drop(DAYS[-2]),
                       holding, margin)
    assert rep.streaks["外資"] == 1  # 倒數第二天沒有紀錄，連續天數斷在那裡
