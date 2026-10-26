import numpy as np
import pandas as pd
import pytest
from conftest import CATALOG_PARAMS

from hermes.services import operand_labels
from hermes.strategies import StrategyConfigError, describe, load_strategy, parse_strategy, run_rules
from hermes.technical import CATALOG
from hermes.technical.indicators import sma

CCI_STRATEGY_YAML = """
name: CCI 超賣反轉
indicators:
  cci:  {type: cci, period: 88}
  ma60: {type: sma, period: 60}
buy:
  all:
    - {left: cci,   op: crosses_above, right: -100}
    - {left: close, op: gt,            right: ma60}
sell:
  any:
    - {left: cci,   op: crosses_below, right: 100}
    - {left: close, op: lt, right: ma60, days: 3}
"""


def _params(spec):
    return CATALOG_PARAMS[spec.type]


def test_catalog_outputs_match_declared_lines(prices):
    for spec in CATALOG.values():
        out = spec.compute(prices, **spec.resolve_params(_params(spec)))
        if spec.outputs:
            assert isinstance(out, pd.DataFrame), spec.type
            assert set(spec.outputs) <= set(out.columns), spec.type
        else:
            assert isinstance(out, pd.Series), spec.type
        assert out.index.equals(prices.index), spec.type


def test_catalog_has_no_lookahead(prices):
    """在第 t 天截斷資料重算，第 t 天的值必須和用全部資料算的一樣。"""
    for spec in CATALOG.values():
        params = spec.resolve_params(_params(spec))
        full = pd.DataFrame(spec.compute(prices, **params))
        for t in (300, 600):
            part = pd.DataFrame(spec.compute(prices.iloc[: t + 1], **params))
            a, b = full.iloc[t], part.iloc[-1]
            assert (a.isna() == b.isna()).all(), f"{spec.type} t={t}"
            both = a.notna()
            assert np.allclose(a[both].astype(float), b[both].astype(float)), f"{spec.type} t={t}"


def test_pattern_warmup_and_no_volume(prices):
    """型態類指標（指標目錄有給暖機天數）：暖機期間一定沒有值，暖機以後每條線都會出現值；而且不能用到成交量。

    和還原價格比對時，型態類的空值當成「當時沒有型態」，不當成成交量算不出來；用到成交量就會把後者漏掉。
    """
    patterns = [s for s in CATALOG.values() if s.warmup is not None]
    assert patterns
    for spec in patterns:
        params = spec.resolve_params(_params(spec))
        out = pd.DataFrame(spec.compute(prices, **params))
        days = spec.warmup(**params)
        assert out.iloc[:days].isna().all().all(), spec.type
        assert out.iloc[days:].notna().any().all(), spec.type
        no_volume = pd.DataFrame(spec.compute(prices.assign(volume=np.nan), **params))
        pd.testing.assert_frame_equal(no_volume, out, obj=spec.type)


def test_resolve_params():
    cci = CATALOG["cci"]
    with pytest.raises(ValueError, match="需要指定 period"):
        cci.resolve_params({})  # 沒有預設值
    assert cci.resolve_params({"period": 88}) == {"period": 88}
    with pytest.raises(ValueError, match="沒有參數 n"):
        cci.resolve_params({"n": 88})
    with pytest.raises(ValueError, match="要是整數"):
        cci.resolve_params({"period": 8.5})
    with pytest.raises(ValueError, match="要大於 0"):
        cci.resolve_params({"period": 0})
    with pytest.raises(ValueError, match="要是數字"):
        cci.resolve_params({"period": "abc"})
    with pytest.raises(ValueError, match="需要指定 period"):
        CATALOG["sma"].resolve_params({})


def test_load_strategy_yaml(tmp_path):
    path = tmp_path / "cci.yaml"
    path.write_text(CCI_STRATEGY_YAML, encoding="utf-8")
    s = load_strategy(path)
    assert s.name == "CCI 超賣反轉"
    assert s.indicators["cci"].params == {"period": 88}
    assert s.sell.any[1].days == 3


@pytest.mark.parametrize(
    "rule, message",
    [
        ({"left": "cci", "op": "bigger", "right": 1}, "buy.op：只能是"),
        ({"left": "cci", "op": "gt"}, "buy.right：缺少這個欄位"),
        ({"left": "cci", "op": "gt", "right": 1, "dayz": 2}, "buy.dayz：不認得這個欄位"),
        ({"all": [{"left": "cci", "op": "gt", "right": 1}, {"left": "x", "op": "lt", "right": 1}]}, "buy 用到 x"),
        ({"left": "cci", "op": "crosses_above", "right": 1, "days": 2}, "不能搭配 days"),
        ({"all": [], "any": []}, "剛好寫一個"),
        ({"left": "kd", "op": "gt", "right": 80}, "kd.k、kd.d"),
        ({"left": "kd.x", "op": "gt", "right": 80}, "kd.k、kd.d"),
        ({"left": "cci", "op": "gt", "right": {"ref": "x", "ago": 1}}, "buy 用到 x"),
        ({"left": {"ref": "kd", "ago": 1}, "op": "gt", "right": 80}, "kd.k、kd.d"),
        ({"left": "cci", "op": "gt", "right": {"ref": "cci", "ago": -1}}, r"buy.right.ago：要大於或等於 0"),
        ({"left": "cci", "op": "gt", "right": {"ref": "cci", "days_ago": 1}}, "buy.right.days_ago：不認得這個欄位"),
        ({"left": "cci", "op": "gt", "right": {"ago": 1}}, "buy.right.ref：缺少這個欄位"),
        ({"left": 100, "op": "lt", "right": "cci"}, "buy.left：要寫指標或價格欄位的名稱，不能是數字"),
        ({"left": "cci", "op": "gt", "right": [1]}, r"buy.right：要寫名稱、數字，或 \{ref: 名稱, ago: 天數\}"),
    ],
)
def test_config_errors_point_to_the_problem(rule, message):
    cfg = {"name": "t", "indicators": {"cci": {"type": "cci", "period": 20}, "kd": {"type": "kd", **CATALOG_PARAMS["kd"]}}, "buy": rule}
    with pytest.raises(StrategyConfigError, match=message):
        parse_strategy(cfg)


def test_plain_labels():
    """網頁顯示的白話名稱：天數寫成「N 日」，有 short 的照 short，標題結尾是括號時參數放進同一個括號。"""
    def label(type_, **params):
        spec = CATALOG[type_]
        return spec.label(spec.resolve_params(params))

    assert label("sma", period=200) == "簡單移動平均（200 日）"
    assert label("bias", period=20) == "乖離率（%，20 日）"
    assert label("kd", period=9, k_smooth=3, d_smooth=3) == "KD 隨機指標（RSV 9 日、K 平滑 3、D 平滑 3）"
    assert label("darvas", new_high_days=252, confirm_days=3) == "Darvas 箱型（創 252 日新高、確認 3 天）"
    assert label("bollinger", period=20, k=2.5) == "布林通道（20 日、2.5 倍標準差）"
    assert label("obv") == "能量潮 OBV"
    for spec in CATALOG.values():  # 每條線都有白話名稱
        assert len(spec.line_labels) == len(spec.outputs), spec.type
    s = parse_strategy({"name": "t", "indicators": {"box": {"type": "darvas", "new_high_days": 252, "confirm_days": 3}},
                        "buy": {"left": "high", "op": "gt", "right": {"ref": "box.top", "ago": 1}}})
    assert describe(s.buy) == "high > 前 1 天的 box.top"
    assert describe(s.buy, operand_labels(s)) == "最高價 > 前 1 天的 Darvas 箱型（創 252 日新高、確認 3 天）：箱頂"


def test_reference_lines():
    """參考線只畫線：鍵要是 indicators 裡的名稱，值要是數字；不影響買賣條件。"""
    base = {"name": "t", "indicators": {"cci": {"type": "cci", "period": 20}},
            "buy": {"left": "cci", "op": "crosses_above", "right": 0}}
    s = parse_strategy({**base, "reference_lines": {"cci": [100, -100]}})
    assert s.reference_lines == {"cci": [100.0, -100.0]}
    assert parse_strategy(base).reference_lines == {}
    with pytest.raises(StrategyConfigError, match="reference_lines 用到 x"):
        parse_strategy({**base, "reference_lines": {"x": [1]}})
    with pytest.raises(StrategyConfigError, match=r"reference_lines\.cci\[0\]"):
        parse_strategy({**base, "reference_lines": {"cci": ["abc"]}})


def test_unknown_indicator_type_and_bad_param():
    with pytest.raises(StrategyConfigError, match="類型 foo 不存在"):
        parse_strategy({"name": "t", "indicators": {"x": {"type": "foo"}}, "buy": {"left": "x", "op": "gt", "right": 0}})
    with pytest.raises(StrategyConfigError, match="指標 c：cci 沒有參數 n"):
        parse_strategy({"name": "t", "indicators": {"c": {"type": "cci", "n": 88}},
                        "buy": {"left": "c", "op": "gt", "right": 0}})
    with pytest.raises(StrategyConfigError, match="指標 c：cci 的 period 要是數字"):
        parse_strategy({"name": "t", "indicators": {"c": {"type": "cci", "period": "abc"}},
                        "buy": {"left": "c", "op": "gt", "right": 0}})


def _flat_prices(closes):
    idx = pd.bdate_range("2026-01-05", periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1000.0})


def _signals(rule, closes, indicators=None):
    s = parse_strategy({"name": "t", "indicators": indicators or {}, "buy": rule})
    return run_rules(s, _flat_prices(closes)).buy.tolist()


def test_state_event_and_days():
    closes = [10, 9, 8, 11, 12, 9, 13]
    assert _signals({"left": "close", "op": "gt", "right": 10}, closes) == [0, 0, 0, 1, 1, 0, 1]
    assert _signals({"left": "close", "op": "crosses_above", "right": 10}, closes) == [0, 0, 0, 1, 0, 0, 1]
    assert _signals({"left": "close", "op": "crosses_below", "right": 10}, closes) == [0, 1, 0, 0, 0, 1, 0]
    assert _signals({"left": "close", "op": "gt", "right": 10, "days": 2}, closes) == [0, 0, 0, 0, 1, 0, 0]


def test_line_vs_line_and_warmup_has_no_signal():
    closes = [10, 10, 10, 13, 9, 8, 12]
    ma = {"ma2": {"type": "sma", "period": 2}}
    # 第一天均線還沒有值（暖機期），不會產生訊號
    assert _signals({"left": "close", "op": "crosses_above", "right": "ma2"}, closes, ma) == [0, 0, 0, 1, 0, 0, 1]


def test_equal_values_with_rounding_error():
    """收盤價數學上剛好等於 5 日平均時，算平均的捨入誤差不能讓「大於」「小於」成立。

    第一組最後一天：(56.24 + 95.72 + 62.0 + 51.32 + 66.32) / 5 = 66.32，但浮點數算出 66.32000000000001；
    第二組：(74.33 + 70.1 + 32.73 + 15.8 + 48.24) / 5 = 48.24，浮點數算出 48.239999999999995。
    """
    ma = {"ma5": {"type": "sma", "period": 5}}
    for closes in ([56.24, 95.72, 62.0, 51.32, 66.32], [74.33, 70.1, 32.73, 15.8, 48.24]):
        assert sma(pd.Series(closes), 5).iloc[-1] != closes[-1]  # 確實有捨入誤差
        # 價格整段乘上同一個比例（像除權息的還原），判斷也要一樣
        for scale in (1.0, 0.97, 0.4321):
            c = [x * scale for x in closes]
            got = {op: _signals({"left": "close", "op": op, "right": "ma5"}, c, ma)[-1] for op in ("gt", "gte", "lt", "lte")}
            assert got == {"gt": False, "gte": True, "lt": False, "lte": True}, (closes, scale)
    # 穿越：前一天剛好相等（浮點數算出的平均比收盤小一點點）、當天在上面，算「由下往上穿越」
    closes = [74.33, 70.1, 32.73, 15.8, 48.24, 60.0]
    assert _signals({"left": "close", "op": "crosses_above", "right": "ma5"}, closes, ma)[-1]
    # 差距大於捨入誤差就照常比：66.33 > 平均
    assert _signals({"left": "close", "op": "gt", "right": "ma5"}, [56.24, 95.72, 62.0, 51.32, 66.33], ma)[-1]


def test_nested_all_any():
    closes = [5, 15, 25, 15, 5]
    rule = {"any": [{"left": "close", "op": "lt", "right": 10},
                    {"all": [{"left": "close", "op": "gt", "right": 12}, {"left": "close", "op": "lt", "right": 20}]}]}
    assert _signals(rule, closes) == [1, 1, 0, 1, 1]


def test_multi_line_indicator_columns(prices):
    s = parse_strategy({"name": "t", "indicators": {"kd": {"type": "kd", **CATALOG_PARAMS["kd"]}},
                        "buy": {"left": "kd.k", "op": "crosses_above", "right": "kd.d"}})
    res = run_rules(s, prices)
    assert {"kd.k", "kd.d"} <= set(res.values.columns)
    assert res.buy.any() and not res.sell.any()


def test_compare_with_previous_days():
    """唐奇安通道包含當天，所以「收盤 > 當天的上緣」永遠不成立；要比前一天的上緣。

    收盤 10 11 12 11 13 12 14，3 日上緣（含當天）：— — 12 12 13 13 14，前一天的上緣：— — — 12 12 13 13。
    """
    closes = [10, 11, 12, 11, 13, 12, 14]
    dc = {"dc": {"type": "donchian", "period": 3}}
    assert _signals({"left": "close", "op": "gt", "right": "dc.upper"}, closes, dc) == [0] * 7
    rule = {"left": "close", "op": "gt", "right": {"ref": "dc.upper", "ago": 1}}
    assert _signals(rule, closes, dc) == [0, 0, 0, 0, 1, 0, 1]
    assert _signals({**rule, "op": "crosses_above"}, closes, dc) == [0, 0, 0, 0, 1, 0, 1]
    # ago: 0 等於直接寫名稱；價格欄位也可以比前 N 天：收盤比 2 天前高
    assert _signals({"left": "close", "op": "gt", "right": {"ref": "dc.upper", "ago": 0}}, closes, dc) == [0] * 7
    assert _signals({"left": "close", "op": "gt", "right": {"ref": "close", "ago": 2}}, closes) == [0, 0, 1, 0, 1, 1, 1]


def test_previous_day_values_are_listed_and_delay_warmup():
    s = parse_strategy({"name": "t", "indicators": {"dc": {"type": "donchian", "period": 3}},
                        "buy": {"left": {"ref": "close", "ago": 1}, "op": "lt", "right": {"ref": "dc.lower", "ago": 2}}})
    assert describe(s.buy) == "前 1 天的 close < 前 2 天的 dc.lower"
    res = run_rules(s, _flat_prices([10, 11, 12, 11, 13, 12, 14]))
    assert res.values["close（前1天）"].tolist()[1:] == [10, 11, 12, 11, 13, 12]
    assert res.values["dc.lower（前2天）"].tolist()[4:] == [10, 11, 11]
    assert res.warmup_end == res.values.index[4]  # 3 日通道第 3 天才有值，再往後 2 天


def test_forming_condition():
    """形成中的條件：寫法和 buy 一樣，名稱也會檢查；只算每天成立與否，不影響買賣，也不延後暖機。"""
    base = {"name": "t", "indicators": {"dc": {"type": "donchian", "period": 3}},
            "buy": {"left": "close", "op": "gt", "right": {"ref": "dc.upper", "ago": 1}},
            "sell": {"left": "close", "op": "lt", "right": {"ref": "dc.lower", "ago": 1}}}
    with pytest.raises(StrategyConfigError, match="forming 用到 x"):
        parse_strategy({**base, "forming": {"left": "x", "op": "gt", "right": 0}})
    with pytest.raises(StrategyConfigError, match=r"forming\.op：只能是"):
        parse_strategy({**base, "forming": {"left": "close", "op": "bigger", "right": 0}})

    closes = [10, 11, 12, 11, 13, 12, 14, 13, 12, 11]
    rule = {"left": "close", "op": "lt", "right": {"ref": "close", "ago": 4}}  # 收盤比 4 天前低
    s = parse_strategy({**base, "forming": rule})
    assert describe(s.forming) == "close < 前 4 天的 close"
    res = run_rules(s, _flat_prices(closes))
    plain = run_rules(parse_strategy(base), _flat_prices(closes))
    # 照定義逐日判斷：第 t 天的收盤 < 第 t − 4 天的收盤
    assert res.forming.tolist() == [t >= 4 and closes[t] < closes[t - 4] for t in range(len(closes))]
    assert res.forming_streak.tolist() == [0] * 8 + [1, 2]
    assert res.values["close（前4天）"].tolist()[4:] == closes[:6]
    # 買賣和沒寫形成中時完全相同；前 4 天的收盤只有形成中用到，暖機不等它（通道第 3 天有值、再晚 1 天）
    assert res.buy.equals(plain.buy) and res.sell.equals(plain.sell)
    assert res.forming_only == ("close（前4天）",)
    assert res.warmup_end == plain.warmup_end == res.values.index[3]
    assert not plain.forming.any() and plain.forming_only == ()
    # 買賣條件也用到的值，暖機照樣要等
    both = parse_strategy({**base, "forming": {"left": "close", "op": "lt", "right": {"ref": "dc.upper", "ago": 1}}})
    assert run_rules(both, _flat_prices(closes)).forming_only == ()
