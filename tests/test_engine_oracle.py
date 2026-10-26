"""策略引擎的獨立對照。

- 隨機產生策略（指標、比較方式、連續天數、而且／或者巢狀），和照定義逐日判斷的簡單寫法比較。
- 只用到第 t 天以前的資料重算，第 t 天的訊號必須和用全部資料算的一樣（不偷看未來）。
"""

import math
import random

import numpy as np
import pandas as pd
import pytest

from conftest import make_prices
from hermes.strategies import parse_strategy, run_rules, walk
from hermes.strategies.engine import indicator_values
from hermes.strategies.rules import EVENT_OPS, STATE_OPS, Ago, Condition

INDICATORS = {
    "ma5": {"type": "sma", "period": 5},
    "ma20": {"type": "sma", "period": 20},
    "e10": {"type": "ema", "period": 10},
    "cci": {"type": "cci", "period": 14},
    "rsi": {"type": "rsi", "period": 6},
    "kd": {"type": "kd", "period": 9, "k_smooth": 3, "d_smooth": 3},
    "bb": {"type": "bollinger", "period": 10, "k": 2.0},
}
# 同一組裡的數值單位相同，拿來互相比較才有意義
GROUPS = [
    ["open", "high", "low", "close", "ma5", "ma20", "e10", "bb.upper", "bb.mid", "bb.lower"],
    ["kd.k", "kd.d", "rsi"],
    ["cci"],
    ["volume"],
    ["bb.pct_b"],
    # 籌碼、本益比：有傳入這些欄位時才會用到
    ["foreign_net", "trust_net", "inst_net"],
    ["foreign_ratio"],
    ["per"],
]


def _random_rule(rng: random.Random, values: pd.DataFrame, depth: int = 0) -> dict:
    if depth < 2 and rng.random() < 0.4:
        key = rng.choice(["all", "any"])
        return {key: [_random_rule(rng, values, depth + 1) for _ in range(rng.randint(2, 3))]}
    group = rng.choice([g for g in GROUPS if all(c in values for c in g)])
    left = rng.choice(group)
    op = rng.choice(STATE_OPS + EVENT_OPS)
    if len(group) > 1 and rng.random() < 0.5:
        right = rng.choice([g for g in group if g != left])
    else:
        right = float(np.nanquantile(values[left], rng.uniform(0.2, 0.8)))
    # 有時候改成比前 N 天的值
    if rng.random() < 0.25:
        left = {"ref": left, "ago": rng.randint(1, 3)}
    if isinstance(right, str) and rng.random() < 0.3:
        right = {"ref": right, "ago": rng.randint(1, 3)}
    cond = {"left": left, "op": op, "right": right}
    if op in STATE_OPS and rng.random() < 0.4:
        cond["days"] = rng.randint(2, 4)
    return cond


def _random_strategy(seed: int, prices: pd.DataFrame, fields: pd.DataFrame | None = None):
    """fields：籌碼、本益比欄位；有傳入時，隨機條件也會用到它們。"""
    rng = random.Random(seed)
    base = parse_strategy({"name": "t", "indicators": INDICATORS, "buy": {"left": "close", "op": "gt", "right": 0}})
    values = indicator_values(base, prices)
    if fields is not None:
        values = values.join(fields)
    return parse_strategy({"name": f"random-{seed}", "indicators": INDICATORS,
                           "buy": _random_rule(rng, values), "sell": _random_rule(rng, values)})


def _naive(rule, v: dict[str, list[float]], t: int) -> bool:
    """照定義判斷第 t 天：沒有值就不成立；穿越看前一天和當天；連續 N 天要每一天都成立；
    前 N 天的值就是第 k − N 天的值。兩邊用 math.isclose（預設相對誤差 1e-9）判斷是否相等。"""

    def cmp(op, a, b):
        eq = math.isclose(a, b)
        return {"gt": a > b and not eq, "gte": a > b or eq, "lt": a < b and not eq, "lte": a < b or eq}[op]

    if not isinstance(rule, Condition):
        results = [_naive(r, v, t) for r in rule.rules]
        return all(results) if rule.all is not None else any(results)

    def val(ref, k):
        if isinstance(ref, Ago):
            return v[ref.ref][k - ref.ago] if k - ref.ago >= 0 else math.nan
        return v[ref][k] if isinstance(ref, str) else ref

    def state(k):
        if k < 0:
            return False
        a, b = val(rule.left, k), val(rule.right, k)
        if math.isnan(a) or math.isnan(b):
            return False
        return cmp(rule.op, a, b)

    if rule.op in EVENT_OPS:
        if t == 0:
            return False
        a, b, pa, pb = val(rule.left, t), val(rule.right, t), val(rule.left, t - 1), val(rule.right, t - 1)
        if any(math.isnan(x) for x in (a, b, pa, pb)):
            return False
        if rule.op == "crosses_above":
            return cmp("gt", a, b) and cmp("lte", pa, pb)
        return cmp("lt", a, b) and cmp("gte", pa, pb)
    return all(state(k) for k in range(t - rule.days + 1, t + 1))


PRICES = make_prices(n=300, seed=3)


def _fields(index: pd.DatetimeIndex, seed: int = 7) -> pd.DataFrame:
    """隨機的籌碼、本益比欄位，其中幾天沒有資料（空值）。"""
    rng = np.random.default_rng(seed)
    n = len(index)
    foreign, trust = rng.normal(0, 5000, n).round(), rng.normal(0, 800, n).round()
    out = pd.DataFrame({"foreign_net": foreign, "trust_net": trust, "inst_net": foreign + trust + rng.normal(0, 300, n),
                        "foreign_ratio": 70 + rng.normal(0, 0.3, n).cumsum(), "per": 20 + rng.normal(0, 0.5, n).cumsum()},
                       index=index)
    for col in out.columns:
        out.loc[out.index[rng.choice(n, 6, replace=False)], col] = np.nan
    return out


FIELDS = _fields(PRICES.index)


@pytest.mark.parametrize("with_fields", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_engine_matches_naive_evaluation(seed, with_fields):
    fields = FIELDS if with_fields else None
    s = _random_strategy(seed, PRICES, fields)
    res = run_rules(s, PRICES, fields)
    v = {c: res.values[c].tolist() for c in res.values.columns}
    assert res.buy.tolist() == [_naive(s.buy, v, t) for t in range(len(PRICES))]
    assert res.sell.tolist() == [_naive(s.sell, v, t) for t in range(len(PRICES))]


def test_random_strategies_are_not_trivial():
    """確認隨機策略有一定比例會在部分日子成立，而且有用到前 N 天的值、籌碼和本益比，對照才有意義。"""
    mixed = lagged = with_fields = 0
    for seed in range(40):
        s = _random_strategy(seed, PRICES, FIELDS)
        res = run_rules(s, PRICES, FIELDS)
        mixed += int(0 < res.buy.sum() < len(PRICES)) + int(0 < res.sell.sum() < len(PRICES))
        lagged += sum(isinstance(x, Ago) for c in [*walk(s.buy), *walk(s.sell)] for x in (c.left, c.right))
        with_fields += bool(s.same_day_fields)
    assert mixed >= 40
    assert lagged >= 20
    assert with_fields >= 15


@pytest.mark.parametrize("seed", range(10))
def test_signal_uses_only_past_data(seed):
    s = _random_strategy(seed, PRICES, FIELDS)
    full = run_rules(s, PRICES, FIELDS)
    for t in (60, 150, 299):
        part = run_rules(s, PRICES.iloc[: t + 1], FIELDS.iloc[: t + 1])
        assert part.buy.iloc[-1] == full.buy.iloc[t]
        assert part.sell.iloc[-1] == full.sell.iloc[t]
