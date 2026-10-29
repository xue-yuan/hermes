"""執行規則型策略：依指標目錄算出用到的指標，再逐日判斷買賣條件。

每一天的判斷只用到當天和之前的資料，所以拿整段歷史一次算出來的結果，
和「在那一天收盤後當下算」的結果相同。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hermes.strategies.base import stateful
from hermes.strategies.rules import (
    DATA_FIELDS,
    PRICE_FIELDS,
    Condition,
    IndicatorConfig,
    Rule,
    RuleStrategy,
    operand_label,
    ref_parts,
    rule_operands,
)
from hermes.technical import CATALOG


def streak(flags: pd.Series) -> pd.Series:
    """條件連續成立的天數，不成立的日子為 0。等於 1 的那天就是「新觸發」。"""
    return flags.astype(int).groupby((~flags).cumsum()).cumsum()


@dataclass
class RuleResult:
    # 價格欄位、用到的籌碼和本益比欄位、用到的指標；多線指標的欄位名稱為「名稱.線」，例如 kd.k；
    # 前 N 天的值例如 dc.upper（前1天）
    values: pd.DataFrame
    buy: pd.Series  # 每天是否符合買進條件（bool）
    sell: pd.Series  # 每天是否符合賣出條件（bool）；沒有賣出規則時全為 False
    ready: pd.DataFrame  # 和 values 同樣的欄位：每天是否已經過了暖機（見 ready_frame）
    forming: pd.Series  # 每天是否符合形成中的條件（bool）；沒有寫時全為 False。不影響買賣
    # 只有形成中的條件用到的欄位（籌碼、本益比、前 N 天的值）；暖機天數不看它們，因為不影響買賣
    forming_only: tuple[str, ...] = ()

    @property
    def buy_streak(self) -> pd.Series:
        return streak(self.buy)

    @property
    def sell_streak(self) -> pd.Series:
        return streak(self.sell)

    @property
    def forming_streak(self) -> pd.Series:
        return streak(self.forming)

    @property
    def warmup_end(self) -> pd.Timestamp | None:
        """買賣用到的指標都過了暖機的第一天；在這之前條件一律不成立。資料不夠時為 None。"""
        skip = [*PRICE_FIELDS, *self.forming_only]
        ready = self.ready.drop(columns=skip).all(axis=1)
        return ready.idxmax() if ready.any() else None

    def position(self, start: pd.Timestamp | None = None) -> pd.Series:
        """回測用的目標部位：從 start 那天空手開始，買進條件成立後持有（1），直到賣出條件成立（0）。

        不加碼、不放空。同一天買進、賣出條件都成立時，已經買進就賣出，還沒買進就不買。
        start 之前為 NaN，代表還沒開始照策略操作。
        """
        out = pd.Series(np.nan, index=self.buy.index)
        live = self.buy.index >= start if start is not None else np.full(len(out), True)
        out[live] = stateful(self.buy[live], self.sell[live]).to_numpy()
        return out


def indicator_frame(
    indicators: dict[str, IndicatorConfig], prices: pd.DataFrame
) -> pd.DataFrame:
    """依指標目錄算出每個有名稱的指標；多線指標的欄位名稱為「名稱.線」，例如 kd.k。"""
    out = pd.DataFrame(index=prices.index)
    for name, cfg in indicators.items():
        spec = CATALOG[cfg.type]
        values = spec.compute(prices, **spec.resolve_params(cfg.params))
        if spec.outputs:
            for line in spec.outputs:
                out[f"{name}.{line}"] = values[line]
        else:
            out[name] = values
    return out


def ready_frame(
    indicators: dict[str, IndicatorConfig], values: pd.DataFrame
) -> pd.DataFrame:
    """每天、每個欄位是否已經過了暖機。

    一般指標有數值就是過了暖機。型態類指標（例如 Darvas 箱）沒有值也是正常的結果（當時沒有箱子），
    改看指標目錄給的暖機天數：暖機以後，有沒有值都算過了暖機。
    """
    ready = values.notna()
    for name, cfg in indicators.items():
        spec = CATALOG[cfg.type]
        if spec.warmup is None:
            continue
        days = spec.warmup(**spec.resolve_params(cfg.params))
        flags = np.arange(len(values)) >= days
        for line in spec.outputs or (None,):
            ready[f"{name}.{line}" if line else name] = flags
    return ready


def indicator_values(
    strategy: RuleStrategy, prices: pd.DataFrame, fields: pd.DataFrame | None = None
) -> pd.DataFrame:
    """fields：籌碼、本益比欄位（index 為日期），策略有用到時才需要；沒有紀錄的日子是空值，條件不成立。"""
    values = prices[list(PRICE_FIELDS)].copy()
    needed = sorted(strategy.data_fields)
    missing = [f for f in needed if fields is None or f not in fields.columns]
    if missing:
        raise ValueError(
            f"{strategy.name} 用到 {'、'.join(missing)}，需要傳入這些欄位的資料"
        )
    for f in needed:
        values[f] = fields[f].reindex(values.index)
    values = values.join(indicator_frame(strategy.indicators, prices))
    # 條件用到前 N 天的值時，另外加一欄平移後的值，歷史表格看得到
    for cond in strategy.conditions():
        for operand in (cond.left, cond.right):
            name, ago = ref_parts(operand)
            if ago:
                values[operand_label(operand)] = values[name].shift(ago)
    return values


def run_rules(
    strategy: RuleStrategy, prices: pd.DataFrame, fields: pd.DataFrame | None = None
) -> RuleResult:
    values = indicator_values(strategy, prices, fields)
    buy = evaluate(strategy.buy, values)
    sell = (
        evaluate(strategy.sell, values)
        if strategy.sell
        else pd.Series(False, index=values.index)
    )
    forming = (
        evaluate(strategy.forming, values)
        if strategy.forming
        else pd.Series(False, index=values.index)
    )
    ready = ready_frame(strategy.indicators, values)
    for cond in strategy.conditions():
        for operand in (cond.left, cond.right):
            name, ago = ref_parts(operand)
            if ago:  # 前 N 天的值：N 天前過了暖機才算
                ready[operand_label(operand)] = ready[name].shift(ago, fill_value=False)
    # 只有形成中的條件用到的籌碼、本益比欄位和前 N 天的值；指標不管有沒有用到都會算，暖機也都看
    trade = {
        operand_label(o)
        for c in strategy.conditions(forming=False)
        for o in (c.left, c.right)
    }
    forming_only = tuple(
        label
        for label in map(operand_label, rule_operands(strategy.forming))
        if label not in trade
        and (label in DATA_FIELDS or label not in strategy.operands())
    )
    return RuleResult(
        values=values,
        buy=buy,
        sell=sell,
        ready=ready,
        forming=forming,
        forming_only=forming_only,
    )


# 兩邊數學上相等時（例如收盤價剛好等於 5 日均線），算平均的捨入誤差會讓結果差一點點，
# 大於、小於就會被誤判。相差在 Python math.isclose 預設的相對誤差（1e-9，約 9 位有效數字，
# PEP 485）以內，就當成相等。
REL_TOL = 1e-9


def compare(op: str, left: pd.Series, right: pd.Series | float) -> pd.Series:
    """逐日比大小；相對誤差 REL_TOL 以內當成相等。有空值的日子一律為 False。"""
    equal = np.abs(left - right) <= REL_TOL * np.maximum(np.abs(left), np.abs(right))
    if op == "gt":
        return (left > right) & ~equal
    if op == "gte":
        return (left > right) | equal
    if op == "lt":
        return (left < right) & ~equal
    return (left < right) | equal


def evaluate(rule: Rule, values: pd.DataFrame) -> pd.Series:
    if not isinstance(rule, Condition):
        parts = [evaluate(r, values) for r in rule.rules]
        out = parts[0]
        for p in parts[1:]:
            out = (out & p) if rule.all is not None else (out | p)
        return out

    left = values[operand_label(rule.left)]
    right = (
        values[operand_label(rule.right)]
        if ref_parts(rule.right)[0] is not None
        else rule.right
    )
    prev_left = left.shift(1)
    prev_right = right.shift(1) if isinstance(right, pd.Series) else right
    # 資料不足（例如指標暖機期）時比較結果為 False，不會產生訊號
    if rule.op in ("gt", "gte", "lt", "lte"):
        hit = compare(rule.op, left, right)
    elif rule.op == "crosses_above":
        hit = compare("gt", left, right) & compare("lte", prev_left, prev_right)
    else:
        hit = compare("lt", left, right) & compare("gte", prev_left, prev_right)
    if rule.days > 1:
        hit = (
            hit.astype(int).rolling(rule.days, min_periods=rule.days).sum() == rule.days
        )
    return hit.astype(bool)
