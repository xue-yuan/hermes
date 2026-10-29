"""回測：設定檔策略、內建策略（機器學習、買進持有），以及多個策略並排比較。

設定檔策略：
- 訊號用實際成交價判斷（還原模式用還原價格判斷）。
- 報酬一律用還原價格計算，才會包含股利；成交價格一律是實際成交價。
- 部位：買進條件成立後持有，直到賣出條件成立；不加碼、不放空。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from hermes.backtest import BacktestResult, CostModel, run_backtest
from hermes.config import strategies_dir
from hermes.data import DataError, DataLoader
from hermes.services.signals import (
    SignalHistory,
    StrategyNotFoundError,
    StrategyState,
    resolve_strategy_path,
    signal_history,
    strategy_state,
)
from hermes.strategies import (
    REGISTRY,
    RuleStrategy,
    StrategyConfigError,
    StrategyContext,
    StrategySpec,
    candidate_closes,
    get_strategy,
)


@dataclass
class RuleBacktest:
    stock_id: str
    strategy: RuleStrategy
    start: pd.Timestamp  # 開始照策略操作的第一天（空手開始）
    warmup_end: pd.Timestamp  # 指標都算得出來的第一天；在這之前條件一律不成立
    execution: str  # open：訊號隔天開盤成交；close：訊號當天收盤成交
    raw_prices: pd.DataFrame  # 實際成交價，用來列出成交價格
    references: pd.Series  # 每天的參考價，用來算漲跌停價
    # 訊號：每天收盤後決定的部位（1 持有、0 空手），start 之前為 NaN
    target: pd.Series
    result: BacktestResult
    adjusted: bool = False  # True：還原模式，訊號用還原價格判斷

    def trades(self) -> pd.DataFrame:
        """每一筆交易：買進訊號日、買進日、買進價、賣出訊號日、賣出日、賣出價、報酬、持有天數。

        價格是實際成交價；報酬用還原價格算，包含股利和交易成本。還沒賣出的交易，賣出欄位為空。
        buy_at_limit、sell_at_limit：用漲停價買進、跌停價賣出，實際上可能排不到。
        """
        res = self.result
        field = "open" if self.execution == "open" else "close"
        index = self.raw_prices.index
        rows = []
        for _, t in res.trades.iterrows():
            entry, exit_ = t["entry"], t["exit"]
            entry_price = float(self.raw_prices.at[entry, field])
            exit_price = (
                float(self.raw_prices.at[exit_, field]) if pd.notna(exit_) else np.nan
            )
            rows.append(
                {
                    "buy_signal": self._signal_day(entry, index),
                    "entry": entry,
                    "entry_price": entry_price,
                    "sell_signal": self._signal_day(exit_, index)
                    if pd.notna(exit_)
                    else pd.NaT,
                    "exit": exit_,
                    "exit_price": exit_price,
                    "return": t["return"],
                    "bars": int(t["bars"]),
                    "buy_at_limit": self._at_limit(entry, entry_price, up=True),
                    "sell_at_limit": pd.notna(exit_)
                    and self._at_limit(exit_, exit_price, up=False),
                }
            )
        columns = [
            "buy_signal",
            "entry",
            "entry_price",
            "sell_signal",
            "exit",
            "exit_price",
            "return",
            "bars",
            "buy_at_limit",
            "sell_at_limit",
        ]
        return pd.DataFrame(rows, columns=columns)

    def _signal_day(self, day: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
        """成交日對應的訊號日：開盤成交是前一個交易日收盤後的訊號，收盤成交是當天。"""
        return index[index.get_loc(day) - 1] if self.execution == "open" else day

    def _at_limit(self, day: pd.Timestamp, price: float, up: bool) -> bool:
        ref = self.references.get(day, np.nan)
        if np.isnan(ref):
            return False
        grid = candidate_closes(float(ref), etf=self.stock_id.startswith("00"))
        return price >= grid[-1] if up else price <= grid[0]

    def latest(self) -> StrategyState:
        """最新一天收盤後：策略是不是買進後還沒賣出，以及這天的訊號要買進、賣出，還是不用動。"""
        return strategy_state(self.target)


def reference_prices(raw: pd.DataFrame, loader: DataLoader, stock_id: str) -> pd.Series:
    """每天的參考價：前一個交易日的收盤價；除權息、減資、分割當天用官方參考價。"""
    ref = raw["close"].shift(1)
    actions = (
        loader.corporate_actions(stock_id)
        .drop_duplicates("date", keep="last")
        .set_index("date")
    )
    on_days = actions.index.intersection(ref.index)
    ref[on_days] = actions.loc[on_days, "reference"].astype(float)
    return ref


def rule_backtest(
    hist: SignalHistory,
    start: date | pd.Timestamp | None = None,
    end: date | pd.Timestamp | None = None,
    costs: CostModel | None = None,
    execution: str = "open",
    loader: DataLoader | None = None,
) -> RuleBacktest:
    """start 預設為指標都算得出來的第一天。指定的 start 比這天早也照用，只是在那之前條件一律不成立。"""
    strategy = hist.strategy
    if strategy.sell is None:
        raise StrategyConfigError(
            f"{strategy.name} 沒有賣出條件（sell），回測不知道什麼時候賣出；請在設定檔加上 sell"
        )
    if execution == "close" and strategy.same_day_fields:
        fields = "、".join(sorted(strategy.same_day_fields))
        raise StrategyConfigError(
            f"{strategy.name} 用到當天的 {fields}，這些收盤後才公布，收盤成交時還不知道（會偷看未來）；"
            f"請改用隔天開盤成交，或改成比前 1 天的值，例如 {{ref: {min(strategy.same_day_fields)}, ago: 1}}"
        )
    warmup_end = hist.result.warmup_end
    if warmup_end is None:
        raise StrategyConfigError(
            f"資料天數不夠，{strategy.name} 用到的指標還沒有數值，沒辦法回測"
        )
    loader = loader or DataLoader()
    # 成交價格用實際成交價，報酬用還原價格（才包含股利）
    raw = loader.prices(hist.stock_id, adjust=False) if hist.adjusted else hist.prices
    adjusted = hist.prices if hist.adjusted else loader.prices(hist.stock_id, adjust=True)
    begin = pd.Timestamp(start) if start is not None else warmup_end
    index = raw.index
    if end is not None:
        index = index[index <= pd.Timestamp(end)]
    index = index[index >= begin]
    if index.empty:
        until = f"{pd.Timestamp(end):%Y-%m-%d}" if end is not None else "最新一天"
        raise DataError(f"{begin:%Y-%m-%d} ～ {until} 之間沒有交易資料可以回測")
    begin = index[0]

    target = hist.result.position(begin)
    return RuleBacktest(
        stock_id=hist.stock_id,
        strategy=strategy,
        start=begin,
        warmup_end=warmup_end,
        execution=execution,
        raw_prices=raw,
        references=reference_prices(raw, loader, hist.stock_id),
        target=target,
        result=run_backtest(adjusted, target, costs, execution, strategy.name, {}, begin, end),
        adjusted=hist.adjusted,
    )


# ---------------------------------------------------------------------------
# 內建策略（機器學習、買進持有）的回測，以及多個策略並排比較

# 機器學習訓練的進度：（策略名稱, 完成幾段, 總共幾段）
Progress = Callable[[str, int, int], None]
SORT_KEYS = ("sharpe", "cagr", "calmar", "max_drawdown")


def strategy_context(
    prices: pd.DataFrame, stock_id: str, loader: DataLoader
) -> StrategyContext:
    """內建策略的輸入；meta-labeling 的主要策略是策略設定檔，用實際成交價判斷。"""

    def rule_position(ref: str) -> pd.Series:
        return (
            signal_history(stock_id, ref, loader=loader)
            .result.position()
            .reindex(prices.index)
        )

    return StrategyContext(
        prices=prices, stock_id=stock_id, loader=loader, rule_position=rule_position
    )


def run_strategy(
    name: str, params: dict, ctx: StrategyContext, on_progress: Progress | None = None
) -> pd.Series:
    spec = get_strategy(name)
    if on_progress is not None and spec.category == "機器學習":
        ctx.progress = lambda done, total: on_progress(spec.title, done, total)
    try:
        return spec.run(ctx, **params)
    finally:
        ctx.progress = None


def is_rule_strategy(ref: str, directory: Path | None = None) -> bool:
    """ref 是 strategies/ 裡的設定檔（True），還是內建策略：機器學習、買進持有（False）。兩邊都有同名時報錯。"""
    try:
        resolve_strategy_path(ref, directory)
    except StrategyNotFoundError:
        return False
    if ref in REGISTRY:
        raise StrategyConfigError(
            f"{ref} 同時是策略設定檔和內建策略（機器學習、買進持有）的名稱，"
            f"請把 {directory or strategies_dir()} 裡的設定檔改名"
        )
    return True


@dataclass
class BuiltinBacktest:
    stock_id: str
    spec: StrategySpec
    params: dict
    label: str  # 策略名稱和指定的參數，例如 HMM(states=3)
    target: pd.Series  # 每天收盤後的目標部位
    result: BacktestResult


def builtin_backtest(
    stock_id: str,
    name: str,
    params: dict | None = None,
    start: date | pd.Timestamp | str | None = None,
    end: date | pd.Timestamp | str | None = None,
    costs: CostModel | None = None,
    execution: str = "open",
    loader: DataLoader | None = None,
    on_progress: Progress | None = None,
) -> BuiltinBacktest:
    """內建策略的回測；start 預設為策略第一天有部位的日子。價格用還原價格。"""
    loader = loader or DataLoader()
    spec = get_strategy(name)
    params = params or {}
    prices = loader.prices(stock_id)
    target = run_strategy(
        name, params, strategy_context(prices, stock_id, loader), on_progress
    )
    label = spec.title + (
        f"({', '.join(f'{k}={v}' for k, v in params.items())})" if params else ""
    )
    result = run_backtest(
        prices,
        target,
        costs,
        execution,
        label,
        params,
        start or target.first_valid_index(),
        end,
    )
    return BuiltinBacktest(stock_id, spec, params, label, target, result)


@dataclass
class StrategyComparison:
    stock_id: str
    sort: str
    results: list[BacktestResult]  # 依 sort 排序，最好的在前面
    ids: list[str]  # 每個結果的策略代號（設定檔檔名或內建策略名稱），和 results 同順序
    benchmark: BacktestResult  # 買進持有
    rules: dict[str, SignalHistory]  # 設定檔策略：{代號: 訊號}
    adjusted: bool = False  # True：還原模式，設定檔策略用還原價格判斷

    @property
    def warnings(self) -> list[str]:
        """設定檔策略缺資料的提醒（籌碼、本益比，或還原價格算不出來的成交量）。"""
        return sorted({w for h in self.rules.values() for w in h.warnings})


def _sort_value(result: BacktestResult, key: str) -> float:
    # 最大回撤是負數，越接近 0 越好，所以全部由大到小排；算不出來的排最後
    return float(np.nan_to_num(result.metrics.get(key, np.nan), nan=-np.inf))


def compare_strategies(
    stock_id: str,
    refs: list[str],
    start: date | pd.Timestamp | str | None = None,
    end: date | pd.Timestamp | str | None = None,
    costs: CostModel | None = None,
    execution: str = "open",
    sort: str = "sharpe",
    loader: DataLoader | None = None,
    directory: Path | None = None,
    on_progress: Progress | None = None,
    adjusted: bool = False,
) -> StrategyComparison:
    """並排比較設定檔策略和內建策略，一定附上買進持有。

    start 預設為所有策略都有訊號的第一天，大家從同一天開始比。
    adjusted=True 是還原模式：設定檔策略用還原價格判斷訊號。
    """
    if not refs:
        raise ValueError("請指定要比較的策略")
    if sort not in SORT_KEYS:
        raise ValueError(f"排序依據只能是 {'、'.join(SORT_KEYS)}")
    if execution not in ("open", "close"):
        raise ValueError("成交方式只能是 open 或 close")
    loader = loader or DataLoader()
    prices = loader.prices(stock_id)
    ctx = strategy_context(prices, stock_id, loader)
    targets, rules = {}, {}
    for ref in dict.fromkeys(refs):
        if is_rule_strategy(ref, directory):
            rules[ref] = signal_history(
                stock_id, ref, adjusted, loader=loader, directory=directory
            )
        else:
            targets[ref] = run_strategy(ref, {}, ctx, on_progress)
    # 設定檔策略從指標都算得出來的那天開始
    firsts = [t.first_valid_index() for t in targets.values()] + [
        h.result.warmup_end for h in rules.values()
    ]
    if start is None and None in firsts:
        raise StrategyConfigError(
            "有策略在整段資料裡都沒有訊號（指標還沒有數值），請指定起始日，或拿掉這個策略"
        )
    start = start or max(firsts)
    entries = [
        (n, run_backtest(prices, t, costs, execution, get_strategy(n).title, {}, start, end))
        for n, t in targets.items()
    ]
    for ref, hist in rules.items():
        entries.append((ref, rule_backtest(hist, start, end, costs, execution, loader).result))
    entries.sort(key=lambda e: _sort_value(e[1], sort), reverse=True)
    benchmark = run_backtest(
        prices,
        get_strategy("buy_hold").run(ctx),
        costs,
        execution,
        "買進持有",
        {},
        start,
        end,
    )
    return StrategyComparison(
        stock_id,
        sort,
        [r for _, r in entries],
        [n for n, _ in entries],
        benchmark,
        rules,
        adjusted,
    )
