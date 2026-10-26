"""向量化回測引擎。

訊號（目標部位權重）以第 t 日收盤後可得的資訊計算，預設在 t+1 日開盤成交：
- 隔夜跳空（t 收盤 → t+1 開盤）由舊部位承擔；
- 開盤調整到新部位並支付成本，日內報酬（t+1 開盤 → 收盤）由新部位承擔。
這樣可避免「用收盤價產生訊號又用同一個收盤價成交」的前視偏差。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hermes.backtest import metrics as m
from hermes.backtest.costs import CostModel


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    position: pd.Series
    trades: pd.DataFrame
    costs: pd.Series
    benchmark_returns: pd.Series
    metrics: dict
    params: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()

    @property
    def benchmark_equity(self) -> pd.Series:
        return (1.0 + self.benchmark_returns).cumprod()

    @property
    def benchmark_metrics(self) -> dict:
        """買進持有（benchmark_returns）的績效；一直持有，所以部位都是 1。"""
        index = self.benchmark_returns.index
        return m.summarize(self.benchmark_returns, pd.Series(1.0, index=index))


def run_backtest(
    prices: pd.DataFrame,
    target: pd.Series,
    costs: CostModel | None = None,
    execution: str = "open",
    name: str = "strategy",
    params: dict | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    rf: float = 0.0,
) -> BacktestResult:
    """target：每日收盤後決定的目標權重（1=滿倉做多，0=空手，負值=放空）。"""
    costs = costs or CostModel()
    df = prices[["open", "close"]].copy()
    target = target.reindex(df.index).astype(float)

    if execution == "open":
        w = target.shift(1).fillna(0.0)  # t 日開盤時持有的部位
    elif execution == "close":
        w = target.fillna(0.0)  # t 日收盤成交（假設能以訊號當根收盤價成交）
    else:
        raise ValueError("execution 必須是 'open' 或 'close'")

    if start is not None or end is not None:
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= df.index >= pd.Timestamp(start)
        if end is not None:
            mask &= df.index <= pd.Timestamp(end)
        df, w = df[mask], w[mask]

    # 期初從空手開始：第一天的 w_prev 為 0，因此首日以開盤價建倉並支付買進成本
    prev_close = df["close"].shift(1)
    w_prev = w.shift(1).fillna(0.0)
    delta = w - w_prev
    cost = (
        delta.clip(lower=0.0) * costs.buy_rate
        + (-delta).clip(lower=0.0) * costs.sell_rate
    )

    if execution == "open":
        gap = (df["open"] / prev_close - 1.0).fillna(0.0)
        intraday = df["close"] / df["open"] - 1.0
        ret = (1.0 + w_prev * gap) * (1.0 + w * intraday - cost) - 1.0
    else:
        daily = (df["close"] / prev_close - 1.0).fillna(0.0)
        ret = w_prev * daily - cost

    bench = (df["close"] / prev_close - 1.0).fillna(0.0)
    if len(bench):
        # 買進持有基準：第一天以開盤價買入
        bench.iloc[0] = df["close"].iloc[0] / df["open"].iloc[0] - 1.0 - costs.buy_rate

    trades = extract_trades(ret, w)
    result_metrics = m.summarize(ret, w, trades, bench, rf=rf)
    return BacktestResult(
        name=name,
        returns=ret,
        position=w,
        trades=trades,
        costs=cost,
        benchmark_returns=bench,
        metrics=result_metrics,
        params=params or {},
    )


def extract_trades(returns: pd.Series, position: pd.Series) -> pd.DataFrame:
    """把連續同方向持倉視為一筆交易；出場當日（含隔夜跳空與賣出成本）也計入該筆交易。"""
    sign = np.sign(position.round(10))
    rows = []
    in_trade = False
    entry_i = 0
    side = 0.0
    idx = position.index
    r = returns.to_numpy()
    s = sign.to_numpy()
    for i in range(len(s)):
        if not in_trade and s[i] != 0:
            in_trade, entry_i, side = True, i, s[i]
        elif in_trade and s[i] != side:
            exit_i = i
            rows.append(
                (
                    idx[entry_i],
                    idx[exit_i],
                    side,
                    float(np.prod(1 + r[entry_i : exit_i + 1]) - 1),
                    exit_i - entry_i,
                )
            )
            in_trade = s[i] != 0
            if in_trade:
                entry_i, side = i, s[i]
    if in_trade:
        rows.append(
            (
                idx[entry_i],
                pd.NaT,
                side,
                float(np.prod(1 + r[entry_i:]) - 1),
                len(s) - entry_i,
            )
        )
    return pd.DataFrame(rows, columns=["entry", "exit", "side", "return", "bars"])
