"""籌碼面摘要：三大法人買賣超、外資持股比例、融資融券。

「近 N 日合計」「和 N 日前相比」的天數一律由使用者指定，沒有預設值。
天數以交易日計算：各資料先對齊到交易日，區間內有一天沒有紀錄，結果就是空值，不補值。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WHO = {"foreign": "外資", "trust": "投信", "dealer": "自營商", "total": "三大法人合計"}


@dataclass
class ChipReport:
    as_of: pd.Timestamp | None  # 三大法人資料的最後一天
    flows: (
        pd.DataFrame
    )  # index=法人, columns=指定的「N 日」買賣超合計（張）；沒指定天數時沒有欄位
    streaks: dict  # 連續買超（正）、賣超（負）天數
    foreign_ratio: dict  # date、now，以及 chg_N：和 N 個交易日前相比（百分點）
    margin: dict  # date、融資融券餘額、使用率、券資比，以及 margin_chg_N：和 N 個交易日前相比（張）
    recent: pd.DataFrame  # 最近幾個交易日的明細（張）


def _on_trading_days(df: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """對齊到交易日，只到這份資料的最後一天（還沒公布的日子不算缺資料）。"""
    return df.reindex(trading_days[trading_days <= df.index.max()])


def _streak(s: pd.Series) -> int:
    """最近連續買超（正）或賣超（負）天數；最後一天沒有紀錄或為 0 時是 0。"""
    sign = np.sign(s.to_numpy())
    if len(sign) == 0 or np.isnan(sign[-1]) or sign[-1] == 0:
        return 0
    n = 0
    for v in sign[::-1]:
        if v != sign[-1]:
            break
        n += 1
    return int(n * sign[-1])


def _sum(s: pd.Series, n: int) -> float:
    """最近 n 個交易日合計；資料不到 n 天或其中有一天沒有紀錄時為空值。"""
    w = s.iloc[-n:]
    return float(w.sum()) if len(w) == n and w.notna().all() else np.nan


def _change(s: pd.Series, n: int) -> float:
    """最新值和 n 個交易日前相比；資料不夠或任一天沒有紀錄時為空值。"""
    return float(s.iloc[-1] - s.iloc[-1 - n]) if len(s) > n else np.nan


def chip_summary(
    trading_days: pd.DatetimeIndex,
    institutional: pd.DataFrame,
    shareholding: pd.DataFrame,
    margin: pd.DataFrame,
    windows: tuple[int, ...] = (),
    recent_days: int = 10,
) -> ChipReport:
    """institutional、shareholding、margin 是 DataLoader 同名方法讀出的資料；windows 是要看的 N 日。"""
    windows = tuple(dict.fromkeys(windows))
    flows, streaks, recent, as_of = pd.DataFrame(), {}, pd.DataFrame(), None
    if not institutional.empty:
        lots = _on_trading_days(institutional / 1000.0, trading_days)
        as_of = lots.index[-1]
        flows = pd.DataFrame(
            {f"{w} 日": {k: _sum(lots[k], w) for k in WHO} for w in windows},
            index=list(WHO),
        )
        flows.index = [WHO[k] for k in flows.index]
        streaks = {WHO[k]: _streak(lots[k]) for k in ("foreign", "trust", "dealer")}
        recent = lots.iloc[-recent_days:].rename(columns=WHO)

    foreign_ratio: dict = {}
    if not shareholding.empty:
        fr = _on_trading_days(shareholding, trading_days)["foreign_ratio"]
        foreign_ratio = {"date": fr.index[-1], "now": float(fr.iloc[-1])}
        foreign_ratio.update({f"chg_{w}": _change(fr, w) for w in windows})

    margin_info: dict = {}
    if not margin.empty:
        mg = _on_trading_days(margin, trading_days)
        mb, limit = mg["margin_balance"], mg["margin_limit"].iloc[-1]
        margin_info = {
            "date": mg.index[-1],
            "margin_balance": float(mb.iloc[-1]),
            "margin_usage": float(mb.iloc[-1] / limit) if limit else np.nan,
            "short_balance": float(mg["short_balance"].iloc[-1]),
            "short_margin_ratio": float(mg["short_margin_ratio"].iloc[-1]),
        }
        margin_info.update({f"margin_chg_{w}": _change(mb, w) for w in windows})
    return ChipReport(
        as_of=as_of,
        flows=flows,
        streaks=streaks,
        foreign_ratio=foreign_ratio,
        margin=margin_info,
        recent=recent,
    )
