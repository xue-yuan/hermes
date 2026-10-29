"""每日選股報告：自選股 × 策略，列出最新一天新觸發、條件仍成立，以及形成中。

用實際成交價判斷（還原模式用還原價格）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from hermes.data import DataError, DataLoader
from hermes.services.signals import (
    SignalHistory,
    StrategyNotFoundError,
    gap_warnings,
    history,
    load_fields,
    missing_volume_warnings,
    resolve_strategy_path,
    strategy_files,
)
from hermes.strategies import RuleStrategy, StrategyConfigError, load_strategy

SIDES = ("buy", "sell")


@dataclass
class ScreenRow:
    stock_id: str
    name: str
    strategy_id: str  # 策略設定檔的檔名（不含副檔名）
    strategy: RuleStrategy
    last_date: pd.Timestamp
    close: float
    streak: dict[str, int]  # 買進、賣出條件連續成立的天數
    forming_streak: int = 0  # 形成中的條件連續成立的天數；沒有寫這個條件時是 0
    values: pd.Series | None = None  # 最新一天的價格、指標數值（欄位同 RuleResult.values）


@dataclass
class ScreenReport:
    adjusted: bool = False  # True：還原模式，用還原價格判斷
    rows: list[ScreenRow] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)  # （對象, 錯誤訊息）
    warnings: list[tuple[str, str]] = field(
        default_factory=list
    )  # （對象, 提醒），例如無法解釋的跳空

    @property
    def latest_date(self) -> pd.Timestamp | None:
        return max((r.last_date for r in self.rows), default=None)

    def stale(self) -> dict[str, pd.Timestamp]:
        """資料日期比其他股票舊的股票（通常是還沒更新）。"""
        latest = self.latest_date
        return {r.stock_id: r.last_date for r in self.rows if r.last_date < latest}

    def triggered_today(self) -> list[tuple[ScreenRow, str]]:
        return [(r, side) for r in self.rows for side in SIDES if r.streak[side] == 1]

    def ongoing(self) -> list[tuple[ScreenRow, str, int]]:
        return [
            (r, side, r.streak[side])
            for r in self.rows
            for side in SIDES
            if r.streak[side] > 1
        ]

    def forming(self) -> list[tuple[ScreenRow, int]]:
        """形成中的條件今天成立、今天沒有觸發買進；依連續天數由少到多排序。"""
        rows = [
            (r, r.forming_streak)
            for r in self.rows
            if r.forming_streak > 0 and r.streak["buy"] == 0
        ]
        return sorted(rows, key=lambda x: x[1])


def screen(
    stock_ids: list[str],
    strategy_refs: list[str] | None = None,
    loader: DataLoader | None = None,
    directory: Path | None = None,
    adjusted: bool = False,
) -> ScreenReport:
    """每檔股票只讀一次資料，每個策略只讀一次設定檔；個別失敗會記在 errors，不影響其他組合。

    adjusted=True 是還原模式：用還原價格判斷。
    """
    loader = loader or DataLoader()
    report = ScreenReport(adjusted)
    refs = strategy_refs or [p.stem for p in strategy_files(directory)]
    strategies: list[tuple[str, RuleStrategy]] = []
    for ref in refs:
        try:
            strategies.append(
                (Path(ref).stem, load_strategy(resolve_strategy_path(ref, directory)))
            )
        except (StrategyConfigError, StrategyNotFoundError) as exc:
            report.errors.append((f"策略 {ref}", str(exc)))

    for sid in dict.fromkeys(stock_ids):
        try:
            prices = loader.prices(sid, adjust=adjusted)
        except DataError as exc:
            report.errors.append((f"股票 {sid}", str(exc)))
            continue
        report.warnings += [
            (f"股票 {sid}", msg) for msg in gap_warnings(loader, sid, adjusted)
        ]
        try:
            name = str(loader.info(sid)["stock_name"])
        except DataError:
            name = ""
        needed = (
            set().union(*(s.data_fields for _, s in strategies))
            if strategies
            else set()
        )
        try:
            fields, missing = load_fields(loader, sid, needed, prices.index)
            field_error = None
        except DataError as exc:
            fields, missing, field_error = None, [], str(exc)
        report.warnings += [(f"股票 {sid}", msg) for msg in missing]
        for strategy_id, strategy in strategies:
            if strategy.data_fields and field_error:
                report.errors.append((f"股票 {sid}（策略 {strategy_id}）", field_error))
                continue
            hist = history(sid, strategy, adjusted, prices, fields)
            report.warnings += [
                (f"股票 {sid}（策略 {strategy_id}）", msg)
                for msg in missing_volume_warnings(loader, hist)
            ]
            report.rows.append(
                ScreenRow(
                    stock_id=sid,
                    name=name,
                    strategy_id=strategy_id,
                    strategy=strategy,
                    last_date=prices.index[-1],
                    # 向前還原：最新一天的還原價格就是實際成交價
                    close=float(prices["close"].iloc[-1]),
                    streak=_streaks(hist),
                    forming_streak=int(hist.result.forming_streak.iloc[-1]),
                    values=hist.result.values.iloc[-1],
                )
            )
    return report


def _streaks(hist: SignalHistory) -> dict[str, int]:
    return {
        "buy": int(hist.result.buy_streak.iloc[-1]),
        "sell": int(hist.result.sell_streak.iloc[-1]),
    }
