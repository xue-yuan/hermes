"""行情資料：下載、每日更新、自選股、查看資料。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from hermes.data import (
    BENCHMARK_ID,
    DataError,
    DataLoader,
    FinMindRateLimitError,
    MarketDataSync,
    StockSyncResult,
    UnknownStockError,
    WatchItem,
    Watchlist,
)


@dataclass
class FetchReport:
    results: list[StockSyncResult] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # 代號 → 錯誤訊息
    skipped: list[str] = field(default_factory=list)  # 請求次數用完而沒處理到的代號
    benchmark_rows: int | None = None

    @property
    def ok(self) -> bool:
        return not self.failed and not self.skipped


@dataclass(frozen=True)
class WatchChange:
    stock_id: str
    status: str  # added、exists、removed、absent、not_found、error
    name: str = ""
    message: str = ""


class MarketDataReader:
    """只讀本機資料：自選股清單、日 K、股票名稱。網頁用這個，不會下載或寫入資料。"""

    def __init__(
        self, loader: DataLoader | None = None, watchlist: Watchlist | None = None
    ):
        self.loader = loader or DataLoader()
        self._watchlist = watchlist

    @property
    def watchlist(self) -> Watchlist:
        if self._watchlist is None:
            self._watchlist = Watchlist()
        return self._watchlist

    def close(self) -> None:
        if self._watchlist is not None:
            self._watchlist.close()
            self._watchlist = None

    def list_watchlist(self) -> list[WatchItem]:
        return self.watchlist.items()

    def price_history(self, stock_id: str, adjust: bool = False) -> pd.DataFrame:
        """本機的日 K。adjust=False 為實際成交價，可直接對照看盤軟體。"""
        return self.loader.prices(stock_id, adjust=adjust)

    def stock_name(self, stock_id: str) -> str:
        try:
            return str(self.loader.info(stock_id)["stock_name"])
        except DataError:
            return ""

    def last_price_date(self, stock_id: str) -> date | None:
        try:
            raw = self.loader.read("TaiwanStockPrice", stock_id)
        except DataError:
            return None
        return None if raw.empty else date.fromisoformat(str(raw["date"].max())[:10])


class MarketDataService(MarketDataReader):
    """下載、每日更新、管理自選股；只有這裡會透過 data/sync.py 連網寫入。"""

    def __init__(
        self,
        sync: MarketDataSync | None = None,
        loader: DataLoader | None = None,
        watchlist: Watchlist | None = None,
    ):
        self.sync = sync or MarketDataSync()
        super().__init__(loader or DataLoader(self.sync.cache), watchlist)

    # ---------------------------------------------------------------- 下載
    def fetch(
        self,
        stock_ids: list[str],
        start: date | None = None,
        end: date | None = None,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> FetchReport:
        """下載多檔股票，最後更新大盤指數。

        單檔失敗會記下來並繼續下一檔；請求次數用完就停止，剩下的代號列在 skipped，
        下次執行會從各檔的最後一筆接續下載。
        """
        report = FetchReport()
        pending = list(dict.fromkeys(stock_ids))
        while pending:
            sid = pending.pop(0)
            notify = (lambda ds, sid=sid: on_progress(sid, ds)) if on_progress else None
            try:
                report.results.append(
                    self.sync.sync_stock(sid, start, end, on_dataset=notify)
                )
            except FinMindRateLimitError as exc:
                report.failed[sid] = str(exc)
                report.skipped = pending
                return report
            except DataError as exc:
                report.failed[sid] = str(exc)
        if on_progress:
            on_progress(BENCHMARK_ID, "TaiwanStockPrice")
        try:
            report.benchmark_rows = self.sync.sync_benchmark()
        except DataError as exc:
            report.failed[BENCHMARK_ID] = str(exc)
        return report

    def update(
        self, on_progress: Callable[[str, str], None] | None = None
    ) -> FetchReport:
        """依自選股清單接續下載，給每日排程使用。"""
        return self.fetch(self.watchlist.stock_ids(), on_progress=on_progress)

    # -------------------------------------------------------------- 自選股
    def add_to_watchlist(self, stock_ids: list[str]) -> list[WatchChange]:
        changes = []
        for sid in dict.fromkeys(stock_ids):
            try:
                info = self.sync.sync_info(sid)
            except UnknownStockError as exc:
                changes.append(WatchChange(sid, "not_found", message=str(exc)))
                continue
            except DataError as exc:
                changes.append(WatchChange(sid, "error", message=str(exc)))
                continue
            name = str(info.iloc[0]["stock_name"])
            status = "added" if self.watchlist.add(sid, name) else "exists"
            changes.append(WatchChange(sid, status, name))
        return changes

    def remove_from_watchlist(self, stock_ids: list[str]) -> list[WatchChange]:
        return [
            WatchChange(sid, "removed" if self.watchlist.remove(sid) else "absent")
            for sid in dict.fromkeys(stock_ids)
        ]
