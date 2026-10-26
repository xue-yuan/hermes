"""從 FinMind 下載資料寫進本機快取；FinMind 沒有的股數比例向證交所下載。
整個專案只有這裡會連網寫入；讀取請用 DataLoader。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from hermes.config import HISTORY_START, home_dir
from hermes.data.errors import DataError, UnknownStockError, UnsupportedMarketError
from hermes.data.finmind import FinMindClient
from hermes.data.loader import SHARE_RATIO_COLUMNS, SHARE_RATIO_DATASET, DataLoader
from hermes.data.store import Cache
from hermes.data.twse import TwseClient

FULL_HISTORY_START = HISTORY_START  # 第一次下載的起始日

# 增量更新時往回多抓幾天，涵蓋 FinMind 事後補上或修正的資料
OVERLAP_DAYS = 7

# 逐日資料集與去除重複用的欄位：每次都更新，有快取時只抓最後幾天再合併
DAILY_DATASETS = {
    "TaiwanStockPrice": ["date"],
    "TaiwanStockPER": ["date"],
    "TaiwanStockInstitutionalInvestorsBuySell": ["date", "name"],
    "TaiwanStockMarginPurchaseShortSale": ["date"],
    "TaiwanStockShareholding": ["date"],
    # 除權息、減資、分割事件要每天更新，否則還原會漏掉最新一次事件
    "TaiwanStockDividendResult": ["date"],
    "TaiwanStockCapitalReductionReferencePrice": ["date"],
    "TaiwanStockSplitPrice": ["date"],
}

# 基本資料（名稱、市場別）：快取超過 INFO_MAX_AGE_HOURS 才重新下載
INFO_DATASET = "TaiwanStockInfo"
INFO_MAX_AGE_HOURS = 24 * 7

# 只處理上市股票：基本資料的市場別（type）要是這個
LISTED_MARKET = "twse"

BENCHMARK_ID = "TAIEX"

DATASET_LABELS = {
    "TaiwanStockInfo": "基本資料",
    "TaiwanStockPrice": "股價",
    "TaiwanStockPER": "本益比／淨值比",
    "TaiwanStockInstitutionalInvestorsBuySell": "三大法人",
    "TaiwanStockMarginPurchaseShortSale": "融資融券",
    "TaiwanStockShareholding": "外資持股",
    "TaiwanStockDividendResult": "除權息結果",
    "TaiwanStockCapitalReductionReferencePrice": "減資參考價",
    "TaiwanStockSplitPrice": "分割／面額變更",
    SHARE_RATIO_DATASET: "股數比例（證交所）",
}


@dataclass
class StockSyncResult:
    stock_id: str
    added_price_rows: int
    price_rows: int
    price_start: date | None
    price_end: date | None
    refreshed: list[str] = field(default_factory=list)
    # 股數改變、但沒拿到證交所股數比例的事件：這之前的還原成交量算不出來
    missing_share_ratios: list[str] = field(default_factory=list)


class MarketDataSync:
    def __init__(
        self,
        client: FinMindClient | None = None,
        cache: Cache | None = None,
        info_max_age_hours: float = INFO_MAX_AGE_HOURS,
        twse: TwseClient | None = None,
    ):
        self.client = client or FinMindClient()
        self.cache = cache or Cache(home_dir())
        self.info_max_age_hours = info_max_age_hours
        self._twse = twse

    @property
    def twse(self) -> TwseClient:
        """有新的股數變動事件才會用到，用到時才建立。"""
        if self._twse is None:
            self._twse = TwseClient()
        return self._twse

    def sync_stock(
        self,
        stock_id: str,
        start: date | None = None,
        end: date | None = None,
        on_dataset: Callable[[str], None] | None = None,
    ) -> StockSyncResult:
        """下載單一股票的所有資料集。

        start 為 None：有快取就從最後一筆往回 OVERLAP_DAYS 天接續下載，沒有快取就下載完整歷史。
        指定 start：重新下載 start～end 的逐日資料，覆蓋同日期的舊資料，其他日期保留。
        不是上市股票時丟出 UnsupportedMarketError，什麼都不下載。
        """
        notify = on_dataset or (lambda _: None)
        refreshed = []
        if self._is_fresh(INFO_DATASET, stock_id):
            _require_listed(stock_id, self.cache.read(INFO_DATASET, stock_id))
        else:
            notify(INFO_DATASET)
            self.sync_info(stock_id)
            refreshed.append(INFO_DATASET)

        before = self._rows("TaiwanStockPrice", stock_id)
        for dataset, keys in DAILY_DATASETS.items():
            notify(dataset)
            self._sync_daily(dataset, stock_id, keys, start, end)
            refreshed.append(dataset)
        notify(SHARE_RATIO_DATASET)
        missing = self.sync_share_ratios(stock_id)
        refreshed.append(SHARE_RATIO_DATASET)

        price = self.cache.read("TaiwanStockPrice", stock_id)
        has_price = price is not None and not price.empty
        return StockSyncResult(
            stock_id=stock_id,
            added_price_rows=self._rows("TaiwanStockPrice", stock_id) - before,
            price_rows=len(price) if has_price else 0,
            price_start=date.fromisoformat(str(price["date"].min())[:10])
            if has_price
            else None,
            price_end=date.fromisoformat(str(price["date"].max())[:10])
            if has_price
            else None,
            refreshed=refreshed,
            missing_share_ratios=missing,
        )

    def sync_share_ratios(self, stock_id: str) -> list[str]:
        """股數改變的事件（除權、減資、變更面額），向證交所下載每股變成幾股，存進本機。

        已經下載過的事件不再下載，所以平常沒有新事件時不會連到證交所。
        回傳沒拿到比例的事件和原因；證交所忙碌、下載失敗的，下次執行會再試。
        """
        changes = DataLoader(self.cache).share_changes(stock_id)
        todo = changes[changes["ratio"].isna()]
        if todo.empty:
            return []
        days = self._trading_days(stock_id)
        records, missing = [], []
        for _, ev in todo.iterrows():
            day = ev["date"].date()
            label = f"{day} {ev['source']}"
            try:
                found = self._share_ratio(stock_id, ev["source"], day, days)
            except DataError as exc:
                missing.append(f"{label}：{exc}，下次執行會再試")
                continue
            if isinstance(found, str):
                missing.append(f"{label}：{found}")
                continue
            records.append(
                {"date": day.isoformat(), "source": ev["source"]}
                | {k: float(found[k]) for k in SHARE_RATIO_COLUMNS}
            )
        if records:
            cached = self.cache.read(SHARE_RATIO_DATASET, stock_id)
            merged = pd.concat(
                [x for x in (cached, pd.DataFrame(records)) if x is not None and not x.empty],
                ignore_index=True,
            ).drop_duplicates(["date", "source"], keep="last")
            self.cache.write(
                SHARE_RATIO_DATASET, stock_id, merged.sort_values("date").reset_index(drop=True)
            )
        return missing

    def _share_ratio(
        self, stock_id: str, source: str, day: date, days: pd.DatetimeIndex
    ) -> dict | str:
        """官方明細；找不到時回傳原因。day 是事件當天（除權息日、恢復買賣日）。"""
        if source == "除權":
            found = self.twse.rights_detail(stock_id, day)
        elif source == "減資":
            found = self.twse.reduction_detail(stock_id, day)
        elif source == "面額變更":
            # 明細的檔案日期是停止買賣前最後一個交易日，也就是本機資料裡事件前的最後一個交易日；
            # 回傳的停止買賣日期要在這天之後、不晚於恢復買賣日，才是這次事件
            before = days[days < pd.Timestamp(day)]
            if before.empty:
                return "本機沒有停止買賣前的交易資料，找不到證交所的明細"
            found = self.twse.par_change_detail(stock_id, before[-1].date())
            if found is not None and not (
                before[-1] < pd.Timestamp(found["stop_date"]) <= pd.Timestamp(day)
            ):
                found = None
        else:
            return f"證交所沒有公布{source}的股數比例"
        return found if found is not None else "證交所查不到這次的明細"

    def _trading_days(self, stock_id: str) -> pd.DatetimeIndex:
        """FinMind 有紀錄的交易日，包含沒有成交的日子（記為全 0）。"""
        raw = self.cache.read("TaiwanStockPrice", stock_id)
        if raw is None or raw.empty:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(pd.to_datetime(raw["date"]).sort_values().unique())

    def sync_info(self, stock_id: str) -> pd.DataFrame:
        """下載股票基本資料。查無此代號時丟出 UnknownStockError，不是上市股票時丟出 UnsupportedMarketError，
        兩種情況都不寫入快取。"""
        info = self._get(INFO_DATASET, stock_id, FULL_HISTORY_START)
        if info.empty:
            raise UnknownStockError(f"查無股票代號 {stock_id}")
        _require_listed(stock_id, info)
        self.cache.write(INFO_DATASET, stock_id, info)
        return info

    def sync_benchmark(self) -> int:
        """更新加權指數（TAIEX）日 K，回傳總筆數。"""
        self._sync_daily("TaiwanStockPrice", BENCHMARK_ID, ["date"], None, None)
        return self._rows("TaiwanStockPrice", BENCHMARK_ID)

    def _sync_daily(
        self,
        dataset: str,
        data_id: str,
        keys: list[str],
        start: date | None,
        end: date | None,
    ) -> None:
        cached = self.cache.read(dataset, data_id)
        has_cache = cached is not None and not cached.empty
        if start is None:
            if has_cache:
                last = date.fromisoformat(str(cached["date"].max())[:10])
                start = last - timedelta(days=OVERLAP_DAYS)
            else:
                start = FULL_HISTORY_START
        fresh = self._get(dataset, data_id, start, end)
        if not has_cache:
            merged = fresh
        elif fresh.empty:
            return
        else:
            merged = pd.concat([cached, fresh], ignore_index=True)
            merged = merged.drop_duplicates(subset=keys, keep="last").sort_values(keys)
        self.cache.write(dataset, data_id, merged.reset_index(drop=True))

    def _get(
        self, dataset: str, data_id: str, start: date, end: date | None = None
    ) -> pd.DataFrame:
        return self.client.get(
            dataset,
            data_id,
            start_date=start.isoformat(),
            end_date=end.isoformat() if end else None,
        )

    def _is_fresh(self, dataset: str, data_id: str) -> bool:
        age = self.cache.age_hours(dataset, data_id)
        return age is not None and age < self.info_max_age_hours

    def _rows(self, dataset: str, data_id: str) -> int:
        cached = self.cache.read(dataset, data_id)
        return 0 if cached is None else len(cached)


MARKET_NAMES = {"tpex": "上櫃", "emerging": "興櫃"}


def _require_listed(stock_id: str, info: pd.DataFrame) -> None:
    """市場別和 DataLoader.info 一樣看第一列。"""
    market = str(info.iloc[0].get("type", ""))
    if market != LISTED_MARKET:
        name = str(info.iloc[0].get("stock_name", ""))
        raise UnsupportedMarketError(
            f"{stock_id} {name}是{MARKET_NAMES.get(market, market or '市場別不明')}股票，"
            f"hermes 只處理上市股票"
        )
