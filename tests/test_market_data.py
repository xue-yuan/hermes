from datetime import date

import pandas as pd
import pytest

from hermes.data import (
    Cache,
    DataLoader,
    FinMindRateLimitError,
    MarketDataSync,
    StockSyncResult,
    UnknownStockError,
    UnsupportedMarketError,
    Watchlist,
)
from hermes.data.sync import DAILY_DATASETS
from hermes.services import MarketDataService


class FakeClient:
    """模擬 FinMind：依 start_date、end_date 篩選日期，並記錄每次請求。"""

    def __init__(self, data: dict[tuple[str, str], pd.DataFrame]):
        self.data = data
        self.calls: list[tuple[str, str, str, str | None]] = []

    def get(self, dataset, data_id=None, start_date="2020-01-01", end_date=None):
        self.calls.append((dataset, data_id, start_date, end_date))
        df = self.data.get((dataset, data_id), pd.DataFrame())
        if not df.empty and "date" in df.columns:
            mask = df["date"] >= start_date
            if end_date:
                mask &= df["date"] <= end_date
            df = df[mask]
        return df.reset_index(drop=True)

    def requested(self, dataset):
        return [c for c in self.calls if c[0] == dataset]


def _daily_rows(days, **extra):
    return pd.DataFrame([{"date": d, "stock_id": "2330", **extra} for d in days])


def _prices(days, close=100.0):
    return pd.DataFrame([{"date": d, "stock_id": "2330", "Trading_Volume": 1000, "Trading_money": 100_000,
                          "open": close, "max": close, "min": close, "close": close, "spread": 0.0,
                          "Trading_turnover": 10} for d in days])


@pytest.fixture
def client():
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-01-05", "2026-02-27")]
    data = {("TaiwanStockInfo", "2330"): pd.DataFrame([{"stock_id": "2330", "stock_name": "台積電", "type": "twse"}]),
            ("TaiwanStockInfo", "6488"): pd.DataFrame([{"stock_id": "6488", "stock_name": "環球晶", "type": "tpex"}]),
            ("TaiwanStockPrice", "2330"): _prices(days),
            ("TaiwanStockPrice", "TAIEX"): _prices(days)}
    for ds in DAILY_DATASETS:
        if ds != "TaiwanStockPrice":
            data[(ds, "2330")] = _daily_rows(days, name="Foreign_Investor", value=1)
    return FakeClient(data)


def test_first_sync_downloads_full_history(client, tmp_path):
    res = MarketDataSync(client, Cache(tmp_path)).sync_stock("2330")
    for ds in DAILY_DATASETS:
        assert client.requested(ds)[0][2] == "2020-01-01"
    assert {c[0] for c in client.calls} == {"TaiwanStockInfo", *DAILY_DATASETS}
    assert res.price_rows == res.added_price_rows == 40
    assert (res.price_start, res.price_end) == (date(2026, 1, 5), date(2026, 2, 27))


def test_incremental_sync_overlaps_and_skips_fresh_info(client, tmp_path):
    sync = MarketDataSync(client, Cache(tmp_path))
    sync.sync_stock("2330")
    client.data[("TaiwanStockPrice", "2330")] = _prices(
        [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-01-05", "2026-03-04")])
    client.calls.clear()

    res = sync.sync_stock("2330")
    assert client.requested("TaiwanStockPrice")[0][2] == "2026-02-20"  # 最後一筆 2/27 往回 7 天
    assert not client.requested("TaiwanStockInfo")
    assert res.added_price_rows == 3
    cached = Cache(tmp_path).read("TaiwanStockPrice", "2330")
    assert cached["date"].is_unique and len(cached) == 43


def test_start_redownloads_range_and_keeps_other_rows(client, tmp_path):
    sync = MarketDataSync(client, Cache(tmp_path))
    sync.sync_stock("2330")
    revised = client.data[("TaiwanStockPrice", "2330")].copy()
    revised.loc[revised["date"] == "2026-02-02", "close"] = 123.0
    client.data[("TaiwanStockPrice", "2330")] = revised

    sync.sync_stock("2330", start=date(2026, 2, 1), end=date(2026, 2, 10))
    assert client.requested("TaiwanStockPrice")[-1][2:] == ("2026-02-01", "2026-02-10")
    cached = Cache(tmp_path).read("TaiwanStockPrice", "2330").set_index("date")
    assert len(cached) == 40
    assert cached.loc["2026-02-02", "close"] == 123.0
    assert cached.index[0] == "2026-01-05"


def test_stale_info_is_refreshed(client, tmp_path):
    sync = MarketDataSync(client, Cache(tmp_path), info_max_age_hours=0)
    sync.sync_stock("2330")
    sync.sync_stock("2330")
    assert len(client.requested("TaiwanStockInfo")) == 2


def test_non_listed_stock_raises_and_writes_nothing(client, tmp_path):
    """只處理上市股票：上櫃股票連基本資料都不寫入，也不下載其他資料集。"""
    sync = MarketDataSync(client, Cache(tmp_path))
    with pytest.raises(UnsupportedMarketError, match="6488 環球晶是上櫃股票"):
        sync.sync_stock("6488")
    with pytest.raises(UnsupportedMarketError):
        sync.sync_info("6488")
    assert not list(tmp_path.rglob("6488.parquet"))
    assert [c[0] for c in client.calls] == ["TaiwanStockInfo", "TaiwanStockInfo"]


def test_cached_non_listed_info_is_rejected(client, tmp_path):
    """本機已經有上櫃股票的基本資料（例如之前下載過）時，也不會下載。"""
    cache = Cache(tmp_path)
    cache.write("TaiwanStockInfo", "6488", client.data[("TaiwanStockInfo", "6488")])
    with pytest.raises(UnsupportedMarketError):
        MarketDataSync(client, cache).sync_stock("6488")
    assert not client.calls


def test_unknown_stock_raises_and_writes_nothing(client, tmp_path):
    with pytest.raises(UnknownStockError, match="9999"):
        MarketDataSync(client, Cache(tmp_path)).sync_stock("9999")
    assert not list(tmp_path.rglob("9999.parquet"))


def test_watchlist_add_remove(tmp_path):
    with Watchlist(tmp_path / "t.db") as wl:
        assert wl.add("2330", "台積電")
        assert not wl.add("2330", "台積電")
        wl.add("2317", "鴻海")
        assert wl.stock_ids() == ["2317", "2330"]
        assert wl.remove("2317")
        assert not wl.remove("2317")
        assert [(i.stock_id, i.name) for i in wl.items()] == [("2330", "台積電")]


def _service(client, tmp_path, sync=None):
    sync = sync or MarketDataSync(client, Cache(tmp_path))
    return MarketDataService(sync=sync, loader=DataLoader(Cache(tmp_path)), watchlist=Watchlist(tmp_path / "t.db"))


def test_add_to_watchlist_validates_codes(client, tmp_path):
    svc = _service(client, tmp_path)
    changes = svc.add_to_watchlist(["2330", "9999", "2330", "6488"])
    assert [(c.stock_id, c.status, c.name) for c in changes] == [("2330", "added", "台積電"), ("9999", "not_found", ""),
                                                                 ("6488", "error", "")]
    assert "只處理上市股票" in changes[-1].message
    assert svc.watchlist.stock_ids() == ["2330"]
    assert svc.add_to_watchlist(["2330"])[0].status == "exists"
    assert [c.status for c in svc.remove_from_watchlist(["2330", "2317"])] == ["removed", "absent"]


class FakeSync:
    def __init__(self, errors: dict[str, Exception]):
        self.errors = errors
        self.calls: list[str] = []
        self.benchmark_calls = 0

    def sync_stock(self, stock_id, start=None, end=None, on_dataset=None):
        self.calls.append(stock_id)
        if stock_id in self.errors:
            raise self.errors[stock_id]
        return StockSyncResult(stock_id, 1, 1, None, None)

    def sync_benchmark(self):
        self.benchmark_calls += 1
        return 10


def test_fetch_continues_after_failure_and_stops_on_rate_limit(client, tmp_path):
    sync = FakeSync({"B": UnknownStockError("查無股票代號 B"), "C": FinMindRateLimitError("額度用完")})
    report = _service(client, tmp_path, sync).fetch(["A", "B", "C", "D"])
    assert [r.stock_id for r in report.results] == ["A"]
    assert set(report.failed) == {"B", "C"}
    assert report.skipped == ["D"]
    assert sync.benchmark_calls == 0 and not report.ok


def test_update_fetches_watchlist_and_benchmark(client, tmp_path):
    sync = FakeSync({})
    svc = _service(client, tmp_path, sync)
    svc.watchlist.add("2330")
    svc.watchlist.add("2317")
    report = svc.update()
    assert sync.calls == ["2317", "2330"]
    assert report.benchmark_rows == 10 and report.ok
