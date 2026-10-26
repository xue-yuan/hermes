"""成交量照 CRSP 還原：股數比例來自證交所的除權息、減資、變更面額明細。

- 除權：每股變成 1 + 每千股無償配股 ÷ 1000 股；現金增資照 CRSP 不調整成交量。
- 減資（彌補虧損、退還股款）：照 Yahoo 的做法當成反分割，每股變成「每千股換發新股 ÷ 1000」股。
- 變更面額：每股換發新股票幾股。
- ETF 分割、反分割：證交所沒有公布比例，以前的成交量是空值，不估算。
- 對照：還原成交量和「照定義逐日乘上之後每次事件的比例」相同；用比例照官方公式算回的參考價，要等於官方參考價。
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from conftest import make_prices
from hermes.data import Cache, DataError, DataLoader, MarketDataSync
from hermes.data.loader import SHARE_RATIO_DATASET
from hermes.data.twse import (
    parse_par_change_detail,
    parse_reduction_detail,
    parse_reduction_list,
    parse_rights_detail,
)
from hermes.ml.features import price_features
from hermes.services import indicator_snapshot, signal_history
from hermes.services.verify import check_share_changes

# ---------------------------------------------------------------------------
# 證交所的明細（2026-10-26 下載的實際回應，只留用到的部分）

RIGHTS_2884 = {  # 玉山金 2025-07-22 除權息
    "stat": "ok",
    "fields": ["股票代號", "股票名稱", "(每股配發現金股利)除息", "(增資配股) 除權",
               "A. 按普通股股東持股比例每千股無償配股", "B. 員工紅利轉增資", "C. (有償) 現金增資", "每股認購金額",
               "a. 公開承銷", "b. 員工認購", " c. 原股東認購", "按股東持股比例每千股認購"],
    "data": [["2884  ", "玉山金          ", "1.2 元／股", "", "10 股", "18,000,000 股", "0 股", "0 元／股", "0 股",
              "0 股", "0 股", "0.00000000 股"]],
}
REDUCTION_LIST_3481 = {  # 群創 2024-08-26 減資恢復買賣
    "stat": "OK",
    "fields": ["恢復買賣日期", "股票代號", "名稱", "停止買賣前收盤價格", "恢復買賣參考價", "漲停價格", "跌停價格",
               "開盤競價基準", "除權參考價", "減資原因", "詳細資料"],
    "data": [["113/08/26", "3481", "群創", "14.55", "15.17", "16.65", "13.70", "15.15", "--", "退還股款",
              "3481  ,20240814"]],
}
REDUCTION_3481 = {
    "stat": "OK",
    "fields": ["股票代號：", "股票名稱：", "停止買賣日期：", "每壹仟股換發新股票：", "每股退還股款：", "原股每股配發現金股利：",
               "減資並(有償)現金增資：", "每股認購金額：", "a. 公開承銷：", "b. 員工認購：", "c. 原股東認購：",
               "按股東持股比例每千股認購："],
    "data": [["3481  ", "群創", "113/08/15", "880.00000000 股", "1.200000 元/股", "0 元/股", "0 股", "0 元/股", "0 股",
              "0 股", "0 股", "0 股"]],
}
PAR_2327 = {  # 國巨 2025-08-25 變更面額恢復買賣
    "stat": "OK",
    "fields": ["股票代號", "股票名稱", "停止買賣日期", "每股換發新股票", "變更前面額", "變更後面額"],
    "data": [["2327", "國巨", "114/08/14", "4.00000000 股", "10.00 元/股", "2.50 元/股"]],
}


def test_parse_official_details():
    assert parse_rights_detail(RIGHTS_2884, "2884") == {
        "ratio": 1.01, "cash": 1.2, "rights_ratio": 0.0, "rights_price": 0.0}
    assert parse_rights_detail(RIGHTS_2884, "2330") is None
    assert parse_reduction_list(REDUCTION_LIST_3481) == {"3481": "20240814"}
    assert parse_reduction_detail(REDUCTION_3481, "3481") == {
        "stop_date": "2024-08-15", "ratio": 0.88, "cash": 1.2, "rights_ratio": 0.0, "rights_price": 0.0}
    assert parse_par_change_detail(PAR_2327, "2327") == {
        "stop_date": "2025-08-14", "ratio": 4.0, "cash": 0.0, "rights_ratio": 0.0, "rights_price": 0.0}
    # 查不到（證交所回「無相關資料」、空的 data）
    assert parse_reduction_detail({"stat": "無相關資料"}, "3481") is None
    assert parse_par_change_detail({**PAR_2327, "data": []}, "2327") is None


def test_parse_rejects_unexpected_layout():
    changed = {**RIGHTS_2884, "fields": [f.replace("無償配股", "配股") for f in RIGHTS_2884["fields"]]}
    with pytest.raises(DataError, match="欄位和預期不同"):
        parse_rights_detail(changed, "2884")
    garbled = {**PAR_2327, "data": [["2327", "國巨", "114/08/14", "四股", "", ""]]}
    with pytest.raises(DataError, match="看不懂"):
        parse_par_change_detail(garbled, "2327")


RIGHTS_2884_2023 = {  # 玉山金 2023-04-07 只有現金增資：總共 8 億股，原股東認購 6 億股（每千股 42.03122920 股）
    **RIGHTS_2884,
    "data": [["2884  ", "玉山金          ", "0 元／股", "", "0 股", "0 股", "800,000,000 股", "20 元／股",
              "80,000,000 股", "120,000,000 股", "600,000,000 股", "42.03122920 股"]],
}


def test_rights_with_cash_capital_increase():
    """有現金增資時，ratio 只算無償配股（照 CRSP，現金增資不調整成交量）。
    參考價公式的現金增資配股率是全部現金增資股數 ÷ 參與除權的股數，不只原股東認購的部分。"""
    out = parse_rights_detail(RIGHTS_2884_2023, "2884")
    assert out["ratio"] == 1.0 and out["cash"] == 0.0 and out["rights_price"] == 20.0
    assert out["rights_ratio"] == pytest.approx(0.0420312292 * 800 / 600, rel=1e-12)
    # 照公式算回的參考價 25.1608，官方公布 25.16；只用原股東認購的部分會算出 25.2302
    ref = (25.45 + 20 * out["rights_ratio"]) / (1 + out["rights_ratio"])
    assert ref == pytest.approx(25.1608, abs=1e-4)
    # 有現金增資、卻沒有原股東認購：算不出配股率，不影響 ratio
    row = list(RIGHTS_2884_2023["data"][0])
    row[10], row[11] = "0 股", "0.00000000 股"
    out = parse_rights_detail({**RIGHTS_2884, "data": [row]}, "2884")
    assert out["ratio"] == 1.0 and np.isnan(out["rights_ratio"])


# ---------------------------------------------------------------------------
# 讀取：還原成交量

N = 300


def _write(home, events, ratios=()):
    """events：（第幾天, 資料集代號, 類別, 前一天收盤, 參考價）；資料集代號 div、cut、split。
    ratios：（第幾天, 類別, ratio, cash, rights_ratio, rights_price）。"""
    p = make_prices(n=N, seed=3)
    days = pd.bdate_range("2024-01-01", periods=N)
    d = days.strftime("%Y-%m-%d")
    cache = Cache(home)
    cache.write("TaiwanStockPrice", "1234", pd.DataFrame([
        {"date": d[i], "stock_id": "1234", "Trading_Volume": r.volume, "Trading_money": r.close * r.volume,
         "open": r.open, "max": r.high, "min": r.low, "close": r.close, "spread": 0.0, "Trading_turnover": 10}
        for i, r in enumerate(p.itertuples())]))
    rows = {"div": [], "cut": [], "split": []}
    for i, ds, kind, before, ref in events:
        if ds == "div":
            rows[ds].append({"date": d[i], "stock_id": "1234", "before_price": before, "after_price": ref,
                             "stock_or_cache_dividend": kind})
        elif ds == "cut":
            rows[ds].append({"date": d[i], "stock_id": "1234", "ClosingPriceonTheLastTradingDay": before,
                             "PostReductionReferencePrice": ref})
        else:
            rows[ds].append({"date": d[i], "stock_id": "1234", "type": kind, "before_price": before,
                             "after_price": ref})
    cache.write("TaiwanStockDividendResult", "1234", pd.DataFrame(rows["div"]))
    cache.write("TaiwanStockCapitalReductionReferencePrice", "1234", pd.DataFrame(rows["cut"]))
    cache.write("TaiwanStockSplitPrice", "1234", pd.DataFrame(rows["split"]))
    cache.write("TaiwanStockInfo", "1234", pd.DataFrame([{"stock_id": "1234", "stock_name": "測試", "type": "twse",
                                                        "industry_category": ""}]))
    if ratios:
        cache.write(SHARE_RATIO_DATASET, "1234", pd.DataFrame([
            {"date": d[i], "source": src, "ratio": r, "cash": c, "rights_ratio": rr, "rights_price": rp}
            for i, src, r, c, rr, rp in ratios]))
    return DataLoader(cache), days


# 2884、3481、2327 的實際數字：前一天收盤、官方參考價、比例
EVENTS = [(60, "div", "權息", 34.20, 32.67), (120, "cut", None, 14.55, 15.17), (200, "split", "面額變更", 546.0, 136.5)]
RATIOS = [(60, "除權", 1.01, 1.2, 0.0, 0.0), (120, "減資", 0.88, 1.2, 0.0, 0.0), (200, "面額變更", 4.0, 0.0, 0.0, 0.0)]


def _expected_volume(raw: pd.DataFrame, days, ratios) -> pd.Series:
    """照定義逐日計算：每一天的成交量，乘上這天以後每次事件的比例。"""
    out = []
    for t, v in zip(raw.index, raw["volume"]):
        factor = 1.0
        for i, ratio in ratios:
            if t < days[i]:
                factor *= ratio
        out.append(v * factor)
    return pd.Series(out, index=raw.index)


def test_adjusted_volume_matches_definition(tmp_path):
    loader, days = _write(tmp_path, EVENTS, RATIOS)
    raw, adj = loader.prices("1234", adjust=False), loader.prices("1234", adjust=True)
    expected = _expected_volume(raw, days, [(i, r) for i, _, r, *_ in RATIOS])
    np.testing.assert_allclose(adj["volume"], expected, rtol=1e-12)
    # 例：第一次事件以前，成交量 × 1.01 × 0.88 × 4
    assert adj["volume"].iloc[0] == pytest.approx(raw["volume"].iloc[0] * 1.01 * 0.88 * 4, rel=1e-12)
    assert adj["volume"].iloc[200:].equals(raw["volume"].iloc[200:])
    # 成交金額、筆數不調整
    assert adj["value"].equals(raw["value"]) and adj["trades"].equals(raw["trades"])
    changes = loader.share_changes("1234")
    assert changes["source"].tolist() == ["除權", "減資", "面額變更"]
    assert changes["ratio"].tolist() == [1.01, 0.88, 4.0]


def test_missing_ratio_leaves_earlier_volume_empty(tmp_path):
    """中間的減資沒有比例：減資以前都是空值；減資以後照常乘上之後事件的比例。"""
    loader, days = _write(tmp_path, EVENTS, [RATIOS[0], RATIOS[2]])
    raw, adj = loader.prices("1234", adjust=False), loader.prices("1234", adjust=True)
    assert adj["volume"].iloc[:120].isna().all()
    np.testing.assert_allclose(adj["volume"].iloc[120:200], raw["volume"].iloc[120:200] * 4.0, rtol=1e-12)
    assert adj["volume"].iloc[200:].equals(raw["volume"].iloc[200:])


def test_etf_split_has_no_official_ratio(tmp_path):
    loader, days = _write(tmp_path, [(150, "split", "分割", 188.65, 47.16)])
    [ev] = loader.share_changes("1234").to_dict("records")
    assert ev["source"] == "分割" and np.isnan(ev["ratio"])
    assert loader.prices("1234", adjust=True)["volume"].iloc[:150].isna().all()


def test_cash_dividend_does_not_change_volume(tmp_path):
    loader, _ = _write(tmp_path, [(150, "div", "息", 100.0, 97.0)])
    assert loader.share_changes("1234").empty
    raw, adj = loader.prices("1234", adjust=False), loader.prices("1234", adjust=True)
    assert adj["volume"].equals(raw["volume"])


def test_volume_indicators_and_ml_work_with_ratios(tmp_path):
    """有股數比例時：還原模式下成交量類指標算得出來、沒有提醒，機器學習可以跑。"""
    loader, days = _write(tmp_path, EVENTS, RATIOS)
    sdir = tmp_path / "strategies"
    sdir.mkdir()
    (sdir / "obv.yaml").write_text("name: OBV 上升\nindicators:\n  obv: {type: obv}\n"
                                   "buy: {left: obv, op: gt, right: {ref: obv, ago: 1}}\n"
                                   "sell: {left: obv, op: lt, right: {ref: obv, ago: 1}}\n", encoding="utf-8")
    hist = signal_history("1234", "obv", adjust=True, loader=loader, directory=sdir)
    assert hist.warnings == []
    assert hist.result.values["obv"].iloc[1:].notna().all()
    idir = tmp_path / "indicators"
    idir.mkdir()
    (idir / "v.yaml").write_text("name: 量\nindicators:\n  obv: {type: obv}\n  vr: {type: volume_ratio, period: 20}\n",
                                 encoding="utf-8")
    snap = indicator_snapshot("1234", "v", None, loader, idir, adjusted=True)
    assert not snap.unavailable.any() and snap.warnings == []
    assert snap.values.notna().all()
    price_features(loader.prices("1234", adjust=True))  # 不報錯


# ---------------------------------------------------------------------------
# verify：用比例照官方公式算回參考價


def test_verify_reference_prices(tmp_path):
    loader, _ = _write(tmp_path / "ok", EVENTS, RATIOS)
    check = check_share_changes("1234", loader)
    assert check.status == "ok" and check.problems == []
    assert "3 次有證交所的股數比例，其中 3 次依官方公式算回的參考價都和官方相符" in check.summary

    # 比例錯了（例如把每千股 10 股當成每股 10 股）：算回的參考價對不上
    wrong = [(60, "除權", 11.0, 1.2, 0.0, 0.0), *RATIOS[1:]]
    loader, days = _write(tmp_path / "wrong", EVENTS, wrong)
    check = check_share_changes("1234", loader)
    assert check.status == "fail"
    assert check.problems == [f"{days[60]:%Y-%m-%d} 除權：每股變成 11 股，依官方公式算回的參考價 3.0000，官方參考價 32.67"]

    # 有一次沒有比例：提醒，不是錯誤
    loader, days = _write(tmp_path / "missing", EVENTS, RATIOS[:2])
    check = check_share_changes("1234", loader)
    assert check.status == "warn"
    assert check.problems == [f"{days[200]:%Y-%m-%d} 面額變更：沒有證交所的股數比例"]
    assert f"1 次沒有比例，還原價格在 {days[200]:%Y-%m-%d} 以前的成交量算不出來" in check.summary


def test_verify_rights_formula_with_cash_capital_increase(tmp_path):
    """除權息參考價 = (前收 − 息值 + 認購價 × 現增配股率) / (1 + 無償配股率 + 現增配股率)。
    2884 在 2023-04-07 的實際數字：前收 25.45，官方參考價 25.16。"""
    rights = parse_rights_detail(RIGHTS_2884_2023, "2884")["rights_ratio"]
    event = [(150, "div", "權", 25.45, 25.16)]
    loader, _ = _write(tmp_path, event, [(150, "除權", 1.0, 0.0, rights, 20.0)])
    assert check_share_changes("1234", loader).status == "ok"
    # 只用原股東認購的部分（每千股 42.03 股）：算回 25.2302，對不上
    loader, _ = _write(tmp_path / "holders", event, [(150, "除權", 1.0, 0.0, 0.0420312292, 20.0)])
    assert check_share_changes("1234", loader).status == "fail"
    # 算不出現金增資配股率：沒辦法對照，只提醒
    loader, days = _write(tmp_path / "unknown", event, [(150, "除權", 1.0, 0.0, np.nan, 20.0)])
    check = check_share_changes("1234", loader)
    assert check.status == "warn" and "沒辦法用公式對照參考價" in check.problems[0]
    assert loader.prices("1234", adjust=True)["volume"].notna().all()  # 股數比例照用


def test_verify_allows_published_precision(tmp_path):
    """國巨 2024-08-15：前收 745，官方參考價 623.51；明細寫每千股無償配股 194.8 股（實際 194.8365），
    用 1.1948 算回 623.5353。差距來自明細只到小數第一位，在容許範圍內。"""
    event = [(150, "div", "權", 745.0, 623.51)]
    loader, _ = _write(tmp_path, event, [(150, "除權", 1.1948, 0.0, 0.0, 0.0)])
    assert check_share_changes("1234", loader).status == "ok"
    # 明細寫 195.0 股（實際在 194.95～195.05 之間）時，算回的參考價最高 623.46，公布的位數解釋不了 623.51
    loader, _ = _write(tmp_path / "off", event, [(150, "除權", 1.195, 0.0, 0.0, 0.0)])
    assert check_share_changes("1234", loader).status == "fail"
    # 減資、變更面額的比例是完整的位數，只容許參考價的位數：群創 2023-08-28 算回 14.5789，官方 14.57
    loader, _ = _write(tmp_path / "cut", [(150, "cut", None, 14.35, 14.57)], [(150, "減資", 0.95, 0.5, 0.0, 0.0)])
    assert check_share_changes("1234", loader).status == "ok"
    loader, _ = _write(tmp_path / "cut_off", [(150, "cut", None, 14.35, 14.57)], [(150, "減資", 0.949, 0.5, 0.0, 0.0)])
    assert check_share_changes("1234", loader).status == "fail"


# ---------------------------------------------------------------------------
# 下載：只下載還沒有比例的事件


class FakeTwse:
    def __init__(self, fail: bool = False):
        self.calls: list[tuple[str, str, date]] = []
        self.fail = fail

    def _record(self, kind, stock_id, day):
        self.calls.append((kind, stock_id, day))
        if self.fail:
            raise DataError("證交所除權息明細下載失敗：系統忙碌中")

    def rights_detail(self, stock_id, day):
        self._record("rights", stock_id, day)
        return parse_rights_detail(RIGHTS_2884, "2884")

    def reduction_detail(self, stock_id, day):
        self._record("reduction", stock_id, day)
        return parse_reduction_detail(REDUCTION_3481, "3481")

    def par_change_detail(self, stock_id, file_date):
        self._record("par", stock_id, file_date)
        # 停止買賣日期是查詢日的下一個交易日
        return {**parse_par_change_detail(PAR_2327, "2327"), "stop_date": f"{file_date + pd.offsets.BDay(1):%Y-%m-%d}"}


def test_sync_downloads_missing_ratios_once(tmp_path):
    loader, days = _write(tmp_path, [*EVENTS, (250, "split", "分割", 188.65, 47.16)])
    twse = FakeTwse()
    sync = MarketDataSync(client=object(), cache=loader.cache, twse=twse)
    missing = sync.sync_share_ratios("1234")
    # 變更面額：用本機資料裡事件前最後一個交易日查明細
    assert twse.calls == [("rights", "1234", days[60].date()), ("reduction", "1234", days[120].date()),
                          ("par", "1234", days[199].date())]
    assert missing == [f"{days[250]:%Y-%m-%d} 分割：證交所沒有公布分割的股數比例"]  # ETF 分割不用查
    changes = loader.share_changes("1234")
    assert changes["ratio"].tolist()[:3] == [1.01, 0.88, 4.0] and np.isnan(changes["ratio"].iloc[3])
    assert check_share_changes("1234", loader).problems == [f"{days[250]:%Y-%m-%d} 分割：沒有證交所的股數比例"]

    twse.calls.clear()
    assert sync.sync_share_ratios("1234") == missing
    assert twse.calls == []  # 已經有的不再下載


def test_sync_retries_after_failure(tmp_path):
    loader, days = _write(tmp_path, EVENTS[:1])
    sync = MarketDataSync(client=object(), cache=loader.cache, twse=FakeTwse(fail=True))
    [msg] = sync.sync_share_ratios("1234")
    assert msg == f"{days[60]:%Y-%m-%d} 除權：證交所除權息明細下載失敗：系統忙碌中，下次執行會再試"
    assert loader.cache.read(SHARE_RATIO_DATASET, "1234") is None
    sync._twse = FakeTwse()
    assert sync.sync_share_ratios("1234") == []
    assert loader.share_changes("1234")["ratio"].tolist() == [1.01]


def test_sync_rejects_par_change_detail_of_another_event(tmp_path):
    """變更面額明細的停止買賣日期不在事件前最後一個交易日和恢復買賣日之間：不是這次事件，不用。"""
    loader, days = _write(tmp_path, EVENTS[2:])

    class Elsewhere(FakeTwse):
        def par_change_detail(self, stock_id, file_date):
            return parse_par_change_detail(PAR_2327, "2327")  # 停止買賣日期 2025-08-14，不在測試資料的期間

    sync = MarketDataSync(client=object(), cache=loader.cache, twse=Elsewhere())
    assert sync.sync_share_ratios("1234") == [f"{days[200]:%Y-%m-%d} 面額變更：證交所查不到這次的明細"]


def test_sync_without_share_changes_does_not_touch_twse(tmp_path):
    loader, _ = _write(tmp_path, [(150, "div", "息", 100.0, 97.0)])

    class Untouchable:
        def __getattr__(self, name):
            raise AssertionError("不應該連到證交所")

    assert MarketDataSync(client=object(), cache=loader.cache, twse=Untouchable()).sync_share_ratios("1234") == []
