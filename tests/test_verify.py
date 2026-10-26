from datetime import date

import pandas as pd
import pytest

from hermes.data import Cache, DataLoader
from hermes.data.twse import (
    parse_foreign_holding,
    parse_institutional,
    parse_margin,
    parse_stock_day,
    parse_valuation,
)
from hermes.services.verify import check_adjustment, check_chips, check_official, check_tick_grid, check_valuation

TWSE_PAYLOAD = {
    "stat": "OK",
    "fields": ["日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差", "成交筆數", "註記"],
    "data": [
        ["115/01/02", "39,836,425", "62,469,965,712", "1,555.00", "1,585.00", "1,545.00", "1,585.00", "+35.00",
         "77,972", ""],
        ["115/01/05", "0", "0", "--", "--", "--", "--", " 0.00", "0", ""],
    ],
}


def test_parse_twse_stock_day():
    df = parse_stock_day(TWSE_PAYLOAD)
    assert df.to_dict("records") == [{"date": "2026-01-02", "open": 1555.0, "max": 1585.0, "min": 1545.0,
                                      "close": 1585.0, "Trading_Volume": 39836425,
                                      "Trading_money": 62469965712, "Trading_turnover": 77972}]
    assert parse_stock_day({"stat": "很抱歉，沒有符合條件的資料!"}).empty


def _row(d, o, h, lo, c, sid="9999"):
    return {"date": d, "stock_id": sid, "Trading_Volume": 1000, "Trading_money": 1000 * c, "open": o, "max": h,
            "min": lo, "close": c, "spread": 0.0, "Trading_turnover": 10}


def _loader(tmp_path, rows, dividends=(), reductions=(), splits=()):
    cache = Cache(tmp_path)
    cache.write("TaiwanStockPrice", "9999", pd.DataFrame(rows))
    cache.write("TaiwanStockDividendResult", "9999", pd.DataFrame(list(dividends)))
    cache.write("TaiwanStockCapitalReductionReferencePrice", "9999", pd.DataFrame(list(reductions)))
    cache.write("TaiwanStockSplitPrice", "9999", pd.DataFrame(list(splits)))
    return DataLoader(cache)


def _days(n, start="2026-01-05"):
    return list(pd.bdate_range(start, periods=n).strftime("%Y-%m-%d"))


def test_capital_reduction_and_split_adjustment(tmp_path):
    d = _days(8)
    rows = [_row(d[0], 68, 68, 68, 68.1), _row(d[1], 68, 68.5, 68, 68.1),
            # 減資：退還現金，參考價 82.62（高於前收），恢復買賣當天收 76.2
            _row(d[2], 79.8, 79.8, 75.5, 76.2), _row(d[3], 76, 77, 75.5, 76.5),
            # 分割：1 股變 4 股，參考價 19.125
            _row(d[4], 19.1, 19.5, 19.0, 19.3), _row(d[5], 19.3, 19.4, 19.2, 19.35),
            _row(d[6], 19.35, 19.5, 19.3, 19.4), _row(d[7], 19.4, 19.6, 19.3, 19.5)]
    rows[3]["close"] = 76.5
    reductions = [{"date": d[2], "stock_id": "9999", "ClosingPriceonTheLastTradingDay": 68.1,
                   "PostReductionReferencePrice": 82.62, "LimitUp": 90.8, "LimitDown": 74.4}]
    splits = [{"date": d[4], "stock_id": "9999", "before_price": 76.5, "after_price": 19.125,
               "max_price": 21.0, "min_price": 17.25}]
    loader = _loader(tmp_path, rows, reductions=reductions, splits=splits)
    adj = loader.prices("9999")
    ret = adj["close"].pct_change()
    assert ret.iloc[2] == pytest.approx(76.2 / 82.62 - 1)
    assert ret.iloc[4] == pytest.approx(19.3 / 19.125 - 1)
    assert adj["adj_factor"].iloc[-1] == 1.0
    events = loader.adjustments("9999")
    assert list(events["source"]) == ["減資", "分割／面額變更"]
    assert events["ratio"].iloc[0] == pytest.approx(82.62 / 68.1)
    assert check_adjustment("9999", loader).status == "ok"


def test_tick_grid_check(tmp_path):
    d = _days(4, "2026-03-02")
    rows = [_row(d[0], 100, 100, 100, 100), _row(d[1], 100, 105, 99, 104.5),
            _row(d[2], 0, 0, 0, 0),  # 暫停交易：FinMind 記為全 0，應該略過
            _row(d[3], 104.5, 115.0, 104, 114.9)]  # 114.9 不在 100 元以上每檔 0.5 的格點上
    reductions = [{"date": d[1], "stock_id": "9999", "ClosingPriceonTheLastTradingDay": 100,
                   "PostReductionReferencePrice": 100, "LimitUp": 110.0, "LimitDown": 91.0}]
    check = check_tick_grid("9999", _loader(tmp_path, rows, reductions=reductions))
    assert check.status == "fail"
    assert any("114.9" in p for p in check.problems)
    assert any("官方 110／91" in p for p in check.problems)  # 程式算 110／90，和故意寫錯的官方跌停價不同
    assert not any(d[2] in p for p in check.problems)


def test_adjustment_check_requires_all_datasets(tmp_path):
    cache = Cache(tmp_path)
    cache.write("TaiwanStockPrice", "9999", pd.DataFrame([_row("2026-01-05", 10, 10, 10, 10)]))
    cache.write("TaiwanStockDividendResult", "9999", pd.DataFrame())
    check = check_adjustment("9999", DataLoader(cache))
    assert check.status == "fail" and "減資" in check.summary


class FakeTwse:
    def __init__(self, df=None, valuation=None, institutional=None, margin=None, holding=None):
        self.df, self.valuation_df = df, valuation
        self.daily = {"institutional": institutional, "margin": margin, "foreign_holding": holding}

    def stock_day(self, stock_id, month: date):
        return self.df[self.df["date"].str[:7] == month.strftime("%Y-%m")]

    def valuation(self, stock_id, month: date):
        return self.valuation_df[self.valuation_df["date"].str[:7] == month.strftime("%Y-%m")]

    def institutional(self, day: date):
        return self.daily["institutional"][day]

    def margin(self, day: date):
        return self.daily["margin"][day]

    def foreign_holding(self, day: date):
        return self.daily["foreign_holding"][day]


def test_official_check_finds_differences():
    days = _days(3, "2026-03-02")
    finmind = pd.DataFrame([_row(days[0], 10, 10, 10, 10), _row(days[1], 10, 11, 10, 11)])
    official = pd.DataFrame([{k: v for k, v in _row(days[0], 10, 10, 10, 10).items() if k not in ("stock_id", "spread")},
                             {k: v for k, v in _row(days[1], 10, 11, 10, 10.5).items()
                              if k not in ("stock_id", "spread")},
                             {k: v for k, v in _row(days[2], 10, 10, 10, 10).items() if k not in ("stock_id", "spread")}])
    check = check_official("9999", finmind, "twse", 1, FakeTwse(official))
    assert check.status == "fail"
    assert any(days[1] in p and "close" in p for p in check.problems)
    assert any(days[2] in p and "FinMind 沒有" in p for p in check.problems)
    assert check_official("9999", finmind, "tpex", 1, FakeTwse(official)).status == "skip"


def test_official_check_accepts_no_trade_day():
    """FinMind 把沒有成交的日子記成全 0、官方沒有這天：兩邊一致（例如 2317 的 2025-07-30）。
    同樣全 0、官方卻有成交，就是 FinMind 缺資料。"""
    days = _days(3, "2026-03-02")
    official_row = lambda d: {k: v for k, v in _row(d, 10, 10, 10, 10).items() if k not in ("stock_id", "spread")}  # noqa: E731
    finmind = pd.DataFrame([_row(days[0], 10, 10, 10, 10), _row(days[1], 0, 0, 0, 0), _row(days[2], 10, 10, 10, 10)])
    finmind.loc[1, ["Trading_Volume", "Trading_money", "Trading_turnover"]] = 0
    check = check_official("9999", finmind, "twse", 1, FakeTwse(pd.DataFrame([official_row(days[0]),
                                                                              official_row(days[2])])))
    assert check.status == "ok", check.problems
    assert f"{days[1]} 兩邊都沒有成交" in check.summary
    check = check_official("9999", finmind, "twse", 1, FakeTwse(pd.DataFrame([official_row(d) for d in days])))
    assert check.status == "fail" and any(days[1] in p for p in check.problems)


def test_unexplained_gap_fails_adjustment_check(tmp_path):
    d = _days(8)
    rows = [_row(day, 100, 100, 100, 100) for day in d[:6]]
    rows += [_row(d[6], 120, 120, 120, 120),  # +20%，沒有任何官方紀錄
             _row(d[7], 120, 121, 119, 120)]
    check = check_adjustment("9999", _loader(tmp_path, rows))
    assert check.status == "fail"
    assert any(d[6] in p and "沒有官方" in p for p in check.problems)


def test_new_listing_first_five_days_have_no_limit(tmp_path):
    d = _days(6)  # 資料從 2026 年才開始，代表是新上市股票
    rows = [_row(d[0], 100, 100, 100, 100), _row(d[1], 130, 130, 130, 130)]
    rows += [_row(day, 130, 130, 130, 130) for day in d[2:]]
    assert check_adjustment("9999", _loader(tmp_path, rows)).status == "ok"


# ---------------------------------------------------------------------------
# 籌碼、本益比：證交所的格式（取自 2026-10-22 的實際回應，只留 2330）

T86_FIELDS = ["證券代號", "證券名稱", "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)",
              "外陸資買賣超股數(不含外資自營商)", "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數",
              "投信買進股數", "投信賣出股數", "投信買賣超股數", "自營商買賣超股數", "自營商買進股數(自行買賣)",
              "自營商賣出股數(自行買賣)", "自營商買賣超股數(自行買賣)", "自營商買進股數(避險)", "自營商賣出股數(避險)",
              "自營商買賣超股數(避險)", "三大法人買賣超股數"]
T86_2330 = ["2330", "台積電          ", "12,403,967", "25,098,593", "-12,694,626", "0", "0", "0", "1,370,000", "20,049",
            "1,349,951", "1,195,500", "190,854", "479,840", "-288,986", "1,592,141", "107,655", "1,484,486",
            "-10,149,175"]
MARGIN_FIELDS = ["代號", "名稱", "買進", "賣出", "現金償還", "前日餘額", "今日餘額", "次一營業日限額", "買進", "賣出",
                 "現券償還", "前日餘額", "今日餘額", "次一營業日限額", "資券互抵", "註記"]
MARGIN_2330 = ["2330", "台積電", "597", "434", "21", "35,330", "35,472", "6,483,092", "1", "10", "0", "6", "15",
               "6,483,092", "0", " "]
QFIIS_FIELDS = ["證券代號", "證券名稱", "國際證券編碼", "發行股數", "外資及陸資尚可投資股數", "全體外資及陸資持有股數",
                "外資及陸資尚可投資比率", "全體外資及陸資持股比率", "外資及陸資共用法令投資上限比率", "陸資法令投資上限比率",
                "與前日異動原因(註)", "最近一次上市公司申報外資及陸資持股異動日期"]
QFIIS_2330 = ["2330", "台積電", "TW0002330008", "25,932,370,067", "8,147,108,615", "17,785,261,452", 31.41, 68.58,
              "100.00", "100.00", "", "115/05/26"]


def test_parse_twse_chip_tables():
    inst = parse_institutional({"stat": "OK", "fields": T86_FIELDS, "data": [T86_2330]}).loc["2330"]
    assert inst["Foreign_Investor_buy"] == 12403967 and inst["Foreign_Investor_sell"] == 25098593
    assert inst["Dealer_Hedging_buy"] == 1592141 and inst["total"] == -10149175
    margin = parse_margin({"stat": "OK", "tables": [
        {"title": "115年10月22日 信用交易統計", "fields": ["項目"], "data": []},
        {"title": "115年10月22日 融資融券彙總 (全部)", "fields": MARGIN_FIELDS, "data": [MARGIN_2330]}]}).loc["2330"]
    assert margin["MarginPurchaseTodayBalance"] == 35472 and margin["ShortSaleSell"] == 10
    assert margin["OffsetLoanAndShort"] == 0
    holding = parse_foreign_holding({"stat": "OK", "fields": QFIIS_FIELDS, "data": [QFIIS_2330]}).loc["2330"]
    assert holding["ForeignInvestmentSharesRatio"] == 68.58 and holding["NumberOfSharesIssued"] == 25932370067
    val = parse_valuation({"stat": "OK", "fields": ["日期", "殖利率(%)", "股利年度", "本益比", "股價淨值比", "財報年/季"],
                           "data": [["115年10月01日", "0.88", 114, "29.09", "10.12", "115/2"],
                                    ["115年10月02日", "0.00", 114, "-", "1.20", "115/2"]]})
    assert val.iloc[0].to_dict() == {"date": "2026-10-01", "dividend_yield": 0.88, "PER": 29.09, "PBR": 10.12}
    assert val.at[1, "PBR"] == 1.2 and pd.isna(val.at[1, "PER"])  # 官方寫「-」：沒有本益比（例如虧損）
    for parse in (parse_institutional, parse_margin, parse_foreign_holding, parse_valuation):
        assert parse({"stat": "很抱歉，沒有符合條件的資料!"}).empty


def test_parser_rejects_unexpected_layout():
    from hermes.data import DataError

    with pytest.raises(DataError, match="缺少：投信買進股數"):
        parse_institutional({"stat": "OK", "fields": [f for f in T86_FIELDS if f != "投信買進股數"], "data": []})
    with pytest.raises(DataError, match="融資融券彙總"):
        parse_margin({"stat": "OK", "tables": [{"title": "融資融券彙總", "fields": MARGIN_FIELDS[:-1], "data": []}]})


def _chip_loader(tmp_path, days, trust_sell=20049, margin_balance=35472):
    cache = Cache(tmp_path)
    cache.write("TaiwanStockPrice", "2330", pd.DataFrame([_row(d, 10, 10, 10, 10, "2330") for d in days]))
    names = {"Foreign_Investor": (12403967, 25098593), "Foreign_Dealer_Self": (0, 0),
             "Investment_Trust": (1370000, trust_sell), "Dealer_self": (190854, 479840),
             "Dealer_Hedging": (1592141, 107655)}
    cache.write("TaiwanStockInstitutionalInvestorsBuySell", "2330", pd.DataFrame([
        {"date": d, "stock_id": "2330", "name": n, "buy": b, "sell": s} for d in days for n, (b, s) in names.items()]))
    margin_values = dict(zip(MARGIN_COLUMNS_FINMIND, [597, 434, 21, 35330, margin_balance, 6483092, 1, 10, 0, 6, 15,
                                                      6483092, 0]))
    cache.write("TaiwanStockMarginPurchaseShortSale", "2330",
                pd.DataFrame([{"date": d, "stock_id": "2330", **margin_values} for d in days]))
    cache.write("TaiwanStockShareholding", "2330", pd.DataFrame([
        {"date": d, "stock_id": "2330", "NumberOfSharesIssued": 25932370067,
         "ForeignInvestmentRemainingShares": 8147108615, "ForeignInvestmentShares": 17785261452,
         "ForeignInvestmentRemainRatio": 31.41, "ForeignInvestmentSharesRatio": 68.58} for d in days]))
    cache.write("TaiwanStockPER", "2330", pd.DataFrame([
        {"date": d, "stock_id": "2330", "dividend_yield": 0.88, "PER": 29.09, "PBR": 10.12} for d in days]))
    return DataLoader(cache)


MARGIN_COLUMNS_FINMIND = ["MarginPurchaseBuy", "MarginPurchaseSell", "MarginPurchaseCashRepayment",
                          "MarginPurchaseYesterdayBalance", "MarginPurchaseTodayBalance", "MarginPurchaseLimit",
                          "ShortSaleBuy", "ShortSaleSell", "ShortSaleCashRepayment", "ShortSaleYesterdayBalance",
                          "ShortSaleTodayBalance", "ShortSaleLimit", "OffsetLoanAndShort"]


def _official_chips(days):
    inst = parse_institutional({"stat": "OK", "fields": T86_FIELDS, "data": [T86_2330]})
    margin = parse_margin({"stat": "OK", "tables": [{"title": "融資融券彙總 (全部)", "fields": MARGIN_FIELDS,
                                                     "data": [MARGIN_2330]}]})
    holding = parse_foreign_holding({"stat": "OK", "fields": QFIIS_FIELDS, "data": [QFIIS_2330]})
    as_day = lambda df: {date.fromisoformat(d): df for d in days}  # noqa: E731
    return FakeTwse(institutional=as_day(inst), margin=as_day(margin), holding=as_day(holding),
                    valuation=pd.DataFrame([{"date": d, "dividend_yield": 0.88, "PER": 29.09, "PBR": 10.12}
                                            for d in days]))


def test_chip_and_valuation_checks_pass_when_identical(tmp_path):
    days = _days(3, "2026-10-20")
    loader, twse = _chip_loader(tmp_path, days), _official_chips(days)
    assert check_chips("2330", loader, "twse", 3, twse).status == "ok"
    assert check_valuation("2330", loader, "twse", 1, twse).status == "ok"
    assert check_chips("2330", loader, "tpex", 3, twse).status == "skip"


def test_chip_check_finds_differences(tmp_path):
    days = _days(3, "2026-10-20")
    check = check_chips("2330", _chip_loader(tmp_path, days, trust_sell=20000, margin_balance=35470), "twse", 2,
                        _official_chips(days))
    assert check.status == "fail"
    assert any("Investment_Trust sell" in p for p in check.problems)
    assert any("三大法人合計買賣超" in p for p in check.problems)
    assert any("融資融券 MarginPurchaseTodayBalance" in p for p in check.problems)
    assert all(days[0] not in p for p in check.problems)  # 只比最近 2 天


def test_stock_missing_from_official_institutional_table_means_no_trades(tmp_path):
    days = _days(1, "2026-10-22")
    loader = _chip_loader(tmp_path, days)
    twse = _official_chips(days)
    twse.daily["institutional"] = {date.fromisoformat(days[0]): parse_institutional(
        {"stat": "OK", "fields": T86_FIELDS, "data": []})}
    check = check_chips("2330", loader, "twse", 1, twse)
    assert check.status == "fail"  # FinMind 有買賣，官方卻說沒有
    assert any("Foreign_Investor buy：FinMind 12403967，官方 0" in p for p in check.problems)
