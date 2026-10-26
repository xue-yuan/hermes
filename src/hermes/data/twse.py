"""證交所官方資料。

驗證 FinMind 的資料（hermes verify），不寫入快取：
- 個股日成交資訊、本益比／淨值比／殖利率：一次回傳一檔股票一整個月。
- 三大法人買賣超、融資融券、外資持股：一次回傳一天的全部股票，同一次執行裡會重複使用。
欄位都轉成和 FinMind 相同的名稱，方便直接比對。

股數比例：除權息、減資、變更面額的明細（每股配幾股、每千股換發幾股），FinMind 沒有。
由 hermes.data.sync 下載後存進本機，還原成交量時使用。

證交所會封鎖短時間內大量請求的 IP，所以每次請求至少間隔 min_interval 秒。
"""

from __future__ import annotations

import re
import time
from datetime import date

import pandas as pd
import requests

from hermes.data.errors import DataError

BASE_URL = "https://www.twse.com.tw/rwd/zh"

# 三大法人買賣超日報（T86）：FinMind 的 name → 證交所的（買進、賣出）欄位
INSTITUTIONAL_FIELDS = {
    "Foreign_Investor": (
        "外陸資買進股數(不含外資自營商)",
        "外陸資賣出股數(不含外資自營商)",
    ),
    "Foreign_Dealer_Self": ("外資自營商買進股數", "外資自營商賣出股數"),
    "Investment_Trust": ("投信買進股數", "投信賣出股數"),
    "Dealer_self": ("自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)"),
    "Dealer_Hedging": ("自營商買進股數(避險)", "自營商賣出股數(避險)"),
}
INSTITUTIONAL_TOTAL = "三大法人買賣超股數"

# 融資融券彙總（MI_MARGN）：欄位名稱有重複（融資、融券都有「買進」），所以照位置對應
MARGIN_LAYOUT = [
    "代號",
    "名稱",
    "買進",
    "賣出",
    "現金償還",
    "前日餘額",
    "今日餘額",
    "次一營業日限額",
    "買進",
    "賣出",
    "現券償還",
    "前日餘額",
    "今日餘額",
    "次一營業日限額",
    "資券互抵",
    "註記",
]
MARGIN_COLUMNS = [
    "MarginPurchaseBuy",
    "MarginPurchaseSell",
    "MarginPurchaseCashRepayment",
    "MarginPurchaseYesterdayBalance",
    "MarginPurchaseTodayBalance",
    "MarginPurchaseLimit",
    "ShortSaleBuy",
    "ShortSaleSell",
    "ShortSaleCashRepayment",
    "ShortSaleYesterdayBalance",
    "ShortSaleTodayBalance",
    "ShortSaleLimit",
    "OffsetLoanAndShort",
]

# 外資及陸資投資持股統計（MI_QFIIS）：證交所欄位 → FinMind 欄位
FOREIGN_HOLDING_FIELDS = {
    "發行股數": "NumberOfSharesIssued",
    "外資及陸資尚可投資股數": "ForeignInvestmentRemainingShares",
    "全體外資及陸資持有股數": "ForeignInvestmentShares",
    "外資及陸資尚可投資比率": "ForeignInvestmentRemainRatio",
    "全體外資及陸資持股比率": "ForeignInvestmentSharesRatio",
}

# 個股日本益比、殖利率及股價淨值比（BWIBBU）：證交所欄位 → FinMind 欄位
VALUATION_FIELDS = {"殖利率(%)": "dividend_yield", "本益比": "PER", "股價淨值比": "PBR"}


def _number(text: str | float) -> float | None:
    if isinstance(text, (int, float)):
        return float(text)
    text = text.replace(",", "").strip()
    return (
        None
        if not text or text.startswith("-") and not text[1:2].isdigit()
        else float(text)
    )


def _roc_date(text: str) -> str:
    """民國日期（115/01/02 或 115年01月02日）轉成 2026-01-02。"""
    parts = (
        text.replace("年", "/").replace("月", "/").replace("日", "").strip().split("/")
    )
    y, m, d = (int(x) for x in parts)
    return date(y + 1911, m, d).isoformat()


def _require(fields: list[str], needed, what: str) -> dict[str, int]:
    col = {name: i for i, name in enumerate(fields)}
    missing = [f for f in needed if f not in col]
    if missing:
        raise DataError(f"證交所{what}的欄位和預期不同，缺少：{'、'.join(missing)}")
    return col


def parse_stock_day(payload: dict) -> pd.DataFrame:
    """轉成和 FinMind 相同的欄位：date、open、max、min、close、Trading_Volume、Trading_money、Trading_turnover。"""
    columns = [
        "date",
        "open",
        "max",
        "min",
        "close",
        "Trading_Volume",
        "Trading_money",
        "Trading_turnover",
    ]
    if payload.get("stat") != "OK":
        return pd.DataFrame(columns=columns)
    col = {name: i for i, name in enumerate(payload["fields"])}
    rows = []
    for r in payload.get("data", []):
        prices = [_number(r[col[k]]) for k in ("開盤價", "最高價", "最低價", "收盤價")]
        if None in prices:  # 當天沒有成交
            continue
        rows.append(
            [
                _roc_date(r[col["日期"]]),
                *prices,
                int(_number(r[col["成交股數"]]) or 0),
                int(_number(r[col["成交金額"]]) or 0),
                int(_number(r[col["成交筆數"]]) or 0),
            ]
        )
    return pd.DataFrame(rows, columns=columns)


def parse_institutional(payload: dict) -> pd.DataFrame:
    """三大法人買賣超日報。index：股票代號；欄位：<FinMind name>_buy、<FinMind name>_sell、total（合計買賣超）。

    單位都是股。官方沒列出的股票，代表當天三大法人都沒有買賣。
    """
    columns = [
        f"{n}_{side}" for n in INSTITUTIONAL_FIELDS for side in ("buy", "sell")
    ] + ["total"]
    if payload.get("stat") != "OK":
        return pd.DataFrame(columns=columns)
    needed = [
        "證券代號",
        INSTITUTIONAL_TOTAL,
        *(f for pair in INSTITUTIONAL_FIELDS.values() for f in pair),
    ]
    col = _require(payload["fields"], needed, "三大法人買賣超日報")
    rows = {}
    for r in payload.get("data", []):
        values = [
            _number(r[col[f]]) for pair in INSTITUTIONAL_FIELDS.values() for f in pair
        ]
        rows[r[col["證券代號"]].strip()] = [
            *values,
            _number(r[col[INSTITUTIONAL_TOTAL]]),
        ]
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


def parse_margin(payload: dict) -> pd.DataFrame:
    """融資融券彙總。index：股票代號；欄位和 FinMind 相同（融資、融券的單位都是張）。"""
    if payload.get("stat") != "OK":
        return pd.DataFrame(columns=MARGIN_COLUMNS)
    table = next(
        (
            t
            for t in payload.get("tables", [])
            if "融資融券彙總" in (t.get("title") or "")
        ),
        None,
    )
    if table is None or table.get("fields") != MARGIN_LAYOUT:
        raise DataError("證交所融資融券彙總的欄位和預期不同")
    rows = {r[0].strip(): [_number(x) for x in r[2:15]] for r in table.get("data", [])}
    return pd.DataFrame.from_dict(rows, orient="index", columns=MARGIN_COLUMNS)


def parse_foreign_holding(payload: dict) -> pd.DataFrame:
    """外資及陸資持股統計。index：股票代號；欄位和 FinMind 相同。"""
    columns = list(FOREIGN_HOLDING_FIELDS.values())
    if payload.get("stat") != "OK":
        return pd.DataFrame(columns=columns)
    col = _require(
        payload["fields"], ["證券代號", *FOREIGN_HOLDING_FIELDS], "外資持股統計"
    )
    rows = {
        r[col["證券代號"]].strip(): [_number(r[col[f]]) for f in FOREIGN_HOLDING_FIELDS]
        for r in payload.get("data", [])
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


def parse_valuation(payload: dict) -> pd.DataFrame:
    """個股日本益比、殖利率及股價淨值比（一個月）。欄位：date、dividend_yield、PER、PBR；官方寫「-」的是 NaN。"""
    columns = ["date", *VALUATION_FIELDS.values()]
    if payload.get("stat") != "OK":
        return pd.DataFrame(columns=columns)
    col = _require(
        payload["fields"], ["日期", *VALUATION_FIELDS], "本益比、殖利率及股價淨值比"
    )
    rows = [
        [_roc_date(r[col["日期"]]), *(_number(r[col[f]]) for f in VALUATION_FIELDS)]
        for r in payload.get("data", [])
    ]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# 股數比例：每股變成幾股（ratio），以及依官方公式算回參考價要用的數字
#   參考價 = (前一日收盤 − cash + rights_price × rights_ratio) / (ratio + rights_ratio)
# 除權息（TWT49U 的公式）：(除權息前收盤價 − 息值 + 現金增資認購價 × 現金增資配股率) / (1 + 無償配股率 + 現金增資配股率)
# 減資（TWTAUU 的公式）：(停止買賣前收盤價 − 息值 − 每股退還股款) / 減資換股率；彌補虧損沒有息值和退還股款
# 變更面額（TWTB8U 的公式）：停止買賣前收盤價 / 變更股票面額換股率

RIGHTS_FIELDS = {
    "cash": "(每股配發現金股利)除息",
    "stock": "A. 按普通股股東持股比例每千股無償配股",
    "rights_total": "C. (有償) 現金增資",
    "rights_holders": "c. 原股東認購",
    "rights_price": "每股認購金額",
    "rights": "按股東持股比例每千股認購",
}
REDUCTION_FIELDS = {
    "stop": "停止買賣日期",
    "new": "每壹仟股換發新股票",
    "refund": "每股退還股款",
    "cash": "原股每股配發現金股利",
}
PAR_CHANGE_FIELDS = {"stop": "停止買賣日期", "new": "每股換發新股票"}


def _amount(text: str, what: str) -> float:
    """有單位的數字，例如「1.2 元／股」「18,000,000 股」「880.00000000 股」。"""
    m = re.search(r"-?[\d,]*\.?\d+", str(text))
    if m is None:
        raise DataError(f"證交所{what}的數字看不懂：{text!r}")
    return float(m.group().replace(",", ""))


def _detail(payload: dict, stock_id: str, fields: dict[str, str], what: str) -> dict | None:
    """明細表裡這檔股票的那一列，欄位換成 fields 的鍵；沒有這檔股票時回傳 None。

    欄位名稱前後的空白、結尾的全形冒號不算（證交所的明細表有時會加）。
    """
    if str(payload.get("stat", "")).lower() != "ok" or not payload.get("data"):
        return None
    names = [str(f).strip().removesuffix("：").strip() for f in payload["fields"]]
    col = _require(names, ["股票代號", *fields.values()], what)
    for r in payload["data"]:
        if str(r[col["股票代號"]]).strip() == stock_id:
            return {k: r[col[name]] for k, name in fields.items()}
    return None


def parse_rights_detail(payload: dict, stock_id: str) -> dict | None:
    """除權息明細（TWT49UDetail）。

    ratio 只算無償配股（盈餘、資本公積配股，每千股幾股）：照 CRSP，成交量只依股票股利和分割調整，現金增資不調整。
    明細的每千股無償配股只到小數第一位（例如國巨 2024 年實際是 194.8365 股，明細寫 194.8 股），
    所以 ratio 最多差 0.00005。
    rights_ratio 是參考價公式的現金增資配股率：現金增資的全部股數（含員工認購、公開承銷）÷ 參與除權的股數
    （TWT48U 的定義）。明細的「按股東持股比例每千股認購」只算原股東認購的部分，所以再乘上全部股數 ÷ 原股東認購股數。
    有現金增資、卻沒有原股東認購時算不出來，是 NaN（不影響 ratio）。
    """
    what = f" {stock_id} 除權息明細"
    row = _detail(payload, stock_id, RIGHTS_FIELDS, what)
    if row is None:
        return None
    v = {k: _amount(x, what) for k, x in row.items()}
    if v["rights_holders"] > 0:
        rights = v["rights"] / 1000 * v["rights_total"] / v["rights_holders"]
    else:
        rights = 0.0 if v["rights_total"] == 0 else float("nan")
    return {
        "ratio": 1 + v["stock"] / 1000,
        "cash": v["cash"],
        "rights_ratio": rights,
        "rights_price": v["rights_price"],
    }


def parse_reduction_list(payload: dict) -> dict[str, str]:
    """減資恢復買賣參考價格（TWTAUU）：{股票代號: 明細的檔案日期（YYYYMMDD）}。"""
    if str(payload.get("stat", "")).lower() != "ok":
        return {}
    col = _require(payload["fields"], ["股票代號", "詳細資料"], "減資恢復買賣參考價格")
    out = {}
    for r in payload.get("data", []):
        # 詳細資料寫成「3481  ,20240814」：股票代號、明細的檔案日期
        out[str(r[col["股票代號"]]).strip()] = str(r[col["詳細資料"]]).partition(",")[2].strip()
    return out


def parse_reduction_detail(payload: dict, stock_id: str) -> dict | None:
    """減資明細（TWTAVUDetail）。彌補虧損、退還股款都照 Yahoo 的做法當成反分割：每股變成「每千股換發新股 ÷ 1000」股。

    減資同時辦理的現金增資照 CRSP 不調整成交量，官方的恢復買賣參考價也不含它（另外公布除權參考價）。
    """
    what = f" {stock_id} 減資明細"
    row = _detail(payload, stock_id, REDUCTION_FIELDS, what)
    if row is None:
        return None
    return {
        "stop_date": _roc_date(row["stop"]),
        "ratio": _amount(row["new"], what) / 1000,
        "cash": _amount(row["cash"], what) + _amount(row["refund"], what),
        "rights_ratio": 0.0,
        "rights_price": 0.0,
    }


def parse_par_change_detail(payload: dict, stock_id: str) -> dict | None:
    """變更面額明細（TWTB7UDetail）：每股換發新股票幾股。"""
    what = f" {stock_id} 變更面額明細"
    row = _detail(payload, stock_id, PAR_CHANGE_FIELDS, what)
    if row is None:
        return None
    return {
        "stop_date": _roc_date(row["stop"]),
        "ratio": _amount(row["new"], what),
        "cash": 0.0,
        "rights_ratio": 0.0,
        "rights_price": 0.0,
    }


class TwseClient:
    def __init__(self, min_interval: float = 3.0, timeout: float = 20.0):
        self.min_interval = min_interval
        self.timeout = timeout
        self.session = requests.Session()
        self._last = float("-inf")
        self._daily: dict[
            tuple[str, date], pd.DataFrame
        ] = {}  # 一天全部股票的資料，多檔股票共用

    def _get(self, path: str, params: dict, what: str) -> dict:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        try:
            resp = self.session.get(
                f"{BASE_URL}/{path}",
                timeout=self.timeout,
                params={**params, "response": "json"},
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise DataError(f"證交所{what}下載失敗：{exc}") from exc

    def stock_day(self, stock_id: str, month: date) -> pd.DataFrame:
        payload = self._get(
            "afterTrading/STOCK_DAY",
            {"date": month.strftime("%Y%m01"), "stockNo": stock_id},
            f" {stock_id} {month:%Y-%m} 日成交資訊",
        )
        return parse_stock_day(payload)

    def valuation(self, stock_id: str, month: date) -> pd.DataFrame:
        payload = self._get(
            "afterTrading/BWIBBU",
            {"date": month.strftime("%Y%m01"), "stockNo": stock_id},
            f" {stock_id} {month:%Y-%m} 本益比、殖利率及股價淨值比",
        )
        return parse_valuation(payload)

    def _daily_table(self, kind: str, day: date) -> pd.DataFrame:
        key = (kind, day)
        if key not in self._daily:
            path, params, what, parse = {
                "institutional": (
                    "fund/T86",
                    {"selectType": "ALLBUT0999"},
                    "三大法人買賣超",
                    parse_institutional,
                ),
                "margin": (
                    "marginTrading/MI_MARGN",
                    {"selectType": "ALL"},
                    "融資融券",
                    parse_margin,
                ),
                "foreign_holding": (
                    "fund/MI_QFIIS",
                    {"selectType": "ALLBUT0999"},
                    "外資持股",
                    parse_foreign_holding,
                ),
            }[kind]
            payload = self._get(
                path, {**params, "date": day.strftime("%Y%m%d")}, f" {day} {what}"
            )
            self._daily[key] = parse(payload)
        return self._daily[key]

    def institutional(self, day: date) -> pd.DataFrame:
        return self._daily_table("institutional", day)

    def margin(self, day: date) -> pd.DataFrame:
        return self._daily_table("margin", day)

    def foreign_holding(self, day: date) -> pd.DataFrame:
        return self._daily_table("foreign_holding", day)

    # ------------------------------------------------------------- 股數比例
    def rights_detail(self, stock_id: str, day: date) -> dict | None:
        """day：除權息日。"""
        payload = self._get(
            "exRight/TWT49UDetail",
            {"STK_NO": stock_id, "T1": day.strftime("%Y%m%d")},
            f" {stock_id} {day} 除權息明細",
        )
        return parse_rights_detail(payload, stock_id)

    def reduction_detail(self, stock_id: str, day: date) -> dict | None:
        """day：減資恢復買賣日。先從當天的減資列表找到明細的檔案日期，再下載明細。"""
        listing = self._get(
            "reducation/TWTAUU",
            {"startDate": day.strftime("%Y%m%d"), "endDate": day.strftime("%Y%m%d")},
            f" {day} 減資恢復買賣參考價格",
        )
        file_date = parse_reduction_list(listing).get(stock_id)
        if not file_date:
            return None
        payload = self._get(
            "reducation/TWTAVUDetail",
            {"STK_NO": stock_id, "FILE_DATE": file_date},
            f" {stock_id} {day} 減資明細",
        )
        return parse_reduction_detail(payload, stock_id)

    def par_change_detail(self, stock_id: str, file_date: date) -> dict | None:
        """file_date：明細的檔案日期（停止買賣前最後一個交易日，見 hermes.data.sync）。"""
        payload = self._get(
            "change/TWTB7UDetail",
            {"STK_NO": stock_id, "FILE_DATE": file_date.strftime("%Y%m%d")},
            f" {stock_id} {file_date} 變更面額明細",
        )
        return parse_par_change_detail(payload, stock_id)
