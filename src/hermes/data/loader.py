"""高階資料讀取：還原權值、籌碼、本益比。

只讀本機快取，不連網；下載請用 hermes.data.sync.MarketDataSync。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.config import HISTORY_START, home_dir
from hermes.data.errors import MissingDataError
from hermes.data.store import Cache

# 漲跌幅限制，以及判斷「跳空超過限制」時容許的誤差（漲跌停價會向下取到升降單位）
PRICE_LIMIT = 0.10
LIMIT_TOLERANCE = 0.005

# 官方的公司行動紀錄：(資料集, 名稱, 前一日收盤欄位, 參考價欄位, 漲停價欄位, 跌停價欄位, 權／息欄位)
# 權／息欄位：除權息紀錄標示「權」「息」「權息」，有「權」代表股數可能改變（配股或現金增資，要看證交所的明細）；
# 減資、分割沒有這欄（None），一定會改變股數
CORPORATE_ACTIONS = (
    (
        "TaiwanStockDividendResult",
        "除權息",
        "before_price",
        "after_price",
        "max_price",
        "min_price",
        "stock_or_cache_dividend",
    ),
    (
        "TaiwanStockCapitalReductionReferencePrice",
        "減資",
        "ClosingPriceonTheLastTradingDay",
        "PostReductionReferencePrice",
        "LimitUp",
        "LimitDown",
        None,
    ),
    (
        "TaiwanStockSplitPrice",
        "分割／面額變更",
        "before_price",
        "after_price",
        "max_price",
        "min_price",
        None,
    ),
)

# 證交所的股數比例（hermes.data.sync 下載）：date、source、ratio、cash、rights_ratio、rights_price，
# 欄位意思見 hermes.data.twse
SHARE_RATIO_DATASET = "TwseShareRatio"
SHARE_RATIO_COLUMNS = ["ratio", "cash", "rights_ratio", "rights_price"]

PRICE_COLUMNS = {
    "max": "high",
    "min": "low",
    "Trading_Volume": "volume",
    "Trading_money": "value",
    "Trading_turnover": "trades",
}


def _float_column(df: pd.DataFrame, name: str) -> pd.Series | float:
    """數值欄位；資料集沒有這欄時是 NaN。"""
    return df[name].astype(float) if name in df else np.nan


class DataLoader:
    def __init__(self, cache: Cache | None = None):
        self.cache = cache or Cache(home_dir())

    # ------------------------------------------------------------------ raw
    def read(self, dataset: str, data_id: str) -> pd.DataFrame:
        """讀取本機快取的原始資料集；沒下載過就丟出 MissingDataError。"""
        cached = self.cache.read(dataset, data_id)
        if cached is None:
            raise MissingDataError(
                f"本機沒有 {data_id} 的資料（{dataset}），請先執行 hermes fetch {data_id}"
            )
        return cached

    # --------------------------------------------------------------- prices
    def prices(self, stock_id: str, adjust: bool = True) -> pd.DataFrame:
        """日 K 線，index 為日期；adjust=True 時依官方紀錄向前還原除權息、減資、分割。

        只使用官方紀錄，不做任何估算；沒有官方紀錄的跳空列在 unexplained_gaps()。
        沒有成交的日子（例如暫停交易，FinMind 記為全 0）不會出現。
        成交量照 CRSP 的做法，依股數比例還原：股數改變的事件（share_changes()）以前的成交量，乘上每股變成幾股，
        單位就和現在的股數相同。沒有官方股數比例的事件，以前的成交量是空值，不估算。成交金額、筆數不調整。
        欄位：open high low close volume value trades raw_close adj_factor
        """
        df = self._raw_prices(stock_id)
        df["raw_close"] = df["close"]
        factor = pd.Series(1.0, index=df.index)
        if adjust and stock_id.isdigit():
            for _, ev in self.adjustments(stock_id).iterrows():
                factor[df.index < ev["date"]] *= ev["ratio"]
            for col in ("open", "high", "low", "close"):
                df[col] = df[col] * factor
            for _, ev in self.share_changes(stock_id).iterrows():
                # 沒有比例（NaN）時，這次事件以前的成交量都是空值
                df.loc[df.index < ev["date"], "volume"] *= ev["ratio"]
        df["adj_factor"] = factor
        return df

    def _raw_prices(self, stock_id: str) -> pd.DataFrame:
        raw = self.read("TaiwanStockPrice", stock_id)
        if raw.empty:
            raise MissingDataError(
                f"本機沒有 {stock_id} 的股價資料，請確認代號或重新執行 hermes fetch {stock_id}"
            )
        df = raw.rename(columns=PRICE_COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df[
            (df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["open"] > 0)
        ]
        return df[["open", "high", "low", "close", "volume", "value", "trades"]].astype(
            float
        )

    def corporate_actions(self, stock_id: str) -> pd.DataFrame:
        """官方的除權息、減資、分割紀錄（事件發生後才有）。

        欄位：date、source、before（前一個交易日收盤）、reference（事件當天的參考價）、
        limit_up、limit_down（官方公布的當天漲跌停價，沒有時為 NaN）、shares_may_change（股數可能改變）、
        kind（除權、除息、減資；分割資料集照官方的類別：面額變更、分割、反分割）。
        還原比例 = reference / before。
        這些是補充資料，沒下載過的資料集視為沒有紀錄；hermes verify 會檢查是否都下載了。
        """
        frames = []
        for dataset, label, before, reference, up, down, rights in CORPORATE_ACTIONS:
            raw = self.cache.read(dataset, stock_id)
            if raw is None or raw.empty:
                continue
            # 只有標示「息」才確定股數不變；沒有這欄、或標示看不懂時，當成股數可能改變
            shares = (
                raw[rights].astype(str).str.strip() != "息" if rights in raw else True
            )
            if label == "除權息":
                kind = pd.Series(shares, index=raw.index).map({True: "除權", False: "除息"})
            elif "type" in raw:
                kind = raw["type"].astype(str).str.strip()
            else:
                kind = label
            frames.append(
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(raw["date"]),
                        "source": label,
                        "before": _float_column(raw, before),
                        "reference": _float_column(raw, reference),
                        "limit_up": _float_column(raw, up),
                        "limit_down": _float_column(raw, down),
                        "shares_may_change": shares,
                        "kind": kind,
                    }
                )
            )
        if not frames:
            return pd.DataFrame(
                columns=[
                    "date",
                    "source",
                    "before",
                    "reference",
                    "limit_up",
                    "limit_down",
                    "shares_may_change",
                    "kind",
                ]
            )
        out = pd.concat(frames, ignore_index=True)
        return (
            out[(out["before"] > 0) & (out["reference"] > 0)]
            .sort_values("date")
            .reset_index(drop=True)
        )

    def adjustments(self, stock_id: str) -> pd.DataFrame:
        """向前還原用到的官方事件：date、source、ratio（事件前的價格要乘上的比例）。"""
        actions = self.corporate_actions(stock_id)
        return actions.assign(ratio=actions["reference"] / actions["before"])[
            ["date", "source", "ratio"]
        ]

    def share_changes(self, stock_id: str) -> pd.DataFrame:
        """可能改變股數的官方事件，以及證交所公布的股數比例。

        欄位：date、source（除權、減資、面額變更、分割…）、ratio（每股變成幾股；沒有官方比例時是 NaN）、
        cash、rights_ratio、rights_price（依官方公式算回參考價用，見 hermes.data.twse）、
        before、reference（官方的前一日收盤、參考價）。
        除權的 ratio 只算無償配股，現金增資照 CRSP 不調整成交量，所以只有現金增資時是 1。
        比例由 hermes fetch、update 向證交所下載；ETF 的分割、反分割證交所沒有公布比例，一直是 NaN。
        """
        actions = self.corporate_actions(stock_id)
        actions = actions[actions["shares_may_change"].astype(bool)]
        events = pd.DataFrame(
            {
                "date": pd.to_datetime(actions["date"]).astype("datetime64[ns]").to_numpy(),
                "source": actions["kind"].astype(str).to_numpy(),
                "before": actions["before"].astype(float).to_numpy(),
                "reference": actions["reference"].astype(float).to_numpy(),
            }
        )
        ratios = self.cache.read(SHARE_RATIO_DATASET, stock_id)
        if ratios is None or ratios.empty:
            ratios = pd.DataFrame(columns=["date", "source", *SHARE_RATIO_COLUMNS])
        ratios = ratios.assign(
            date=pd.to_datetime(ratios["date"]).astype("datetime64[ns]"),
            source=ratios["source"].astype(str),
        ).drop_duplicates(["date", "source"], keep="last")
        out = events.merge(ratios[["date", "source", *SHARE_RATIO_COLUMNS]], how="left")
        out[SHARE_RATIO_COLUMNS] = out[SHARE_RATIO_COLUMNS].astype(float)
        return out[["date", "source", *SHARE_RATIO_COLUMNS, "before", "reference"]]

    def unexplained_gaps(self, stock_id: str) -> pd.DataFrame:
        """收盤跳空超過漲跌幅限制，卻沒有官方除權息、減資、分割紀錄的日子。

        漲跌幅以前一天收盤為基準最多 10%，超過只可能是參考價被調整過；沒有官方紀錄就是資料有問題，
        這裡不估算還原比例，只列出來（hermes verify 視為錯誤，分析指令會顯示警告）。
        新上市股票的前 5 個交易日沒有漲跌幅限制，不列入。
        欄位：date、prev_close、close、gap。
        """
        df = self._raw_prices(stock_id)
        prev_close = df["close"].shift(1)
        gap = df["close"] / prev_close - 1.0
        suspicious = (gap.abs() > PRICE_LIMIT + LIMIT_TOLERANCE) & prev_close.notna()
        if df.index[0] > pd.Timestamp(HISTORY_START) + pd.Timedelta(days=30):
            suspicious.iloc[:5] = False  # 資料開頭不是 HISTORY_START，代表是之後才上市
        suspicious &= ~df.index.isin(self.corporate_actions(stock_id)["date"])
        hits = df.index[suspicious.to_numpy()]
        return pd.DataFrame(
            {
                "date": hits,
                "prev_close": prev_close[hits].to_numpy(),
                "close": df.loc[hits, "close"].to_numpy(),
                "gap": gap[hits].to_numpy(),
            }
        )

    def benchmark(self) -> pd.DataFrame:
        """加權指數（TAIEX）日 K。"""
        return self.prices("TAIEX", adjust=False)

    def info(self, stock_id: str) -> dict:
        df = self.read("TaiwanStockInfo", stock_id)
        if df.empty:
            return {
                "stock_id": stock_id,
                "stock_name": stock_id,
                "industry": "",
                "market": "",
            }
        row = df.iloc[0]
        industries = [i for i in df["industry_category"].unique() if i]
        return {
            "stock_id": stock_id,
            "stock_name": row["stock_name"],
            "industry": "、".join(industries),
            "market": row["type"],
        }

    def valuation(self, stock_id: str) -> pd.DataFrame:
        """每日本益比、股價淨值比、殖利率（證交所公布值）。

        本益比、淨值比是 0 的代表官方寫「-」（例如虧損），換成空值；殖利率 0 是真的沒有配息，保留。
        """
        raw = self.read("TaiwanStockPER", stock_id)
        if raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = (
            df.set_index("date")
            .sort_index()[["PER", "PBR", "dividend_yield"]]
            .astype(float)
        )
        df[["PER", "PBR"]] = df[["PER", "PBR"]].replace(0.0, np.nan)
        return df

    # ----------------------------------------------------------------- 籌碼
    def institutional(self, stock_id: str) -> pd.DataFrame:
        """三大法人每日買賣超（股）：foreign、trust、dealer、total。"""
        raw = self.read("TaiwanStockInstitutionalInvestorsBuySell", stock_id)
        if raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["net"] = df["buy"].astype(float) - df["sell"].astype(float)
        wide = df.pivot_table(
            index="date", columns="name", values="net", aggfunc="sum"
        ).fillna(0.0)
        group = {
            "foreign": ["Foreign_Investor", "Foreign_Dealer_Self"],
            "trust": ["Investment_Trust"],
            "dealer": ["Dealer", "Dealer_self", "Dealer_Hedging"],
        }
        out = pd.DataFrame(index=wide.index)
        for key, names in group.items():
            cols = [c for c in names if c in wide.columns]
            out[key] = wide[cols].sum(axis=1) if cols else 0.0
        out["total"] = out.sum(axis=1)
        return out.sort_index()

    def margin(self, stock_id: str) -> pd.DataFrame:
        """融資融券餘額（張）。"""
        raw = self.read("TaiwanStockMarginPurchaseShortSale", stock_id)
        if raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        out = pd.DataFrame(index=df.index)
        out["margin_balance"] = df["MarginPurchaseTodayBalance"].astype(float)
        out["short_balance"] = df["ShortSaleTodayBalance"].astype(float)
        out["margin_limit"] = df["MarginPurchaseLimit"].astype(float)
        out["short_margin_ratio"] = out["short_balance"] / out[
            "margin_balance"
        ].replace(0, np.nan)
        return out

    def shareholding(self, stock_id: str) -> pd.DataFrame:
        """外資持股比例與發行股數。"""
        raw = self.read("TaiwanStockShareholding", stock_id)
        if raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return pd.DataFrame(
            {
                "foreign_ratio": df["ForeignInvestmentSharesRatio"].astype(float),
                "shares_issued": df["NumberOfSharesIssued"].astype(float),
            }
        )
