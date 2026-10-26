"""策略條件可以直接使用的籌碼數值：每個交易日一個值，名稱就是設定檔裡寫的名稱。

都是證交所在當天收盤後才公布的資料，所以要等收盤後才知道當天的值。
"""

from __future__ import annotations

import pandas as pd

# 名稱 →（說明, 單位, 來源資料）；來源資料對應 DataLoader 的同名方法
FIELDS = {
    "foreign_net": ("外資買賣超（外陸資＋外資自營商）", "張", "institutional"),
    "trust_net": ("投信買賣超", "張", "institutional"),
    "dealer_net": ("自營商買賣超（自行買賣＋避險）", "張", "institutional"),
    "inst_net": ("三大法人合計買賣超", "張", "institutional"),
    "margin_balance": ("融資餘額", "張", "margin"),
    "short_balance": ("融券餘額", "張", "margin"),
    "foreign_ratio": ("外資持股比例", "%", "shareholding"),
}


def daily_fields(
    institutional: pd.DataFrame | None = None,
    margin: pd.DataFrame | None = None,
    shareholding: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """三大法人（股，DataLoader.institutional）、融資融券（張）、外資持股，轉成每日欄位；買賣超換算成張。

    沒傳入的資料，對應的欄位不會出現。某一天沒有紀錄時就是空值，不補值。
    """
    cols = {}
    if institutional is not None and not institutional.empty:
        for name, col in (
            ("foreign_net", "foreign"),
            ("trust_net", "trust"),
            ("dealer_net", "dealer"),
            ("inst_net", "total"),
        ):
            cols[name] = institutional[col] / 1000
    if margin is not None and not margin.empty:
        cols["margin_balance"] = margin["margin_balance"]
        cols["short_balance"] = margin["short_balance"]
    if shareholding is not None and not shareholding.empty:
        cols["foreign_ratio"] = shareholding["foreign_ratio"]
    return pd.DataFrame(cols)
