"""策略條件可以直接使用的估值數值：每個交易日一個值，名稱就是設定檔裡寫的名稱。

證交所每天收盤後用當天收盤價計算公布，所以要等收盤後才知道當天的值。
"""

from __future__ import annotations

import pandas as pd

# 名稱 →（說明, 單位, 來源資料）；來源資料對應 DataLoader 的同名方法
FIELDS = {
    "per": ("本益比（虧損時沒有值）", "倍", "valuation"),
    "pbr": ("股價淨值比", "倍", "valuation"),
    "dividend_yield": ("殖利率", "%", "valuation"),
}


def daily_fields(valuation: pd.DataFrame | None = None) -> pd.DataFrame:
    """DataLoader.valuation 轉成每日欄位。某一天沒有紀錄時就是空值，不補值。"""
    if valuation is None or valuation.empty:
        return pd.DataFrame()
    return valuation.rename(columns={"PER": "per", "PBR": "pbr"})[list(FIELDS)]
