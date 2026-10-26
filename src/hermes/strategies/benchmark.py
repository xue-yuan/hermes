"""比較基準：買進持有。沒有參數。"""

from __future__ import annotations

import pandas as pd

from hermes.strategies.base import StrategyContext, register


@register("buy_hold", "買進持有", "基準", "第一天買進後一路持有，作為比較基準。")
def buy_hold(ctx: StrategyContext) -> pd.Series:
    return pd.Series(1.0, index=ctx.prices.index)
