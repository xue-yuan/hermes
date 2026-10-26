"""升降單位和漲跌停（證交所的價格規則）。

hermes verify 用來檢查每天的價格都落在漲跌停範圍內、對齊升降單位；回測用來標出漲停買進、跌停賣出的交易。
"""

from __future__ import annotations

# 升降單位，以「分」為單位避免浮點誤差：(價格未滿, 跳動單位)
_STOCK_TICKS = (
    (1_000, 1),
    (5_000, 5),
    (10_000, 10),
    (50_000, 50),
    (100_000, 100),
    (None, 500),
)
_ETF_TICKS = ((5_000, 1), (None, 5))


def tick_size(price_cents: int, etf: bool = False) -> int:
    """股票：未滿 10 元 0.01、10～50 元 0.05、50～100 元 0.1、100～500 元 0.5、500～1000 元 1、1000 元以上 5。
    ETF：未滿 50 元 0.01、50 元以上 0.05。"""
    for bound, tick in _ETF_TICKS if etf else _STOCK_TICKS:
        if bound is None or price_cents < bound:
            return tick
    raise AssertionError("unreachable")


def candidate_closes(reference: float, etf: bool = False) -> list[float]:
    """參考價（前一日收盤）的漲跌停範圍內，所有可能的收盤價（由低到高）。

    漲停價 = 參考價 × 1.1 向下取到升降單位；跌停價 = 參考價 × 0.9 向上取到升降單位。
    """
    ref = round(reference * 100)
    up_raw = ref * 11 // 10
    up_tick = tick_size(up_raw, etf)
    up = up_raw // up_tick * up_tick
    down_raw = -(-ref * 9 // 10)
    down_tick = tick_size(down_raw, etf)
    down = -(-down_raw // down_tick) * down_tick
    prices, p = [], down
    while p <= up:
        prices.append(p / 100)
        p += tick_size(p, etf)
    return prices
