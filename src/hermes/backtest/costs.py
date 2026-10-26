"""台股交易成本模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """手續費 0.1425%（買賣皆收，可設定券商折扣），證交稅 0.3%（賣出收，ETF 為 0.1%），都是官方費率。

    滑價預設 0：回測用開盤價或收盤價成交，台股開盤、收盤都是集合競價，成交就是那個價格。
    漲跌停排不到的情況另外標 ⚠。
    """

    commission: float = 0.001425
    discount: float = 1.0
    tax: float = 0.003
    slippage: float = 0.0

    @property
    def buy_rate(self) -> float:
        return self.commission * self.discount + self.slippage

    @property
    def sell_rate(self) -> float:
        return self.commission * self.discount + self.tax + self.slippage

    @classmethod
    def zero(cls) -> CostModel:
        return cls(commission=0.0, discount=1.0, tax=0.0, slippage=0.0)
