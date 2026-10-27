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

    def __post_init__(self) -> None:
        # 和網頁 API 的輸入範圍相同；超出範圍的成本會算出沒有意義的報酬（例如負的證交稅）
        if not 0 <= self.commission < 0.1:
            raise ValueError("手續費率要在 0 到 0.1 之間")
        if not 0 < self.discount <= 1:
            raise ValueError("手續費折扣要大於 0、最多 1（例如 0.28 代表 2.8 折）")
        if not 0 <= self.tax < 0.1:
            raise ValueError("證交稅要在 0 到 0.1 之間（例如股票 0.003、ETF 0.001）")
        if not 0 <= self.slippage < 0.1:
            raise ValueError("單邊滑價要在 0 到 0.1 之間（例如 0.001 代表 0.1%）")

    @property
    def buy_rate(self) -> float:
        return self.commission * self.discount + self.slippage

    @property
    def sell_rate(self) -> float:
        return self.commission * self.discount + self.tax + self.slippage

    @classmethod
    def zero(cls) -> CostModel:
        return cls(commission=0.0, discount=1.0, tax=0.0, slippage=0.0)
