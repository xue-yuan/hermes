"""籌碼面摘要（hermes chips）：讀本機的三大法人、外資持股、融資融券，N 日由使用者指定。"""

from __future__ import annotations

from hermes.chips import ChipReport, chip_summary
from hermes.data import DataLoader


def chip_report(
    stock_id: str,
    windows: tuple[int, ...] | list[int] = (),
    recent_days: int = 10,
    loader: DataLoader | None = None,
) -> ChipReport:
    """天數以交易日計算；windows 是要看的「近 N 日合計」「和 N 日前相比」。"""
    if any(w <= 0 for w in windows):
        raise ValueError("N 日要是正整數")
    if recent_days <= 0:
        raise ValueError("明細天數要是正整數")
    loader = loader or DataLoader()
    return chip_summary(
        loader.prices(stock_id, adjust=False).index,
        loader.institutional(stock_id),
        loader.shareholding(stock_id),
        loader.margin(stock_id),
        tuple(windows),
        recent_days,
    )
