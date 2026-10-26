"""指標組的數值（hermes tech）：讀 indicators/ 裡的指標組，算出某一天每個指標的數值。

用實際成交價算（還原模式用還原價格）。只算數值，不判斷買賣。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from hermes.config import indicators_dir
from hermes.data import DataError, DataLoader
from hermes.services.signals import adjusted_volume_note
from hermes.strategies import (
    IndicatorSet,
    indicator_frame,
    load_indicator_set,
    ready_frame,
)
from hermes.strategies import (
    DATA_FIELDS,
    RuleStrategy,
    operand_label,
    operand_text,
    ref_parts,
)
from hermes.strategies.rules import IndicatorConfig
from hermes.technical import CATALOG

# 價格欄位的白話名稱；條件裡的成交量單位是股
PRICE_LABELS = {
    "open": "開盤價",
    "high": "最高價",
    "low": "最低價",
    "close": "收盤價",
    "volume": "成交量（股）",
}

SUFFIXES = (".yaml", ".yml")
COMMON = "common"  # 常用版：indicators/common.yaml


class IndicatorSetNotFoundError(LookupError):
    pass


def indicator_set_files(directory: Path | None = None) -> list[Path]:
    directory = directory or indicators_dir()
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix in SUFFIXES)


def resolve_indicator_set(ref: str, directory: Path | None = None) -> Path:
    """ref 可以是 indicators/ 裡的檔名（不含副檔名），也可以是設定檔路徑。"""
    path = Path(ref).expanduser()
    if path.suffix in SUFFIXES and path.is_file():
        return path
    directory = directory or indicators_dir()
    for suffix in SUFFIXES:
        candidate = directory / f"{ref}{suffix}"
        if candidate.is_file():
            return candidate
    names = (
        "、".join(p.stem for p in indicator_set_files(directory)) or "沒有任何指標組"
    )
    raise IndicatorSetNotFoundError(f"找不到指標組 {ref}（{directory} 裡有：{names}）")


def indicator_label(cfg: IndicatorConfig) -> str:
    """指標的白話名稱和參數，例如「CCI 順勢指標（20 日）」。"""
    spec = CATALOG[cfg.type]
    return spec.label(spec.resolve_params(cfg.params))


def column_labels(indicators: dict[str, IndicatorConfig]) -> dict[str, str]:
    """每個指標欄位的白話名稱：{"ma20": "簡單移動平均（20 日）", "kd.k": "KD 隨機指標（…）：K 值"}。"""
    out = {}
    for name, cfg in indicators.items():
        spec = CATALOG[cfg.type]
        label = indicator_label(cfg)
        if spec.outputs:
            out.update({f"{name}.{line}": f"{label}：{spec.line_label(line)}" for line in spec.outputs})
        else:
            out[name] = label
    return out


def operand_labels(strategy: RuleStrategy) -> dict[str, str]:
    """策略條件裡每個名稱的白話寫法（價格、籌碼和本益比欄位、指標），給網頁顯示條件用。

    條件用到的前 N 天的值也有，例如 {"ma20（前1天）": "前 1 天的 簡單移動平均（20 日）"}。
    """
    data = {name: desc for name, (desc, _, _) in DATA_FIELDS.items()}
    names = {**PRICE_LABELS, **data, **column_labels(strategy.indicators)}
    previous = {
        operand_label(o): operand_text(o, names)
        for cond in strategy.conditions()
        for o in (cond.left, cond.right)
        if ref_parts(o)[1]
    }
    return {**names, **previous}


@dataclass
class IndicatorSnapshot:
    stock_id: str
    indicator_set: IndicatorSet
    date: pd.Timestamp
    adjusted: bool  # True：還原模式，用還原價格算
    close: float
    values: pd.Series  # index：指標欄位（名稱，多線指標為「名稱.線」）
    ready: pd.Series  # 和 values 同樣的 index：是否已經過了暖機（型態類指標沒有值也算，見 ready_frame）
    # 實際成交價算得出來、還原價格算不出來的指標（缺官方股數比例的成交量）；只有還原模式會有
    unavailable: pd.Series
    warnings: list[str] = field(default_factory=list)


def indicator_snapshot(
    stock_id: str,
    ref: str = COMMON,
    day: date | str | None = None,
    loader: DataLoader | None = None,
    directory: Path | None = None,
    adjusted: bool = False,
) -> IndicatorSnapshot:
    """指標用全部歷史計算（暖機天數才夠），再取出指定那天的值；day 預設為最新一天。

    預設用實際成交價算；adjusted=True 是還原模式，用還原價格算。
    """
    indicator_set = load_indicator_set(resolve_indicator_set(ref, directory))
    loader = loader or DataLoader()
    prices = loader.prices(stock_id, adjust=adjusted)
    when = prices.index[-1] if day is None else pd.Timestamp(day)
    if when not in prices.index:
        raise DataError(
            f"本機沒有 {stock_id} 在 {when:%Y-%m-%d} 的交易資料"
            f"（資料期間 {prices.index[0]:%Y-%m-%d} ～ {prices.index[-1]:%Y-%m-%d}）"
        )
    frame = indicator_frame(indicator_set.indicators, prices)
    ready = ready_frame(indicator_set.indicators, frame).loc[when]
    unavailable = pd.Series(False, index=ready.index)
    if adjusted and prices["volume"].isna().any():
        raw = indicator_frame(indicator_set.indicators, loader.prices(stock_id, adjust=False))
        unavailable = ready_frame(indicator_set.indicators, raw).loc[when] & ~ready
    snap = IndicatorSnapshot(
        stock_id,
        indicator_set,
        when,
        adjusted,
        float(prices.at[when, "close"]),
        frame.loc[when],
        ready,
        unavailable,
    )
    if unavailable.any():
        snap.warnings.append(
            f"{adjusted_volume_note(loader, stock_id)}；用到成交量的指標無法計算"
        )
    return snap
