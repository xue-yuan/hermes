"""以 parquet 檔案快取 API 原始資料。只保留 HISTORY_START 之後的資料。"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from hermes.config import HISTORY_START

# 這些資料集的 date 欄位是資料更新日期，不是資料本身的日期，不依日期篩選
UNDATED_DATASETS = {"TaiwanStockInfo"}


def trim_history(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or dataset in UNDATED_DATASETS or "date" not in df.columns:
        return df
    return df[pd.to_datetime(df["date"]) >= pd.Timestamp(HISTORY_START)].reset_index(
        drop=True
    )


class Cache:
    def __init__(self, root: Path):
        self.root = Path(root) / "cache"

    def path(self, dataset: str, data_id: str) -> Path:
        return self.root / dataset / f"{data_id or '_all'}.parquet"

    def read(self, dataset: str, data_id: str) -> pd.DataFrame | None:
        path = self.path(dataset, data_id)
        if not path.exists():
            return None
        return trim_history(dataset, pd.read_parquet(path))

    def write(self, dataset: str, data_id: str, df: pd.DataFrame) -> None:
        path = self.path(dataset, data_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        trim_history(dataset, df).to_parquet(tmp, index=False)
        tmp.replace(path)

    def age_hours(self, dataset: str, data_id: str) -> float | None:
        path = self.path(dataset, data_id)
        if not path.exists():
            return None
        return (time.time() - path.stat().st_mtime) / 3600.0
