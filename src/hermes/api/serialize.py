"""把 pandas、numpy 的值轉成 JSON：空值是 null，日期是 YYYY-MM-DD；只轉格式，不改數字。"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import date, datetime

import numpy as np
import pandas as pd


def value(v: object) -> object:
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return None
        if math.isinf(f):  # 例如沒有虧損時的獲利因子；JSON 沒有無限大
            return "Infinity" if f > 0 else "-Infinity"
        return f
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return v.strftime("%Y-%m-%d")
    return v


def values(seq: Iterable) -> list:
    return [value(v) for v in seq]


def days(index: Iterable) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in index]


def mapping(d: dict) -> dict:
    return {str(k): value(v) for k, v in d.items()}


def records(df: pd.DataFrame) -> list[dict]:
    return [mapping(row) for row in df.to_dict("records")]
