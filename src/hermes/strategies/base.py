"""策略註冊表。策略是一個函式：輸入 StrategyContext 與參數，輸出每日收盤後的目標部位權重。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class StrategyContext:
    prices: pd.DataFrame
    stock_id: str = ""
    loader: Any = None
    progress: Callable[[int, int], None] | None = None
    # 策略設定檔（strategies/*.yaml）每天收盤後的部位，meta-labeling 的主要策略用；由上層提供
    rule_position: Callable[[str], pd.Series] | None = None
    _extra: pd.DataFrame | None = field(default=None, repr=False)

    def exogenous(self) -> pd.DataFrame | None:
        """籌碼、估值、大盤等外生特徵（需要 loader）。"""
        if self.loader is None:
            return None
        if self._extra is None:
            from hermes.ml.features import exogenous_features

            self._extra = exogenous_features(self.loader, self.stock_id, self.prices)
        return self._extra


@dataclass
class StrategySpec:
    name: str
    title: str
    category: str
    description: str
    func: Callable[..., pd.Series]
    defaults: dict

    def run(self, ctx: StrategyContext, **params) -> pd.Series:
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"策略 {self.name} 沒有參數：{', '.join(sorted(unknown))}")
        merged = {**self.defaults, **params}
        return self.func(ctx, **merged).reindex(ctx.prices.index)


REGISTRY: dict[str, StrategySpec] = {}


def register(name: str, title: str, category: str, description: str, **defaults):
    def deco(func):
        REGISTRY[name] = StrategySpec(
            name, title, category, description, func, defaults
        )
        return func

    return deco


def get_strategy(name: str) -> StrategySpec:
    if name not in REGISTRY:
        raise KeyError(f"未知策略：{name}（可用：{', '.join(REGISTRY)}）")
    return REGISTRY[name]


def parse_params(spec: StrategySpec, items: list[str]) -> dict:
    """把 CLI 的 key=value 依預設值型別轉換。"""
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"參數格式應為 key=value：{item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in spec.defaults:
            raise ValueError(
                f"策略 {spec.name} 沒有參數 {key}（可用：{', '.join(spec.defaults)}）"
            )
        out[key] = coerce(raw.strip(), spec.defaults[key])
    return out


def coerce(raw: str, default):
    if isinstance(default, bool):
        if raw.lower() in ("1", "true", "yes", "y", "on"):
            return True
        if raw.lower() in ("0", "false", "no", "n", "off"):
            return False
        raise ValueError(f"無法解析布林值：{raw}")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if default is None:
        if raw.lower() in ("none", ""):
            return None
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


def stateful(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """進場條件成立後持有，直到出場條件成立（同日兩者皆成立時以出場為準）。"""
    pos = []
    holding = 0.0
    for en, ex in zip(entry.fillna(False).to_numpy(), exit_.fillna(False).to_numpy()):
        if holding and ex:
            holding = 0.0
        elif not holding and en and not ex:
            holding = 1.0
        pos.append(holding)
    return pd.Series(pos, index=entry.index)
