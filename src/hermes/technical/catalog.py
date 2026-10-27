"""指標目錄：策略設定檔裡的 type 對應到哪個指標函式、有哪些參數、輸出哪些線。

策略引擎（hermes.strategies）只透過這份目錄計算指標；新增指標時在 _SPECS 加一筆，
並在 tests/test_indicator_reference.py 加上獨立的對照。參數一律由使用者在設定檔指定，沒有預設值。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from hermes.technical import indicators as ind


@dataclass(frozen=True)
class Param:
    label: str
    kind: type = int
    # 白話寫法，{} 換成數值，例如「創 {} 日新高」；空的時候用 label：「天數」→「20 日」、「K 平滑」→「K 平滑 3」
    short: str = ""

    def text(self, value: float) -> str:
        v = f"{value:g}"
        if self.short:
            return self.short.format(v)
        if self.label.endswith("天數"):
            return f"{self.label[:-2].strip()} {v} 日".strip()
        return f"{self.label} {v}"


@dataclass(frozen=True)
class IndicatorSpec:
    type: str
    title: str
    params: dict[str, Param]
    compute: Callable[..., pd.Series | pd.DataFrame]  # compute(prices, **params)
    outputs: tuple[
        str, ...
    ] = ()  # 多條線的名稱，例如 KD 的 ("k", "d")；只有一條線時為空
    overlay: bool | tuple[str, ...] = (
        False  # 畫在價格圖上：單線指標用 True，多線指標列出要畫在價格圖上的線
    )
    # 型態類指標（例如 Darvas 箱）沒有值也是正常的結果（當時沒有箱子），不代表還在暖機；
    # 這裡給暖機天數 warmup(**params)：前幾天一定沒有值，之後有沒有值都算算得出來。其他指標是 None，有值才算。
    warmup: Callable[..., int] | None = None
    line_labels: tuple[str, ...] = ()  # outputs 每條線的白話名稱，順序和 outputs 相同

    def label(self, params: dict) -> str:
        """白話名稱，例如「簡單移動平均（200 日）」「乖離率（%，20 日）」；params 要先 resolve_params。"""
        text = "、".join(self.params[k].text(v) for k, v in params.items())
        if not text:
            return self.title
        if self.title.endswith("）"):
            return f"{self.title[:-1]}，{text}）"
        return f"{self.title}（{text}）"

    def line_label(self, line: str) -> str:
        return dict(zip(self.outputs, self.line_labels)).get(line, line)

    def on_price_chart(self, line: str | None = None) -> bool:
        if isinstance(self.overlay, tuple):
            return line in self.overlay
        return self.overlay

    def resolve_params(self, given: dict) -> dict:
        """檢查參數；每個參數都要指定，有問題時丟出 ValueError（訊息給使用者看）。"""
        unknown = sorted(set(given) - set(self.params))
        if unknown:
            usable = "、".join(self.params) or "無"
            raise ValueError(
                f"{self.type} 沒有參數 {'、'.join(unknown)}（可用參數：{usable}）"
            )
        out = {}
        for key, p in self.params.items():
            value = given.get(key)
            if value is None:
                raise ValueError(f"{self.type} 需要指定 {key}（{p.label}）")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{self.type} 的 {key} 要是數字")
            if p.kind is int and not float(value).is_integer():
                raise ValueError(f"{self.type} 的 {key} 要是整數")
            if value <= 0:
                raise ValueError(f"{self.type} 的 {key} 要大於 0")
            out[key] = p.kind(value)
        return out


def _period(label: str = "天數") -> dict[str, Param]:
    return {"period": Param(label)}


def _vol_params() -> dict[str, Param]:
    return {
        "period": Param("天數"),
        "year_days": Param("年化天數（一年幾個交易日）", short="一年 {} 個交易日"),
    }


_SPECS = [
    # 均線類
    IndicatorSpec(
        "sma",
        "簡單移動平均",
        _period(),
        lambda p, period: ind.sma(p["close"], period),
        overlay=True,
    ),
    IndicatorSpec(
        "ema",
        "指數移動平均",
        _period(),
        lambda p, period: ind.ema(p["close"], period),
        overlay=True,
    ),
    IndicatorSpec(
        "wma",
        "加權移動平均",
        _period(),
        lambda p, period: ind.wma(p["close"], period),
        overlay=True,
    ),
    IndicatorSpec(
        "bias", "乖離率（%）", _period(), lambda p, period: ind.bias(p["close"], period)
    ),
    # 動能、擺盪類
    IndicatorSpec(
        "rsi", "RSI", _period(), lambda p, period: ind.rsi(p["close"], period)
    ),
    IndicatorSpec(
        "kd",
        "KD 隨機指標",
        {
            "period": Param("RSV 天數"),
            "k_smooth": Param("K 平滑"),
            "d_smooth": Param("D 平滑"),
        },
        lambda p, period, k_smooth, d_smooth: ind.stochastic_kd(
            p, period, k_smooth, d_smooth
        ),
        outputs=("k", "d"),
        line_labels=("K 值", "D 值"),
    ),
    IndicatorSpec(
        "macd",
        "MACD",
        {
            "fast": Param("快線天數"),
            "slow": Param("慢線天數"),
            "signal": Param("訊號線天數"),
        },
        lambda p, fast, slow, signal: ind.macd(p["close"], fast, slow, signal),
        outputs=("macd", "signal", "hist"),
        line_labels=("MACD 線", "訊號線", "柱狀體"),
    ),
    IndicatorSpec(
        "cci", "CCI 順勢指標", _period(), lambda p, period: ind.cci(p, period)
    ),
    IndicatorSpec(
        "cci_rolling_dev",
        "CCI 順勢指標（逐日均線偏差）",
        _period(),
        lambda p, period: ind.cci_rolling_dev(p, period),
    ),
    IndicatorSpec(
        "williams_r",
        "威廉指標 %R",
        _period(),
        lambda p, period: ind.williams_r(p, period),
    ),
    IndicatorSpec(
        "roc",
        "變動率 ROC（%）",
        _period(),
        lambda p, period: ind.roc(p["close"], period),
    ),
    IndicatorSpec(
        "trix", "TRIX（%）", _period(), lambda p, period: ind.trix(p["close"], period)
    ),
    IndicatorSpec(
        "stoch_rsi",
        "Stoch RSI（0～1）",
        _period(),
        lambda p, period: ind.stoch_rsi(p["close"], period),
    ),
    IndicatorSpec(
        "psy", "心理線（%）", _period(), lambda p, period: ind.psy(p["close"], period)
    ),
    # 通道、波動類
    IndicatorSpec(
        "bollinger",
        "布林通道",
        {"period": Param("天數"), "k": Param("標準差倍數", float, "{} 倍標準差")},
        lambda p, period, k: ind.bollinger(p["close"], period, k),
        outputs=("mid", "upper", "lower", "pct_b", "bandwidth"),
        line_labels=("中線", "上緣", "下緣", "%b（在通道裡的位置）", "帶寬"),
        overlay=("mid", "upper", "lower"),
    ),
    IndicatorSpec(
        "keltner",
        "肯特納通道",
        {"period": Param("天數"), "k": Param("ATR 倍數", float, "{} 倍 ATR")},
        lambda p, period, k: ind.keltner(p, period, k),
        outputs=("mid", "upper", "lower"),
        line_labels=("中線", "上緣", "下緣"),
        overlay=True,
    ),
    IndicatorSpec(
        "donchian",
        "唐奇安通道",
        _period(),
        lambda p, period: ind.donchian(p, period),
        outputs=("upper", "lower", "mid"),
        line_labels=("上緣", "下緣", "中線"),
        overlay=True,
    ),
    IndicatorSpec(
        "atr", "平均真實波幅 ATR", _period(), lambda p, period: ind.atr(p, period)
    ),
    IndicatorSpec(
        "hv",
        "歷史波動率（年化）",
        _vol_params(),
        lambda p, period, year_days: ind.realized_vol(p["close"], period, year_days),
    ),
    IndicatorSpec(
        "parkinson_vol",
        "Parkinson 波動率（年化）",
        _vol_params(),
        lambda p, period, year_days: ind.parkinson_vol(p, period, year_days),
    ),
    IndicatorSpec(
        "garman_klass_vol",
        "Garman-Klass 波動率（年化）",
        _vol_params(),
        lambda p, period, year_days: ind.garman_klass_vol(p, period, year_days),
    ),
    # 趨勢類
    IndicatorSpec(
        "adx",
        "ADX／DMI",
        _period(),
        lambda p, period: ind.adx(p, period),
        outputs=("adx", "plus_di", "minus_di"),
        line_labels=("ADX", "+DI", "−DI"),
    ),
    IndicatorSpec(
        "ichimoku",
        "一目均衡表",
        {
            "tenkan": Param("轉換線天數"),
            "kijun": Param("基準線天數"),
            "senkou": Param("先行帶 B 天數"),
        },
        lambda p, tenkan, kijun, senkou: ind.ichimoku(p, tenkan, kijun, senkou),
        outputs=("tenkan", "kijun", "span_a", "span_b"),
        line_labels=("轉換線", "基準線", "先行帶 A", "先行帶 B"),
        overlay=True,
    ),
    IndicatorSpec(
        "sar",
        "Parabolic SAR",
        {"step": Param("加速因子", float), "max_step": Param("加速因子上限", float)},
        lambda p, step, max_step: ind.parabolic_sar(p, step, max_step),
        overlay=True,
    ),
    IndicatorSpec(
        "supertrend",
        "SuperTrend",
        {"period": Param("ATR 天數"), "multiplier": Param("ATR 倍數", float, "{} 倍 ATR")},
        lambda p, period, multiplier: ind.supertrend(p, period, multiplier),
        outputs=("supertrend", "direction"),
        line_labels=("SuperTrend 線", "方向（1 多頭、−1 空頭）"),
        overlay=("supertrend",),
    ),
    # 型態類
    IndicatorSpec(
        "candle",
        "K 線實體與影線長度",
        {},
        lambda p: ind.candle_parts(p),
        outputs=("body", "upper_shadow", "lower_shadow"),
        line_labels=("實體", "上影線", "下影線"),
    ),
    IndicatorSpec(
        "darvas",
        "Darvas 箱型",
        {
            "new_high_days": Param("創幾日新高才開始找箱子", short="創 {} 日新高"),
            "confirm_days": Param(
                "確認天數：連續幾天沒有突破，箱頂、箱底才算確定", short="確認 {} 天"
            ),
        },
        lambda p, new_high_days, confirm_days: ind.darvas_box(
            p, new_high_days, confirm_days
        ),
        outputs=("top", "bottom", "stop"),
        line_labels=("箱頂", "箱底", "停損線"),
        overlay=True,
        # 要有 new_high_days 天的最高價，才知道是不是創新高
        warmup=lambda new_high_days, confirm_days: new_high_days - 1,
    ),
    # 轉折點確定以後一直有值（沿用到下一個），沒有「當時沒有型態」的空值；
    # 第一個轉折點確定以前沒有值，是還算不出來，所以不給暖機天數
    IndicatorSpec(
        "swing",
        "波段轉折點",
        {
            "percent": Param(
                "反轉幅度（%）：從最高收盤價回跌、從最低收盤價回漲幾 %，才確定是高點、低點",
                float,
                "回跌、回漲 {}% 才確定",
            ),
        },
        lambda p, percent: ind.swing_points(p, percent),
        outputs=("peak", "trough"),
        line_labels=("最近確定的高點", "最近確定的低點"),
        overlay=True,
    ),
    # 量能類
    IndicatorSpec("obv", "能量潮 OBV", {}, lambda p: ind.obv(p)),
    IndicatorSpec(
        "mfi", "資金流量指標 MFI", _period(), lambda p, period: ind.mfi(p, period)
    ),
    IndicatorSpec(
        "cmf", "蔡金資金流 CMF", _period(), lambda p, period: ind.cmf(p, period)
    ),
    IndicatorSpec(
        "vwap",
        "成交量加權均價 VWAP",
        _period(),
        lambda p, period: ind.vwap(p, period),
        overlay=True,
    ),
    IndicatorSpec(
        "volume_ratio", "量比", _period(), lambda p, period: ind.volume_ratio(p, period)
    ),
]

CATALOG: dict[str, IndicatorSpec] = {spec.type: spec for spec in _SPECS}
