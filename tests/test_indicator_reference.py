"""用獨立的實作對照技術指標。

- 一律透過指標目錄（CATALOG）計算，也就是策略設定檔實際用到的那條路徑。
- 指標目錄裡的每個指標、每條輸出的線，都要在 REFERENCES 有對照；少一個測試就失敗。
- TA-Lib 有的指標：直接和 TA-Lib 比較。
  EMA、Wilder 平滑類（RSI、ATR、ADX、MACD、TRIX、KD）的起始值算法和 TA-Lib 不同，
  只比較暖機期之後（差異會隨時間指數衰減到 0）。
- TA-Lib 沒有的指標：用照定義逐日計算的簡單寫法對照。
"""

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import talib

from conftest import make_prices
from hermes.technical import CATALOG

WARMUP = 300


def _tick_prices() -> pd.DataFrame:
    """價格對齊到 0.5 元，會出現「剛好碰到」的情況；第 400～415 天連續上漲。"""
    p = make_prices(n=900, seed=7)
    for col in ("open", "high", "low", "close"):
        p[col] = (p[col] * 2).round() / 2
    for k, i in enumerate(range(400, 416)):
        p.iloc[i, p.columns.get_indexer(["open", "high", "low", "close"])] = p["close"].iloc[399] + k + 1
    p["high"] = p[["open", "high", "close"]].max(axis=1)
    p["low"] = p[["open", "low", "close"]].min(axis=1)
    return p


DATASETS = {"continuous": make_prices(n=900), "tick": _tick_prices()}


@pytest.fixture(params=list(DATASETS))
def data(request):
    p = DATASETS[request.param]
    arrays = {k: p[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume")}
    return p, arrays


def assert_same(ours, ref, warmup=0, label=""):
    ours, ref = np.asarray(ours, dtype=float), np.asarray(ref, dtype=float)
    if warmup == 0:
        assert (np.isnan(ours) == np.isnan(ref)).all(), f"{label} 沒有值的位置不同"
    both = ~np.isnan(ours) & ~np.isnan(ref)
    both[:warmup] = False
    assert both.sum() > 100, label
    np.testing.assert_allclose(ours[both], ref[both], rtol=1e-9, atol=1e-7, err_msg=label)


# 每個指標的對照：(prices, TA-Lib 用的陣列) → [(參數, {線: (對照值, 暖機天數)})]；只有一條線的指標，線寫 None
Lines = dict[str | None, tuple[object, int]]
REFERENCES: dict[str, Callable[[pd.DataFrame, dict], list[tuple[dict, Lines]]]] = {}


def reference(type_: str):
    def deco(fn):
        REFERENCES[type_] = fn
        return fn
    return deco


def test_every_indicator_has_reference():
    missing = sorted(set(CATALOG) - set(REFERENCES))
    assert not missing, f"這些指標沒有獨立的對照，請在 REFERENCES 加上：{'、'.join(missing)}"
    assert set(REFERENCES) <= set(CATALOG)


@pytest.mark.parametrize("type_", sorted(CATALOG))
def test_indicator_matches_reference(type_, data):
    assert type_ in REFERENCES, f"{type_} 沒有獨立的對照"
    p, a = data
    spec = CATALOG[type_]
    for params, lines in REFERENCES[type_](p, a):
        out = spec.compute(p, **spec.resolve_params(params))
        declared = set(spec.outputs) or {None}
        assert set(lines) == declared, f"{type_} 每條線都要對照，少了：{declared - set(lines)}"
        for line, (ref, warmup) in lines.items():
            assert_same(out[line] if line else out, ref, warmup, f"{type_}{params} {line or ''}")


# ------------------------------------------------------------------- TA-Lib
@reference("sma")
def _sma(p, a):
    return [({"period": n}, {None: (talib.SMA(a["close"], n), 0)}) for n in (5, 20, 60)]


@reference("wma")
def _wma(p, a):
    return [({"period": n}, {None: (talib.WMA(a["close"], n), 0)}) for n in (5, 20, 60)]


@reference("ema")
def _ema(p, a):
    # 起始值的影響每天乘上 (1 − 2/(n+1))，天數越長衰減越慢
    return [({"period": n}, {None: (talib.EMA(a["close"], n), 12 * n + 50)}) for n in (5, 20, 60)]


@reference("cci")
def _cci(p, a):
    return [({"period": n}, {None: (talib.CCI(a["high"], a["low"], a["close"], n), 0)}) for n in (14, 20, 88)]


@reference("williams_r")
def _willr(p, a):
    return [({"period": 14}, {None: (talib.WILLR(a["high"], a["low"], a["close"], 14), 0)})]


@reference("roc")
def _roc(p, a):
    return [({"period": 12}, {None: (talib.ROC(a["close"], 12), 0)})]


@reference("bollinger")
def _bollinger(p, a):
    upper, mid, lower = talib.BBANDS(a["close"], 20, 2.0, 2.0, 0)
    # %b、帶寬照定義用 TA-Lib 的上下軌計算
    return [({"period": 20, "k": 2.0}, {"upper": (upper, 0), "mid": (mid, 0), "lower": (lower, 0),
                                        "pct_b": ((a["close"] - lower) / (upper - lower), 0),
                                        "bandwidth": ((upper - lower) / mid, 0)})]


@reference("rsi")
def _rsi(p, a):
    return [({"period": 14}, {None: (talib.RSI(a["close"], 14), WARMUP)})]


@reference("atr")
def _atr(p, a):
    return [({"period": 14}, {None: (talib.ATR(a["high"], a["low"], a["close"], 14), WARMUP)})]


@reference("adx")
def _adx(p, a):
    hlc = (a["high"], a["low"], a["close"])
    return [({"period": 14}, {"adx": (talib.ADX(*hlc, 14), WARMUP), "plus_di": (talib.PLUS_DI(*hlc, 14), WARMUP),
                              "minus_di": (talib.MINUS_DI(*hlc, 14), WARMUP)})]


@reference("macd")
def _macd(p, a):
    macd, signal, hist = talib.MACD(a["close"], 12, 26, 9)
    return [({"fast": 12, "slow": 26, "signal": 9},
             {"macd": (macd, WARMUP), "signal": (signal, WARMUP), "hist": (hist, WARMUP)})]


@reference("trix")
def _trix(p, a):
    return [({"period": 15}, {None: (talib.TRIX(a["close"], 15), WARMUP)})]


@reference("stoch_rsi")
def _stoch_rsi(p, a):
    return [({"period": 14}, {None: (talib.STOCHRSI(a["close"], 14, 14, 1, 0)[0] / 100, WARMUP)})]


@reference("kd")
def _kd(p, a):
    """台股 KD 的 1/3 權重平滑等於 α = 2/(5+1) 的 EMA，所以可以用 TA-Lib STOCH 對照。"""
    k, d = talib.STOCH(a["high"], a["low"], a["close"], fastk_period=9,
                       slowk_period=5, slowk_matype=1, slowd_period=5, slowd_matype=1)
    return [({"period": 9, "k_smooth": 3, "d_smooth": 3}, {"k": (k, WARMUP), "d": (d, WARMUP)})]


@reference("keltner")
def _keltner(p, a):
    mid = talib.EMA(a["close"], 20)
    atr = talib.ATR(a["high"], a["low"], a["close"], 20)
    warmup = 500  # ATR(20) 起始值的差異每天乘上 19/20，300 天後還有約 2e-7
    return [({"period": 20, "k": 2.0}, {"mid": (mid, warmup), "upper": (mid + 2.0 * atr, warmup),
                                        "lower": (mid - 2.0 * atr, warmup)})]


@reference("sar")
def _sar(p, a):
    return [({"step": 0.02, "max_step": 0.2}, {None: (talib.SAR(a["high"], a["low"], 0.02, 0.2), 0)})]


@reference("obv")
def _obv(p, a):
    # TA-Lib 第一天從當天成交量開始累加，這裡從 0 開始：兩者差一個固定值
    return [({}, {None: (talib.OBV(a["close"], a["volume"]) - a["volume"][0], 0)})]


@reference("mfi")
def _mfi(p, a):
    return [({"period": 14}, {None: (talib.MFI(a["high"], a["low"], a["close"], a["volume"], 14), 0)})]


def test_mfi_is_100_after_straight_rally():
    mfi = CATALOG["mfi"].compute(DATASETS["tick"], period=14)
    assert mfi.iloc[415] == 100.0


# ------------------------------------------------------- 照定義逐日計算的對照
def _rows(p):
    return list(p[["open", "high", "low", "close", "volume"]].itertuples(index=False))


def _window_or_none(values, i, n):
    return values[i - n + 1: i + 1] if i >= n - 1 else None


@reference("bias")
def _bias(p, a):
    close = list(p["close"])
    out = []
    for i in range(len(close)):
        w = _window_or_none(close, i, 20)
        out.append((close[i] - sum(w) / 20) / (sum(w) / 20) * 100 if w else math.nan)
    return [({"period": 20}, {None: (out, 0)})]


@reference("psy")
def _psy(p, a):
    close = list(p["close"])
    out = []
    for i in range(len(close)):
        ups = [close[j] > close[j - 1] for j in range(i - 11, i + 1)] if i >= 12 else None
        out.append(sum(ups) / 12 * 100 if ups else math.nan)
    return [({"period": 12}, {None: (out, 0)})]


@reference("volume_ratio")
def _volume_ratio(p, a):
    volume = list(p["volume"])
    out = []
    for i in range(len(volume)):
        w = _window_or_none(volume, i, 20)
        out.append(volume[i] / (sum(w) / 20) if w else math.nan)
    return [({"period": 20}, {None: (out, 0)})]


@reference("cci_rolling_dev")
def _cci_rolling_dev(p, a):
    tp = [(r.high + r.low + r.close) / 3 for r in _rows(p)]
    cases = []
    for n in (14, 20, 88):
        ma = [sum(w) / n if (w := _window_or_none(tp, i, n)) else None for i in range(len(tp))]
        out = []
        for i in range(len(tp)):
            if i < 2 * n - 2:
                out.append(math.nan)
                continue
            # 每天各自減掉那一天的 n 日平均，再取 n 日平均
            md = sum(abs(tp[j] - ma[j]) for j in range(i - n + 1, i + 1)) / n
            out.append((tp[i] - ma[i]) / (0.015 * md) if md else math.nan)
        cases.append(({"period": n}, {None: (out, 0)}))
    return cases


@reference("candle")
def _candle(p, a):
    body, upper, lower = [], [], []
    for r in _rows(p):
        top, bottom = (r.open, r.close) if r.open >= r.close else (r.close, r.open)
        body.append(top - bottom)
        upper.append(r.high - top)
        lower.append(bottom - r.low)
    return [({}, {"body": (body, 0), "upper_shadow": (upper, 0), "lower_shadow": (lower, 0)})]


def _window_stat(p, n, fn):
    rows = _rows(p)
    return [fn(w) if (w := _window_or_none(rows, i, n)) is not None else math.nan for i in range(len(rows))]


@reference("donchian")
def _donchian(p, a):
    up = _window_stat(p, 20, lambda w: max(r.high for r in w))
    lo = _window_stat(p, 20, lambda w: min(r.low for r in w))
    return [({"period": 20}, {"upper": (up, 0), "lower": (lo, 0), "mid": ((np.array(up) + np.array(lo)) / 2, 0)})]


@reference("vwap")
def _vwap(p, a):
    out = _window_stat(p, 20, lambda w: sum((r.high + r.low + r.close) / 3 * r.volume for r in w)
                       / sum(r.volume for r in w))
    return [({"period": 20}, {None: (out, 0)})]


@reference("cmf")
def _cmf(p, a):
    def cmf(w):
        mfv = [((r.close - r.low) - (r.high - r.close)) / (r.high - r.low) * r.volume if r.high > r.low else 0.0
               for r in w]
        return sum(mfv) / sum(r.volume for r in w)

    return [({"period": 20}, {None: (_window_stat(p, 20, cmf), 0)})]


@reference("hv")
def _hv(p, a):
    rows = _rows(p)
    out = []
    for i in range(len(rows)):
        if i < 21:
            out.append(math.nan)
            continue
        r = [math.log(rows[j].close / rows[j - 1].close) for j in range(i - 20, i + 1)]
        mean = sum(r) / 21
        out.append(math.sqrt(sum((x - mean) ** 2 for x in r) / 20) * math.sqrt(244))
    return [({"period": 21, "year_days": 244}, {None: (out, 0)})]


@reference("parkinson_vol")
def _parkinson(p, a):
    out = _window_stat(p, 21, lambda w: math.sqrt(sum(math.log(x.high / x.low) ** 2 for x in w) / 21
                                                  / (4 * math.log(2)) * 244))
    return [({"period": 21, "year_days": 244}, {None: (out, 0)})]


@reference("garman_klass_vol")
def _garman_klass(p, a):
    def gk(w):
        g = [0.5 * math.log(x.high / x.low) ** 2 - (2 * math.log(2) - 1) * math.log(x.close / x.open) ** 2 for x in w]
        return math.sqrt(max(sum(g) / 21, 0.0) * 244)

    return [({"period": 21, "year_days": 244}, {None: (_window_stat(p, 21, gk), 0)})]


@reference("ichimoku")
def _ichimoku(p, a):
    rows = _rows(p)

    def mid(i, n):
        w = _window_or_none(rows, i, n)
        return (max(r.high for r in w) + min(r.low for r in w)) / 2 if w else math.nan

    tenkan = [mid(i, 9) for i in range(len(rows))]
    kijun = [mid(i, 26) for i in range(len(rows))]
    # 先行帶是 26 天前算出來、往前平移到今天的值
    span_a = [(tenkan[i - 26] + kijun[i - 26]) / 2 if i >= 26 else math.nan for i in range(len(rows))]
    span_b = [mid(i - 26, 52) if i >= 26 else math.nan for i in range(len(rows))]
    return [({"tenkan": 9, "kijun": 26, "senkou": 52},
             {"tenkan": (tenkan, 0), "kijun": (kijun, 0), "span_a": (span_a, 0), "span_b": (span_b, 0)})]


@reference("supertrend")
def _supertrend(p, a):
    """KivancOzbilgic 版 SuperTrend：最終上下軌只往有利方向移動；收盤突破前一天的軌道才換方向。

    ATR 用 atr 指標的結果（它本身已經和 TA-Lib 對照過）；SuperTrend 的路徑會受暖機期的 ATR 影響，
    所以不直接拿 TA-Lib 的 ATR。
    """
    rows = _rows(p)
    atr = CATALOG["atr"].compute(p, period=10).to_numpy()
    final_up = final_lo = math.nan
    direction, line, dirs = 1, [], []
    for i, r in enumerate(rows):
        if math.isnan(atr[i]) or i == 0:
            line.append(math.nan), dirs.append(1.0)
            continue
        hl2 = (r.high + r.low) / 2
        basic_up, basic_lo = hl2 + 3 * atr[i], hl2 - 3 * atr[i]
        prev_close = rows[i - 1].close
        prev_up, prev_lo = final_up, final_lo
        final_up = min(basic_up, prev_up) if not math.isnan(prev_up) and prev_close <= prev_up else basic_up
        final_lo = max(basic_lo, prev_lo) if not math.isnan(prev_lo) and prev_close >= prev_lo else basic_lo
        if r.close > prev_up:
            direction = 1
        elif r.close < prev_lo:
            direction = -1
        line.append(final_lo if direction > 0 else final_up)
        dirs.append(float(direction))
    return [({"period": 10, "multiplier": 3.0}, {"supertrend": (line, 0), "direction": (dirs, 0)})]



# ------------------------------------------------------------------- 型態類
def _first(days, ok):
    return next((d for d in days if ok(d)), None)


def _darvas_reference(high, low, new_high_days, confirm_days):
    """照定義一段一段搜尋，不用計數器：每一步都直接找「第一個符合定義的日子」。

    1. 起點：第一個最高價等於含當天 new_high_days 天最高價的日子。
    2. 箱頂：起點以後第一個「不低於起點到當天的所有最高價，而且之後 k 天的最高價都比它低」的日子。
    3. 箱底：箱頂那天以後第一個「不高於箱頂那天到當天的所有最低價，而且之後 k 天的最低價都比它高」的日子，
       箱子在它之後第 k 天成形。箱頂確定以後、箱子成形以前，最高價超過箱頂：從那天重新開始。
    4. 箱子從成形那天，到第一個最高價超過箱頂或最低價低於箱底的前一天；從破掉那天重新開始。
    停損線：箱子成形那天換成箱底，向上突破那天換成箱頂，跌破那天換成空值，其他日子沿用前一天。
    """
    n, k = len(high), confirm_days
    top, bottom = [math.nan] * n, [math.nan] * n
    changes: dict[int, float] = {}
    start = 0
    while True:
        s = _first(range(max(start, new_high_days - 1), n),
                   lambda d: high[d] == max(high[d - new_high_days + 1: d + 1]))
        if s is None:
            break
        t = _first(range(s, n - k), lambda d: high[d] >= max(high[s: d + 1]) and max(high[d + 1: d + k + 1]) < high[d])
        if t is None:
            break
        box_top, confirmed = high[t], t + k
        u = _first(range(t, n - k), lambda d: low[d] <= min(low[t: d + 1]) and min(low[d + 1: d + k + 1]) > low[d])
        formed = u + k if u is not None else None
        broken = _first(range(confirmed + 1, (formed if formed is not None else n - 1) + 1),
                        lambda d: high[d] > box_top)
        if broken is not None:
            start = broken
            continue
        if formed is None:
            break
        box_bottom = low[u]
        end = _first(range(formed + 1, n), lambda d: high[d] > box_top or low[d] < box_bottom)
        for d in range(formed, end if end is not None else n):
            top[d], bottom[d] = box_top, box_bottom
        changes[formed] = box_bottom
        if end is None:
            break
        changes[end] = box_top if high[end] > box_top else math.nan
        start = end
    stop, level = [], math.nan
    for d in range(n):
        level = changes.get(d, level)
        stop.append(level)
    return top, bottom, stop


@reference("darvas")
def _darvas(p, a):
    high, low = list(a["high"]), list(a["low"])
    out = []
    # 參數挑過：兩組測試資料都要有超過 100 天有箱子；(5, 1) 會遇到同一天突破箱頂又跌破箱底，
    # 0.5 元的那組資料會遇到剛好碰到箱頂、箱底
    for days, k in ((10, 3), (20, 3), (15, 5), (5, 1)):
        top, bottom, stop = _darvas_reference(high, low, days, k)
        out.append(({"new_high_days": days, "confirm_days": k},
                    {"top": (top, 0), "bottom": (bottom, 0), "stop": (stop, 0)}))
    return out


def _swing_events(close, percent):
    """照定義一段一段搜尋，不用狀態變數：每一步都直接找「第一個符合定義的日子」。

    1. 第一個轉折點：第一個「收盤 ≤ 第一天到當天的最高收盤 ×（1 − x）」或「收盤 ≥ 第一天到當天的最低收盤 ×（1 + x）」的日子，
       前者確定高點、後者確定低點，價格是那段的最高、最低收盤。
    2. 之後輪流：確定高點以後，找第一個「收盤 ≥ 高點那天之後到當天的最低收盤 ×（1 + x）」的日子，確定低點；低點相反。
    回傳 [(確定那天, "peak" 或 "trough", 價格, 轉折點那天)]；一樣高時轉折點那天取最後一天。
    """
    def at_most(a, b):
        return a <= b or math.isclose(a, b)

    def at_least(a, b):
        return a >= b or math.isclose(a, b)

    down, up = 1 - percent / 100, 1 + percent / 100
    n, events, start = len(close), [], 0

    def is_peak(d):
        return at_most(close[d], max(close[start: d + 1]) * down)

    def is_trough(d):
        return at_least(close[d], min(close[start: d + 1]) * up)

    day = _first(range(n), lambda d: is_peak(d) or is_trough(d))
    kind = "peak" if day is not None and is_peak(day) else "trough"
    while day is not None:
        seg = close[start: day + 1]
        price = max(seg) if kind == "peak" else min(seg)
        pivot = start + max(i for i, c in enumerate(seg) if c == price)
        events.append((day, kind, price, pivot))
        start = pivot + 1
        kind = "trough" if kind == "peak" else "peak"
        day = _first(range(day + 1, n), is_trough if kind == "trough" else is_peak)
    return events


@reference("swing")
def _swing(p, a):
    close, n = list(a["close"]), len(a["close"])
    out = []
    # 參數挑過：每組都有 20 個以上的轉折點，第一個轉折點有高點也有低點；0.5 元的那組資料
    # 3、5、10 都會遇到剛好回跌或回漲 percent%，其中 10% 有一次不加容許誤差就會漏掉
    for percent in (3.0, 5.0, 10.0):
        lines = {"peak": [math.nan] * n, "trough": [math.nan] * n}
        for day, kind, price, _ in _swing_events(close, percent):
            lines[kind][day:] = [price] * (n - day)
        out.append(({"percent": percent}, {k: (v, 0) for k, v in lines.items()}))
    return out
