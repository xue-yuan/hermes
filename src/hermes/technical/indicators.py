"""技術指標。所有函式只使用當期與過去資料（不會前視），輸入為 OHLCV DataFrame 或 Series。

欄位慣例：open、high、low、close、volume。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _windows(s: pd.Series, n: int) -> np.ndarray:
    """每列是一個長度 n 的滾動視窗；第 i 列對應原序列第 i+n-1 筆。"""
    return sliding_window_view(s.to_numpy(dtype=float), n)


def _from_windows(s: pd.Series, n: int, values: np.ndarray) -> pd.Series:
    out = np.full(len(s), np.nan)
    out[n - 1 :] = values
    return pd.Series(out, index=s.index)


# ------------------------------------------------------------------ 均線類
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder 平滑（RSI、ATR、ADX 使用），等同 alpha=1/n 的 EMA。"""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    if len(s) < n:
        return pd.Series(np.nan, index=s.index)
    weights = np.arange(1, n + 1, dtype=float)
    return _from_windows(s, n, _windows(s, n) @ weights / weights.sum())


def bias(close: pd.Series, n: int) -> pd.Series:
    """乖離率（%）：收盤價相對 n 日均線的偏離程度。"""
    ma = sma(close, n)
    return (close - ma) / ma * 100.0


# ------------------------------------------------------------------ 動能類
def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0.0), n)
    loss = wilder((-delta).clip(lower=0.0), n)
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(loss != 0.0, 100.0).where(gain.notna())


def macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def stochastic_kd(
    df: pd.DataFrame, n: int, k_smooth: int, d_smooth: int
) -> pd.DataFrame:
    """台股慣用 KD：RSV=(C-L9)/(H9-L9)，K=前K×2/3+RSV×1/3，D=前D×2/3+K×1/3，起始值 50。"""
    low_n = df["low"].rolling(n, min_periods=n).min()
    high_n = df["high"].rolling(n, min_periods=n).max()
    rng = (high_n - low_n).replace(0.0, np.nan)
    rsv = ((df["close"] - low_n) / rng * 100.0).fillna(50.0).where(low_n.notna())
    k = np.full(len(df), np.nan)
    d = np.full(len(df), np.nan)
    pk, pd_ = 50.0, 50.0
    a_k, a_d = 1.0 / k_smooth, 1.0 / d_smooth
    for i, v in enumerate(rsv.to_numpy()):
        if np.isnan(v):
            continue
        pk = (1 - a_k) * pk + a_k * v
        pd_ = (1 - a_d) * pd_ + a_d * pk
        k[i], d[i] = pk, pd_
    return pd.DataFrame({"k": k, "d": d}, index=df.index)


def williams_r(df: pd.DataFrame, n: int) -> pd.Series:
    high_n = df["high"].rolling(n, min_periods=n).max()
    low_n = df["low"].rolling(n, min_periods=n).min()
    return (high_n - df["close"]) / (high_n - low_n).replace(0.0, np.nan) * -100.0


def cci(df: pd.DataFrame, n: int) -> pd.Series:
    """CCI =（典型價格 − n 日平均）/（0.015 × n 日平均絕對偏差），典型價格 =（高 + 低 + 收）/ 3。"""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    if len(tp) < n:
        return pd.Series(np.nan, index=df.index)
    w = _windows(tp, n)
    ma = w.mean(axis=1)
    md = np.abs(w - ma[:, None]).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(md == 0.0, np.nan, (w[:, -1] - ma) / (0.015 * md))
    return _from_windows(tp, n, values)


def cci_rolling_dev(df: pd.DataFrame, n: int) -> pd.Series:
    """CCI 的另一種算法：平均偏差 = 每天的 |典型價格 − 當天的 n 日平均|，再取 n 日平均。

    標準 CCI 是視窗裡每天的典型價格都減掉「今天的」n 日平均；這裡每天減掉「那一天自己的」n 日平均，
    所以要 2n − 1 天才有值。有些看盤軟體顯示的 CCI 是這樣算的。
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    if len(tp) < 2 * n - 1:
        return pd.Series(np.nan, index=df.index)
    ma = _from_windows(tp, n, _windows(tp, n).mean(axis=1))
    md = _windows((tp - ma).abs().iloc[n - 1 :], n).mean(axis=1)
    now = 2 * n - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(
            md == 0.0,
            np.nan,
            (tp.to_numpy()[now:] - ma.to_numpy()[now:]) / (0.015 * md),
        )
    out = np.full(len(tp), np.nan)
    out[now:] = values
    return pd.Series(out, index=df.index)


def roc(close: pd.Series, n: int) -> pd.Series:
    return close.pct_change(n) * 100.0


def trix(close: pd.Series, n: int) -> pd.Series:
    e3 = ema(ema(ema(close, n), n), n)
    return e3.pct_change() * 100.0


def stoch_rsi(close: pd.Series, n: int) -> pd.Series:
    r = rsi(close, n)
    lo = r.rolling(n, min_periods=n).min()
    hi = r.rolling(n, min_periods=n).max()
    return (r - lo) / (hi - lo).replace(0.0, np.nan)


def psy(close: pd.Series, n: int) -> pd.Series:
    """心理線：n 日內上漲日數比例（%）。"""
    up = (close.diff() > 0).astype(float).where(close.diff().notna())
    return up.rolling(n, min_periods=n).mean() * 100.0


# ---------------------------------------------------------- 波動 / 通道類
def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
        axis=1,
    )
    return tr.max(axis=1, skipna=False).fillna(df["high"] - df["low"])


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return wilder(true_range(df), n)


def bollinger(close: pd.Series, n: int, k: float) -> pd.DataFrame:
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * std, mid - k * std
    width = (upper - lower) / mid
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "pct_b": pct_b, "bandwidth": width}
    )


def keltner(df: pd.DataFrame, n: int, k: float) -> pd.DataFrame:
    mid = ema(df["close"], n)
    a = atr(df, n)
    return pd.DataFrame({"mid": mid, "upper": mid + k * a, "lower": mid - k * a})


def donchian(df: pd.DataFrame, n: int) -> pd.DataFrame:
    upper = df["high"].rolling(n, min_periods=n).max()
    lower = df["low"].rolling(n, min_periods=n).min()
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2.0})


def realized_vol(close: pd.Series, n: int, periods: int) -> pd.Series:
    r = np.log(close).diff()
    return r.rolling(n, min_periods=n).std() * np.sqrt(periods)


def parkinson_vol(df: pd.DataFrame, n: int, periods: int) -> pd.Series:
    hl = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(hl.rolling(n, min_periods=n).mean() / (4.0 * np.log(2.0)) * periods)


def garman_klass_vol(df: pd.DataFrame, n: int, periods: int) -> pd.Series:
    hl = np.log(df["high"] / df["low"]) ** 2
    co = np.log(df["close"] / df["open"]) ** 2
    v = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    return np.sqrt(v.rolling(n, min_periods=n).mean().clip(lower=0.0) * periods)


# ------------------------------------------------------------------ 趨勢類
def adx(df: pd.DataFrame, n: int) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = wilder(true_range(df), n)
    plus_di = 100.0 * wilder(plus_dm, n) / tr.replace(0.0, np.nan)
    minus_di = 100.0 * wilder(minus_dm, n) / tr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return pd.DataFrame(
        {"adx": wilder(dx, n), "plus_di": plus_di, "minus_di": minus_di}
    )


def ichimoku(df: pd.DataFrame, tenkan: int, kijun: int, senkou: int) -> pd.DataFrame:
    """一目均衡表。先行帶 A/B 已向前平移 kijun 日，所以當日的值只用到過去資料。"""

    def mid(n):
        return (
            df["high"].rolling(n, min_periods=n).max()
            + df["low"].rolling(n, min_periods=n).min()
        ) / 2.0

    t, k = mid(tenkan), mid(kijun)
    span_a = ((t + k) / 2.0).shift(kijun)
    span_b = mid(senkou).shift(kijun)
    return pd.DataFrame({"tenkan": t, "kijun": k, "span_a": span_a, "span_b": span_b})


def parabolic_sar(df: pd.DataFrame, step: float, max_step: float) -> pd.Series:
    """Wilder 拋物線轉向，依 TA-Lib 的演算法（tests 以 TA-Lib 對照）。

    - 第一天沒有值；初始方向：第 2 天的 -DM > 0 為空頭，否則為多頭。
    - 價格「碰到」SAR（多頭時最低價 ≤ SAR）就反轉，反轉當天的 SAR 為前一段的極值點，
      且不能落在前一天和當天的價格區間內。
    - 下一天的 SAR 不能超過最近兩天的最低價（多頭）或低於最近兩天的最高價（空頭）。
    """
    high, low = df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float)
    n = len(df)
    out = np.full(n, np.nan)
    if n < 2:
        return pd.Series(out, index=df.index)
    up_move, down_move = high[1] - high[0], low[0] - low[1]
    is_long = not (down_move > 0 and down_move > up_move)
    ep, sar = (high[1], low[0]) if is_long else (low[1], high[0])
    af = step
    new_high, new_low = high[1], low[1]
    for i in range(1, n):
        prev_high, prev_low = new_high, new_low
        new_high, new_low = high[i], low[i]
        if is_long:
            if new_low <= sar:
                is_long, sar = False, max(ep, prev_high, new_high)
                out[i] = sar
                af, ep = step, new_low
                sar = max(sar + af * (ep - sar), prev_high, new_high)
            else:
                out[i] = sar
                if new_high > ep:
                    ep, af = new_high, min(af + step, max_step)
                sar = min(sar + af * (ep - sar), prev_low, new_low)
        else:
            if new_high >= sar:
                is_long, sar = True, min(ep, prev_low, new_low)
                out[i] = sar
                af, ep = step, new_high
                sar = min(sar + af * (ep - sar), prev_low, new_low)
            else:
                out[i] = sar
                if new_low < ep:
                    ep, af = new_low, min(af + step, max_step)
                sar = max(sar + af * (ep - sar), prev_high, new_high)
    return pd.Series(out, index=df.index)


def supertrend(df: pd.DataFrame, n: int, mult: float) -> pd.DataFrame:
    a = atr(df, n).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2.0).to_numpy()
    close = df["close"].to_numpy()
    upper_basic, lower_basic = hl2 + mult * a, hl2 - mult * a
    upper, lower = upper_basic.copy(), lower_basic.copy()
    direction = np.ones(len(df))
    line = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        if np.isnan(a[i]):
            continue
        if not np.isnan(upper[i - 1]) and close[i - 1] <= upper[i - 1]:
            upper[i] = min(upper_basic[i], upper[i - 1])
        if not np.isnan(lower[i - 1]) and close[i - 1] >= lower[i - 1]:
            lower[i] = max(lower_basic[i], lower[i - 1])
        if close[i] > upper[i - 1]:
            direction[i] = 1
        elif close[i] < lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
        line[i] = lower[i] if direction[i] > 0 else upper[i]
    return pd.DataFrame({"supertrend": line, "direction": direction}, index=df.index)


# ------------------------------------------------------------------ 型態類
def candle_parts(df: pd.DataFrame) -> pd.DataFrame:
    """K 線的實體、上影線、下影線長度（Nison《Japanese Candlestick Charting Techniques》的定義）。

    實體 = |收盤 − 開盤|；上影線 = 最高 − max(開盤, 收盤)；下影線 = min(開盤, 收盤) − 最低。
    只算長度，不分漲跌；漲跌看收盤和開盤。
    """
    top = df[["open", "close"]].max(axis=1)
    bottom = df[["open", "close"]].min(axis=1)
    return pd.DataFrame(
        {
            "body": top - bottom,
            "upper_shadow": df["high"] - top,
            "lower_shadow": bottom - df["low"],
        },
        index=df.index,
    )


_IDLE, _TOP, _BOTTOM, _BOX = range(4)


def darvas_box(df: pd.DataFrame, new_high_days: int, confirm_days: int) -> pd.DataFrame:
    """Darvas 箱型（Nicolas Darvas，1960）。tests 以照定義搜尋的寫法對照。

    - 箱子從「創 new_high_days 日新高」（最高價不低於含當天在內 new_high_days 天的最高價）那天開始找。
    - 箱頂：這段上漲的最高價，之後連續 confirm_days 天的最高價都比它低（碰到也不算），就確定。
      還沒確定以前，最高價碰到或超過它，就改用那天的最高價重新數。
    - 箱底：從箱頂那天起的最低價，之後連續 confirm_days 天的最低價都比它高，就確定，箱子成形。
      箱頂確定以後、箱底確定以前，最高價超過箱頂，就從那天重新找箱頂。
    - 箱子成形以後，最高價超過箱頂（向上突破）或最低價低於箱底（跌破），箱子就結束，
      當天起重新找下一個創新高的日子。同一天兩邊都破，算向上突破。
    - top、bottom：當天收盤時還在的箱子，沒有成形的箱子就是空值。
    - stop：停損線。箱子成形時是箱底；向上突破時移到剛突破的箱頂，直到下一個箱子成形；
      跌破箱底以後是空值，直到下一個箱子成形。
    """
    high, low = df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float)
    rolling_high = df["high"].rolling(new_high_days, min_periods=new_high_days).max()
    new_high = (df["high"] >= rolling_high).to_numpy()
    n = len(df)
    top, bottom, stop = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    phase = _IDLE
    box_top = box_bottom = lowest = level = np.nan
    lower_highs = higher_lows = 0  # 箱頂之後最高價比它低、最低價之後比它高的連續天數
    for i in range(n):
        h, l = high[i], low[i]
        if phase == _BOX and (h > box_top or l < box_bottom):
            level = box_top if h > box_top else np.nan
            phase = _IDLE
        elif (phase == _TOP and h >= box_top) or (phase == _BOTTOM and h > box_top):
            phase = _IDLE
        if phase == _IDLE:
            if new_high[i]:
                phase, box_top, lowest, lower_highs, higher_lows = _TOP, h, l, 0, 0
        elif phase in (_TOP, _BOTTOM):
            if phase == _TOP:
                lower_highs += 1
                if lower_highs == confirm_days:
                    phase = _BOTTOM
            if l <= lowest:
                lowest, higher_lows = l, 0
            else:
                higher_lows += 1
            if phase == _BOTTOM and higher_lows >= confirm_days:
                phase, box_bottom, level = _BOX, lowest, lowest
        if phase == _BOX:
            top[i], bottom[i] = box_top, box_bottom
        stop[i] = level
    return pd.DataFrame({"top": top, "bottom": bottom, "stop": stop}, index=df.index)


# ------------------------------------------------------------------ 量能類
def obv(df: pd.DataFrame) -> pd.Series:
    """累加值：收盤沒漲跌的日子加 0，用不到成交量；其他日子的成交量是空值時，之後都算不出來。"""
    sign = np.sign(df["close"].diff()).fillna(0.0)
    return (sign * df["volume"]).where(sign != 0, 0.0).cumsum(skipna=False)


def mfi(df: pd.DataFrame, n: int) -> pd.Series:
    """MFI = 100 × 正資金流 /（正資金流 + 負資金流），需要 n 個典型價格變化（第一天沒有變化）。

    n 天全部上漲時為 100；n 天典型價格完全沒變時沒有值。
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    change = tp.diff()
    flow = tp * df["volume"]
    pos = (
        flow.where(change > 0, 0.0)
        .where(change.notna())
        .rolling(n, min_periods=n)
        .sum()
    )
    neg = (
        flow.where(change < 0, 0.0)
        .where(change.notna())
        .rolling(n, min_periods=n)
        .sum()
    )
    return 100.0 * pos / (pos + neg).replace(0.0, np.nan)


def cmf(df: pd.DataFrame, n: int) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    mfv = mfm.fillna(0.0) * df["volume"]
    return (
        mfv.rolling(n, min_periods=n).sum()
        / df["volume"].rolling(n, min_periods=n).sum()
    )


def vwap(df: pd.DataFrame, n: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    return (tp * df["volume"]).rolling(n, min_periods=n).sum() / df["volume"].rolling(
        n, min_periods=n
    ).sum()


def volume_ratio(df: pd.DataFrame, n: int) -> pd.Series:
    """量比：當日成交量 / n 日均量。"""
    return df["volume"] / sma(df["volume"], n)
