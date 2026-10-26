"""特徵工程。所有特徵只使用當日收盤（含）以前已公開的資訊。

特徵來源與對應文獻：
- 動能、波動、流動性：Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning", RFS
  —— 所有模型一致認為最重要的三類預測因子。
- 波動度標準化報酬與 MACD 訊號：Lim, Zohren & Roberts (2019), "Enhancing Time Series Momentum
  Strategies Using Deep Neural Networks"；Baz et al. (2015), "Dissecting Investment Strategies in the
  Cross Section and Time Series"。
- 公式化 alpha：Kakushadze (2016), "101 Formulaic Alphas", Wilmott —— 改寫成單一股票的時間序列版本。
- 分數差分：López de Prado (2018), AFML ch.5。
- 變點特徵：Wood, Roberts & Zohren (2021)。
- 台股籌碼（三大法人、融資融券、外資持股）、估值：台股特有的公開資訊。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes import technical as ta
from hermes.ml.changepoint import changepoint_features
from hermes.ml.fracdiff import frac_diff


def _zscore(s: pd.Series, n: int) -> pd.Series:
    mu = s.rolling(n, min_periods=n // 2).mean()
    sd = s.rolling(n, min_periods=n // 2).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def _ts_corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    return a.rolling(n, min_periods=n).corr(b).replace([np.inf, -np.inf], np.nan)


def _require_volume(prices: pd.DataFrame) -> None:
    """成交量有空值就不跑：模型會把空值補成中位數（models.py），等於補值。

    還原價格的成交量，在股數改變、但沒有證交所股數比例的事件以前（例如 ETF 分割）是空值。
    """
    missing = prices["volume"].isna()
    if missing.any():
        raise ValueError(
            f"還原價格的成交量到 {prices.index[missing][-1]:%Y-%m-%d} 為止算不出來（股數改變，"
            f"沒有證交所的股數比例）；機器學習用到成交量，不補值，先不能用在這檔股票"
        )


def momentum_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Lim et al. (2019) 的輸入：波動度標準化的多期報酬 + 三組 MACD 訊號。"""
    c = prices["close"]
    r = c.pct_change()
    sigma = r.ewm(span=60, min_periods=20).std()
    out = pd.DataFrame(index=prices.index)
    for h in (1, 21, 63, 126, 252):
        out[f"norm_ret_{h}"] = (c.pct_change(h) / (sigma * np.sqrt(h))).clip(-10, 10)
    for s, l in ((8, 24), (16, 48), (32, 96)):
        m = (
            c.ewm(halflife=np.log(0.5) / np.log(1 - 1 / s)).mean()
            - c.ewm(halflife=np.log(0.5) / np.log(1 - 1 / l)).mean()
        )
        q = m / c.rolling(63, min_periods=20).std()
        out[f"macd_{s}_{l}"] = q / q.rolling(252, min_periods=60).std()
    return out


def alpha101_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Kakushadze (2016) 101 Formulaic Alphas 中適用單一股票的時間序列版本（價格差分以收盤價標準化）。"""
    _require_volume(prices)
    o, h, l, c, v = (prices[k] for k in ("open", "high", "low", "close", "volume"))
    vw = ta.vwap(prices, 1)
    dc = c.diff()
    out = pd.DataFrame(index=prices.index)
    out["alpha002"] = -_ts_corr(np.log(v.replace(0, np.nan)).diff(2), (c - o) / o, 6)
    # alpha#3 與 #6 去掉截面 rank 後相同
    out["alpha006"] = -_ts_corr(o, v, 10)
    out["alpha009"] = (
        np.where(
            dc.rolling(5).min() > 0, dc, np.where(dc.rolling(5).max() < 0, dc, -dc)
        )
        / c
    )
    out["alpha012"] = np.sign(v.diff()) * (-dc / c)
    out["alpha023"] = np.where(h.rolling(20).mean() < h, -h.diff(2) / c, 0.0)
    out["alpha033"] = -(1 - o / c)
    out["alpha041"] = (np.sqrt(h * l) - vw) / c
    out["alpha053"] = -(((c - l) - (h - c)) / (h - l).replace(0, np.nan)).diff(9)
    out["alpha054"] = (-(l - c) * o**5) / ((l - h).replace(0, np.nan) * c**5)
    out["alpha101"] = (c - o) / ((h - l) + 1e-3 * c)
    return out


def price_features(
    prices: pd.DataFrame, fracdiff_d: float | None = 0.4, changepoint: bool = True
) -> pd.DataFrame:
    _require_volume(prices)
    c, v = prices["close"], prices["volume"]
    logc = np.log(c)
    out = pd.DataFrame(index=prices.index)
    # 動能（各期報酬）與短期反轉
    for h in (1, 2, 5, 10, 21, 63, 126, 252):
        out[f"ret_{h}"] = logc.diff(h)
    out["mom_12_1"] = logc.shift(21) - logc.shift(252)
    # 波動
    r = logc.diff()
    for n in (5, 21, 63):
        out[f"vol_{n}"] = r.rolling(n, min_periods=n).std() * np.sqrt(252)
    out["vol_ratio_5_63"] = out["vol_5"] / out["vol_63"]
    out["parkinson_21"] = ta.parkinson_vol(prices, 21, 252)
    out["gk_21"] = ta.garman_klass_vol(prices, 21, 252)
    out["skew_63"] = r.rolling(63, min_periods=63).skew()
    out["kurt_63"] = r.rolling(63, min_periods=63).kurt()
    # 流動性：量的 z 分數、Amihud 非流動性
    logv = np.log(v.replace(0, np.nan))
    out["volume_z_20"] = _zscore(logv, 20)
    out["volume_z_60"] = _zscore(logv, 60)
    out["amihud_21"] = np.log(
        (r.abs() / prices["value"].replace(0, np.nan)).rolling(21).mean() * 1e12
    )
    # 技術指標（皆轉為無單位）
    out["rsi_6"] = ta.rsi(c, 6) / 100
    out["rsi_14"] = ta.rsi(c, 14) / 100
    kd = ta.stochastic_kd(prices, 9, 3, 3)
    out["kd_k"] = kd["k"] / 100
    out["kd_k_minus_d"] = (kd["k"] - kd["d"]) / 100
    m = ta.macd(c, 12, 26, 9)
    out["macd_hist_norm"] = m["hist"] / c
    bb = ta.bollinger(c, 20, 2.0)
    out["bb_pct_b"] = bb["pct_b"]
    out["bb_width"] = bb["bandwidth"]
    a = ta.adx(prices, 14)
    out["adx"] = a["adx"] / 100
    out["di_diff"] = (a["plus_di"] - a["minus_di"]) / 100
    out["cci"] = ta.cci(prices, 20) / 100
    out["williams_r"] = ta.williams_r(prices, 14) / 100
    out["mfi"] = ta.mfi(prices, 14) / 100
    out["cmf"] = ta.cmf(prices, 20)
    out["atr_norm"] = ta.atr(prices, 14) / c
    out["obv_z"] = _zscore(ta.obv(prices), 60)
    for n in (5, 20, 60, 120, 240):
        out[f"bias_{n}"] = ta.bias(c, n) / 100
    out["dist_52w_high"] = c / c.rolling(252, min_periods=60).max() - 1
    out["dist_52w_low"] = c / c.rolling(252, min_periods=60).min() - 1
    out["psy_12"] = ta.psy(c, 12) / 100
    out = out.join(momentum_features(prices))
    out = out.join(alpha101_features(prices))
    if fracdiff_d is not None:
        out["fracdiff"] = frac_diff(logc, fracdiff_d)
    if changepoint:
        out = out.join(changepoint_features(c))
    return out


def exogenous_features(loader, stock_id: str, prices: pd.DataFrame) -> pd.DataFrame:
    """籌碼、估值與大盤特徵。抓不到的資料集會略過。"""
    _require_volume(prices)
    idx = prices.index
    vol = prices["volume"].replace(0, np.nan)
    out = pd.DataFrame(index=idx)

    try:
        inst = loader.institutional(stock_id).reindex(idx).fillna(0.0)
        for who in ("foreign", "trust", "dealer"):
            for n in (5, 20):
                out[f"{who}_net_{n}"] = (
                    inst[who].rolling(n).sum() / vol.rolling(n).sum()
                )
        out["foreign_net_z_60"] = _zscore(inst["foreign"], 60)
    except Exception:
        pass

    try:
        mg = loader.margin(stock_id).reindex(idx).ffill()
        out["margin_chg_5"] = mg["margin_balance"].pct_change(5)
        out["margin_chg_20"] = mg["margin_balance"].pct_change(20)
        out["short_margin_ratio"] = mg["short_margin_ratio"]
    except Exception:
        pass

    try:
        sh = loader.shareholding(stock_id).reindex(idx).ffill()
        out["foreign_ratio_chg_5"] = sh["foreign_ratio"].diff(5) / 100
        out["foreign_ratio_chg_20"] = sh["foreign_ratio"].diff(20) / 100
    except Exception:
        pass

    try:
        val = loader.valuation(stock_id).reindex(idx).ffill()
        out["per_z_3y"] = _zscore(val["PER"], 756)
        out["pbr_z_3y"] = _zscore(val["PBR"], 756)
        out["dividend_yield"] = val["dividend_yield"] / 100
    except Exception:
        pass

    try:
        mkt = loader.benchmark()["close"].reindex(idx).ffill()
        mr = np.log(mkt).diff()
        for h in (5, 21, 63):
            out[f"mkt_ret_{h}"] = np.log(mkt).diff(h)
        out["mkt_vol_21"] = mr.rolling(21).std() * np.sqrt(252)
        stock_r = np.log(prices["close"]).diff()
        out["beta_126"] = stock_r.rolling(126).cov(mr) / mr.rolling(126).var()
        out["rel_strength_63"] = np.log(prices["close"]).diff(63) - out["mkt_ret_63"]
    except Exception:
        pass

    return out.replace([np.inf, -np.inf], np.nan)


def build_features(
    prices: pd.DataFrame,
    extra: pd.DataFrame | None = None,
    fracdiff_d: float | None = 0.4,
    changepoint: bool = True,
) -> pd.DataFrame:
    X = price_features(prices, fracdiff_d=fracdiff_d, changepoint=changepoint)
    if extra is not None and not extra.empty:
        X = X.join(extra.reindex(X.index))
    return X.replace([np.inf, -np.inf], np.nan)
