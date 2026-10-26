import numpy as np
import pandas as pd

from hermes import technical as ta


def test_rsi_extremes():
    up = pd.Series(np.arange(1.0, 60.0))
    assert ta.rsi(up, 14).iloc[-1] == 100.0
    down = pd.Series(np.arange(60.0, 1.0, -1))
    assert ta.rsi(down, 14).iloc[-1] < 1e-9


def test_rsi_matches_wilder_definition():
    s = pd.Series([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61,
                   46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64])
    d = s.diff()
    gain, loss = d.clip(lower=0), (-d).clip(lower=0)
    ag, al = gain.iloc[1:15].mean(), loss.iloc[1:15].mean()
    # Wilder 的遞迴（以 EWM alpha=1/n、從第一筆差分開始）
    ag2 = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al2 = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    expected = 100 - 100 / (1 + ag2 / al2)
    assert np.allclose(ta.rsi(s, 14).dropna(), expected.dropna())
    assert ag > 0 and al > 0


def test_kd_bounds_and_seed(prices):
    kd = ta.stochastic_kd(prices, 9, 3, 3)
    v = kd.dropna()
    assert ((v >= 0) & (v <= 100)).all().all()
    assert kd["k"].iloc[:8].isna().all()


def test_bollinger_mid_is_sma(prices):
    bb = ta.bollinger(prices["close"], 20, 2.0)
    assert np.allclose(bb["mid"].dropna(), ta.sma(prices["close"], 20).dropna())
    inside = bb.dropna()
    assert (inside["upper"] >= inside["lower"]).all()


def test_cci_matches_definition(prices):
    """逐日照定義計算：CCI =（TP − n 日平均）/（0.015 × n 日平均絕對偏差）。"""
    n = 20
    tp = ((prices["high"] + prices["low"] + prices["close"]) / 3.0).to_numpy()
    expected = [np.nan] * (n - 1)
    for i in range(n - 1, len(tp)):
        w = tp[i - n + 1: i + 1]
        expected.append((w[-1] - w.mean()) / (0.015 * np.mean(np.abs(w - w.mean()))))
    assert np.allclose(ta.cci(prices, n), expected, equal_nan=True)


def test_wma_matches_definition():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(ta.wma(s, 3), [np.nan, np.nan, (1 + 4 + 9) / 6, (2 + 6 + 12) / 6, (3 + 8 + 15) / 6],
                       equal_nan=True)
