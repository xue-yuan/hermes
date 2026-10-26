import numpy as np
import pandas as pd
import pytest


def make_prices(n: int = 800, seed: int = 0, drift: float = 0.0004, vol: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    r = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, vol / 3, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
    df["value"] = df["close"] * df["volume"]
    df["trades"] = 1000.0
    df["raw_close"] = df["close"]
    df["adj_factor"] = 1.0
    return df


# 測試用的指標參數（指標目錄沒有預設值）：每個指標都要明確給參數
CATALOG_PARAMS = {
    "sma": {"period": 20}, "ema": {"period": 20}, "wma": {"period": 20}, "bias": {"period": 20},
    "rsi": {"period": 14}, "kd": {"period": 9, "k_smooth": 3, "d_smooth": 3},
    "macd": {"fast": 12, "slow": 26, "signal": 9}, "cci": {"period": 20}, "cci_rolling_dev": {"period": 20}, "williams_r": {"period": 14},
    "roc": {"period": 12}, "trix": {"period": 15}, "stoch_rsi": {"period": 14}, "psy": {"period": 12},
    "bollinger": {"period": 20, "k": 2.0}, "keltner": {"period": 20, "k": 2.0}, "donchian": {"period": 20},
    "atr": {"period": 14}, "hv": {"period": 21, "year_days": 252}, "parkinson_vol": {"period": 21, "year_days": 252},
    "garman_klass_vol": {"period": 21, "year_days": 252}, "adx": {"period": 14},
    "ichimoku": {"tenkan": 9, "kijun": 26, "senkou": 52}, "sar": {"step": 0.02, "max_step": 0.2},
    "supertrend": {"period": 10, "multiplier": 3.0}, "candle": {}, "darvas": {"new_high_days": 60, "confirm_days": 3}, "obv": {},"mfi": {"period": 14}, "cmf": {"period": 20},
    "vwap": {"period": 20}, "volume_ratio": {"period": 20},
}


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()
