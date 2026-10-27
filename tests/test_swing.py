"""波段轉折點：手算的例子、每一天都不偷看未來。"""

import math

import numpy as np
import pandas as pd
import pytest

from conftest import make_prices
from hermes.technical import CATALOG

nan = math.nan

# 反轉幅度 10%。每一天的狀態寫在右邊。
HAND = [  # 收盤
    100.0,
    105.0,  # 1：還不知道在漲還是在跌；最高 105、最低 100
    95.0,   # 2：最低 95；95 沒有比 105 跌 10%（94.5）
    110.0,  # 3：比 95 漲 10%（104.5）以上，低點 95 確定；開始等高點，從 110 起算
    120.0,  # 4：最高 120
    120.0,  # 5：一樣高
    110.0,  # 6：還沒跌到 108
    108.0,  # 7：剛好比 120 跌 10%，也算，高點 120 確定；開始等低點，從 108 起算
    100.0,  # 8：最低 100
    109.0,  # 9：還沒漲到 110
    110.0,  # 10：剛好比 100 漲 10%（100 × 1.1 算出來是 110.00000000000001，要當成相等），低點 100 確定
    99.0,   # 11：比 10 那天的 110 剛好跌 10%，高點 110 確定
    95.0,   # 12：最低 95
    100.0,  # 13：還沒漲到 104.5
    90.0,   # 14：最低 90
    98.0,   # 15：還沒漲到 99
]
PEAK = [nan] * 7 + [120.0] * 4 + [110.0] * 5
TROUGH = [nan] * 3 + [95.0] * 7 + [100.0] * 6


def _prices(close) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=len(close))
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0},
        index=idx,
    )


def _swing(p, **params):
    spec = CATALOG["swing"]
    return spec.compute(p, **spec.resolve_params(params))


def test_hand_example():
    out = _swing(_prices(HAND), percent=10)
    np.testing.assert_array_equal(out["peak"], PEAK)
    np.testing.assert_array_equal(out["trough"], TROUGH)


def test_first_turn_can_be_a_peak():
    """一開始就跌 10%：先確定的是高點，低點還沒有。"""
    out = _swing(_prices([100.0, 95.0, 90.0]), percent=10)
    np.testing.assert_array_equal(out["peak"], [nan, nan, 100.0])
    np.testing.assert_array_equal(out["trough"], [nan, nan, nan])


def test_needs_every_param():
    with pytest.raises(ValueError, match="需要指定 percent"):
        CATALOG["swing"].resolve_params({})


def test_only_uses_close():
    """只用收盤價：開盤、最高、最低、成交量換掉，結果不變。"""
    p = make_prices(n=400, seed=1)
    out = _swing(p, percent=5)
    changed = p.assign(open=nan, high=p["close"] * 2, low=0.0, volume=nan)
    pd.testing.assert_frame_equal(_swing(changed, percent=5), out)


@pytest.mark.parametrize("seed", [0, 2])
def test_no_lookahead_every_day(seed):
    """每一天都截斷重算：第 t 天的值只能用到第 t 天以前的資料，確定的轉折點之後不會改。"""
    p = make_prices(n=400, seed=seed)
    full = _swing(p, percent=5)
    assert full["peak"].nunique() > 5
    for t in range(len(p)):
        part = _swing(p.iloc[: t + 1], percent=5).iloc[-1]
        np.testing.assert_array_equal(part.to_numpy(), full.iloc[t].to_numpy(), err_msg=f"t={t}")
