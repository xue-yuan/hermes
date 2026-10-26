"""Darvas 箱型：手算的例子、每一天都不偷看未來、附的策略設定檔。"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import make_prices
from hermes.strategies import parse_strategy, run_rules
from hermes.technical import CATALOG

REPO_STRATEGIES = Path(__file__).resolve().parents[1] / "strategies"
nan = math.nan

# 創 3 日新高才開始找、確認 3 天。每一天的狀態寫在右邊。
HAND = [  # (最高, 最低)
    (10.0, 9.0),
    (11.0, 10.0),
    (12.0, 11.0),   # 2：創 3 日新高，箱頂候選 12
    (11.5, 10.5),   # 3：最高價比 12 低第 1 天
    (11.0, 10.0),   # 4：第 2 天；最低價 10 是箱頂以來最低
    (11.8, 10.2),   # 5：第 3 天，箱頂 12 確定；最低價比 10 高第 1 天
    (11.6, 10.4),   # 6：比 10 高第 2 天
    (12.0, 10.6),   # 7：碰到箱頂不算突破；比 10 高第 3 天，箱子 12／10 成形，停損線 10
    (11.9, 10.0),   # 8：碰到箱底不算跌破
    (12.5, 11.8),   # 9：突破箱頂 12，箱子結束，停損線移到 12；又創新高，箱頂候選 12.5
    (12.4, 12.0),   # 10：比 12.5 低第 1 天
    (12.5, 12.1),   # 11：碰到 12.5，從今天重新數，箱頂候選 12.5
    (12.3, 11.9),   # 12：第 1 天
    (12.2, 12.0),   # 13：第 2 天
    (12.0, 11.95),  # 14：第 3 天，箱頂 12.5 確定；箱底還沒確定
    (12.6, 12.2),   # 15：箱底確定以前突破 12.5，從今天重新找；停損線還是 12
    (12.0, 11.5),   # 16：第 1 天；最低價 11.5
    (11.9, 11.6),   # 17：第 2 天
    (11.8, 11.7),   # 18：第 3 天，箱頂 12.6 確定
    (11.9, 11.8),   # 19：比 11.5 高第 3 天，箱子 12.6／11.5 成形，停損線 11.5
    (11.7, 11.4),   # 20：跌破箱底，箱子結束，停損線變空值；沒有創新高
    (11.0, 10.5),   # 21：沒有創新高
]
TOP = [nan] * 7 + [12.0, 12.0] + [nan] * 10 + [12.6] + [nan] * 2
BOTTOM = [nan] * 7 + [10.0, 10.0] + [nan] * 10 + [11.5] + [nan] * 2
STOP = [nan] * 7 + [10.0, 10.0] + [12.0] * 10 + [11.5] + [nan] * 2


def _hand_prices() -> pd.DataFrame:
    high, low = zip(*HAND)
    close = [(h + lo) / 2 for h, lo in HAND]
    idx = pd.bdate_range("2024-01-01", periods=len(HAND))
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1000.0},
        index=idx,
    )


def _darvas(p, **params):
    spec = CATALOG["darvas"]
    return spec.compute(p, **spec.resolve_params(params))


def test_hand_example():
    out = _darvas(_hand_prices(), new_high_days=3, confirm_days=3)
    np.testing.assert_array_equal(out["top"], TOP)
    np.testing.assert_array_equal(out["bottom"], BOTTOM)
    np.testing.assert_array_equal(out["stop"], STOP)


def test_needs_every_param():
    with pytest.raises(ValueError, match="需要指定 confirm_days"):
        CATALOG["darvas"].resolve_params({"new_high_days": 252})


@pytest.mark.parametrize("seed", [0, 2])
def test_no_lookahead_every_day(seed):
    """每一天都截斷重算：第 t 天的值只能用到第 t 天以前的資料。"""
    p = make_prices(n=400, seed=seed)
    full = _darvas(p, new_high_days=10, confirm_days=3)
    assert full["top"].notna().sum() > 50
    for t in range(len(p)):
        part = _darvas(p.iloc[: t + 1], new_high_days=10, confirm_days=3).iloc[-1]
        np.testing.assert_array_equal(part.to_numpy(), full.iloc[t].to_numpy(), err_msg=f"t={t}")


def test_strategy_file_on_hand_example():
    """附的策略：最高價突破昨天的箱頂買進；最低價跌破昨天的停損線賣出。"""
    data = yaml.safe_load((REPO_STRATEGIES / "darvas_box.yaml").read_text(encoding="utf-8"))
    data["indicators"]["box"]["new_high_days"] = 3  # 例子只有 22 天
    result = run_rules(parse_strategy(data), _hand_prices())
    days = list(range(len(HAND)))
    # 第 9 天最高 12.5 > 第 8 天的箱頂 12
    assert [d for d in days if result.buy.iloc[d]] == [9]
    # 停損線在第 9～18 天是 12，隔天的最低價低於 12 就賣：第 12 天 11.9（第 10 天剛好 12，不算跌破）、
    # 第 14 天 11.95、第 16～19 天 11.5、11.6、11.7、11.8；第 20 天最低 11.4 < 第 19 天的停損線 11.5
    assert [d for d in days if result.sell.iloc[d]] == [12, 14, 16, 17, 18, 19, 20]
    # 形成中：箱子還在、收盤在箱頂和箱底之間。第 7、8 天的 12／10，第 19 天的 12.6／11.5；
    # 第 9 天突破、第 15 天箱底確定以前突破、第 20 天跌破，都不是
    assert [d for d in days if result.forming.iloc[d]] == [7, 8, 19]
    assert result.forming_streak.iloc[[7, 8, 19]].tolist() == [1, 2, 1]


@pytest.mark.parametrize("seed", [0, 2])
def test_strategy_forming_means_box_exists(seed):
    """附的策略的形成中：收盤在箱頂、箱底之間，等於當天收盤時箱子還在（有箱頂）；那天一定沒有觸發買進。"""
    data = yaml.safe_load((REPO_STRATEGIES / "darvas_box.yaml").read_text(encoding="utf-8"))
    p = make_prices(n=400, seed=seed)
    for days, k in ((10, 3), (20, 3), (15, 5), (5, 1)):
        data["indicators"]["box"] = {"type": "darvas", "new_high_days": days, "confirm_days": k}
        result = run_rules(parse_strategy(data), p)
        assert result.forming.sum() > 30, (days, k)
        assert result.forming.equals(result.values["box.top"].notna()), (days, k)
        assert not (result.forming & result.buy).any(), (days, k)
