"""升降單位和漲跌停的手算例子（證交所的價格規則）。"""

import pytest

from hermes.strategies import candidate_closes, tick_size


def _steps(prices):
    return sorted({round(b - a, 2) for a, b in zip(prices, prices[1:])})


@pytest.mark.parametrize(
    "ref, first, last, count, steps",
    [
        (2385, 2150, 2620, 95, [5.0]),  # 台積電 10/22
        (245, 220.5, 269.5, 99, [0.5]),  # 鴻海
        (10, 9.0, 11.0, None, [0.01, 0.05]),
        (9.5, 8.55, 10.45, None, [0.01, 0.05]),
        (1001, 901, 1100, None, [1.0, 5.0]),  # 漲停 1101.1 向下取到 1100、跌停 900.9 向上取到 901
    ],
)
def test_candidate_closes(ref, first, last, count, steps):
    prices = candidate_closes(ref)
    assert (prices[0], prices[-1]) == (first, last)
    assert _steps(prices) == steps
    if count:
        assert len(prices) == count


def test_tick_bands_line_up_at_boundaries():
    prices = candidate_closes(52)
    i = prices.index(49.95)
    assert prices[i + 1: i + 3] == [50.0, 50.1]
    assert (prices[0], prices[-1]) == (46.8, 57.2)


def test_etf_ticks():
    assert tick_size(4_999, etf=True) == 1 and tick_size(5_000, etf=True) == 5
    prices = candidate_closes(52, etf=True)
    assert 49.99 in prices and 50.05 in prices and 50.01 not in prices
