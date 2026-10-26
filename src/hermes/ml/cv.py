"""金融時間序列的交叉驗證（AFML ch.7、ch.12）。

- PurgedKFold：移除標籤區間與測試集重疊的訓練樣本（purge），並在測試集之後加上禁運期（embargo）。
- CombinatorialPurgedKFold（CPCV）：把資料切成 N 組，每次取 k 組當測試集，
  可以組出 φ = k/N·C(N,k) 條完整的樣本外回測路徑，得到 Sharpe 的分布而非單一數字。
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def _purge(
    train_idx: np.ndarray,
    test_ranges: list[tuple[int, int]],
    t1_pos: np.ndarray,
    start_pos: np.ndarray,
    n: int,
    embargo: int,
) -> np.ndarray:
    keep = np.ones(len(train_idx), dtype=bool)
    for lo, hi in test_ranges:
        test_start = start_pos[lo]
        test_end = t1_pos[lo : hi + 1].max()
        tr_start = start_pos[train_idx]
        tr_end = t1_pos[train_idx]
        overlap = (tr_start <= test_end) & (tr_end >= test_start)
        emb_end = min(test_end + embargo, n - 1)
        embargoed = (tr_start > test_end) & (tr_start <= emb_end)
        keep &= ~(overlap | embargoed)
    return train_idx[keep]


class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, t1: pd.Series, bar_index: pd.DatetimeIndex):
        """t1：以事件時間為 index、標籤結束時間為值的 Series（需依時間排序）。"""
        n = len(t1)
        start_pos = bar_index.get_indexer(t1.index)
        t1_pos = bar_index.get_indexer(t1.values)
        embargo = int(len(bar_index) * self.embargo_pct)
        bounds = np.array_split(np.arange(n), self.n_splits)
        for test in bounds:
            lo, hi = int(test[0]), int(test[-1])
            train = np.setdiff1d(np.arange(n), test)
            train = _purge(
                train, [(lo, hi)], t1_pos, start_pos, len(bar_index), embargo
            )
            yield train, test


class CombinatorialPurgedKFold:
    def __init__(
        self, n_groups: int = 6, n_test_groups: int = 2, embargo_pct: float = 0.01
    ):
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups 必須小於 n_groups")
        self.n_groups = n_groups
        self.k = n_test_groups
        self.embargo_pct = embargo_pct

    @property
    def n_paths(self) -> int:
        return comb(self.n_groups, self.k) * self.k // self.n_groups

    def split(self, t1: pd.Series, bar_index: pd.DatetimeIndex):
        """yield (train_idx, test_idx, test_groups)。"""
        n = len(t1)
        start_pos = bar_index.get_indexer(t1.index)
        t1_pos = bar_index.get_indexer(t1.values)
        embargo = int(len(bar_index) * self.embargo_pct)
        groups = np.array_split(np.arange(n), self.n_groups)
        for combo in combinations(range(self.n_groups), self.k):
            test = np.concatenate([groups[g] for g in combo])
            train = np.setdiff1d(np.arange(n), test)
            ranges = [(int(groups[g][0]), int(groups[g][-1])) for g in combo]
            train = _purge(train, ranges, t1_pos, start_pos, len(bar_index), embargo)
            yield train, test, combo

    def assemble_paths(
        self, split_predictions: list[tuple[tuple[int, ...], pd.Series]], t1: pd.Series
    ) -> list[pd.Series]:
        """把各次切分的測試集預測組成 φ 條完整路徑。

        split_predictions：依 split() 順序的 (test_groups, 該次測試集的預測 Series)。
        每一組在 C(N−1, k−1) 次切分中當過測試集，第 j 條路徑取該組第 j 次出現的預測。
        """
        n = len(t1)
        groups = np.array_split(np.arange(n), self.n_groups)
        per_group: dict[int, list[pd.Series]] = {g: [] for g in range(self.n_groups)}
        for combo, pred in split_predictions:
            for g in combo:
                per_group[g].append(pred.reindex(t1.index[groups[g]]))
        paths = []
        for j in range(self.n_paths):
            parts = [per_group[g][j] for g in range(self.n_groups)]
            paths.append(pd.concat(parts).sort_index())
        return paths
