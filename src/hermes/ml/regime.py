"""高斯隱馬可夫模型（Gaussian HMM）市場狀態辨識。

- Hamilton (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series and the
  Business Cycle", Econometrica —— 狀態轉換模型。
- Nystrup, Madsen & Lindström (2017), "Long Memory of Financial Time Series and Hidden Markov Models
  with Time-Varying Parameters", J. Forecasting —— 以 HMM 辨識多空狀態並據以調整資產配置。

以 Baum-Welch（EM，scaled forward-backward）估計對角共變異數的高斯 HMM；交易時只使用
前向濾波機率 P(s_t | x_1..t)，不使用平滑機率，避免前視偏差。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _log_gauss(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """回傳 T × K 的對數似然。"""
    diff = X[:, None, :] - means[None, :, :]
    return -0.5 * (
        np.sum(np.log(2 * np.pi * variances), axis=1)[None, :]
        + np.sum(diff**2 / variances[None, :, :], axis=2)
    )


@dataclass
class GaussianHMM:
    n_states: int = 2
    n_iter: int = 100
    tol: float = 1e-6
    min_var: float = 1e-6
    seed: int = 0

    def fit(self, X: np.ndarray) -> GaussianHMM:
        X = _as_2d(X)
        K = self.n_states
        rng = np.random.default_rng(self.seed)
        # 以第一個特徵的分位數初始化各狀態
        order = np.argsort(X[:, 0])
        chunks = np.array_split(order, K)
        self.means_ = np.array([X[c].mean(axis=0) for c in chunks])
        self.vars_ = np.array([X[c].var(axis=0) + self.min_var for c in chunks])
        self.startprob_ = np.full(K, 1.0 / K)
        A = np.full((K, K), 0.05 / max(K - 1, 1))
        np.fill_diagonal(A, 0.95)
        A += rng.uniform(0, 1e-3, size=(K, K))
        self.transmat_ = A / A.sum(axis=1, keepdims=True)

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            B, shift = self._emission(X)
            alpha, c = self._forward(B)
            beta = self._backward(B, c)
            ll = float(np.log(c).sum() + shift.sum())
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = (alpha[:-1].T @ (B[1:] * beta[1:] / c[1:, None])) * self.transmat_

            self.startprob_ = (gamma[0] + 1e-12) / (gamma[0] + 1e-12).sum()
            self.transmat_ = (xi + 1e-12) / (xi + 1e-12).sum(axis=1, keepdims=True)
            w = gamma.sum(axis=0)[:, None]
            self.means_ = (gamma.T @ X) / w
            self.vars_ = np.maximum((gamma.T @ X**2) / w - self.means_**2, self.min_var)
            if abs(ll - prev_ll) < self.tol * max(1.0, abs(ll)):
                break
            prev_ll = ll
        self.loglik_ = ll
        return self

    def _emission(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_b = _log_gauss(X, self.means_, self.vars_)
        shift = log_b.max(axis=1)
        return np.exp(log_b - shift[:, None]), shift

    def _forward(self, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T, K = B.shape
        A = self.transmat_
        alpha = np.empty((T, K))
        c = np.empty(T)
        a = self.startprob_ * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * B[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        return alpha, c

    def _backward(self, B: np.ndarray, c: np.ndarray) -> np.ndarray:
        T, K = B.shape
        A = self.transmat_
        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        return beta

    def filter(self, X: np.ndarray) -> np.ndarray:
        """前向濾波機率 P(s_t | x_1..t)，T × K。"""
        B, _ = self._emission(_as_2d(X))
        alpha, _ = self._forward(B)
        return alpha

    def sort_states(self, key: np.ndarray) -> GaussianHMM:
        """依 key（例如各狀態的平均報酬）由小到大重新排序，解決狀態標籤互換問題。"""
        order = np.argsort(key)
        self.means_ = self.means_[order]
        self.vars_ = self.vars_[order]
        self.startprob_ = self.startprob_[order]
        self.transmat_ = self.transmat_[np.ix_(order, order)]
        return self


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


def hmm_walk_forward(
    prices: pd.DataFrame,
    n_states: int = 2,
    train_min: int = 1000,
    step: int = 63,
    with_vol: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """擴張視窗逐段重新估計 HMM，輸出每日樣本外的「多頭狀態」濾波機率 p_bull 與最可能狀態。

    多頭狀態定義為平均報酬最高的狀態；每次重估後依平均報酬排序狀態，避免標籤互換。
    """
    r = np.log(prices["close"]).diff() * 100
    feats = pd.DataFrame({"ret": r})
    if with_vol:
        feats["logvol"] = np.log(r.rolling(10).std())
    feats = feats.dropna()
    idx = feats.index
    out = pd.DataFrame({"p_bull": np.nan, "state": np.nan}, index=prices.index)
    for s in range(train_min, len(idx), step):
        e = min(s + step, len(idx))
        model = GaussianHMM(n_states=n_states, seed=seed).fit(feats.iloc[:s].to_numpy())
        model.sort_states(model.means_[:, 0])
        probs = model.filter(feats.iloc[:e].to_numpy())[s:e]
        out.loc[idx[s:e], "p_bull"] = probs[:, -1]
        out.loc[idx[s:e], "state"] = probs.argmax(axis=1)
    return out
