"""貝氏線上變點偵測（Bayesian Online Changepoint Detection）。

- Adams & MacKay (2007), "Bayesian Online Changepoint Detection", arXiv:0710.3742。
- Wood, Roberts & Zohren (2021), "Slow Momentum with Fast Reversion: A Trading Strategy Using
  Deep Learning and Changepoint Detection", J. Financial Data Science —— 將變點的「位置」與
  「嚴重度」作為 Deep Momentum Network 的輸入特徵，改善動能策略在轉折點的表現。

原論文以高斯過程做變點偵測；這裡用 Adams & MacKay 的 Normal-Gamma 共軛模型，
屬於純線上（只用過去資料）的濾波器，可以直接當作特徵而不會前視。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln


def bocpd(
    x: np.ndarray,
    hazard: float = 1.0 / 126,
    max_run: int = 252,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> np.ndarray:
    """回傳 run-length 後驗分布矩陣 R（T × (max_run+1)），R[t, r] = P(r_t = r | x_1..t)。"""
    T = len(x)
    R = np.zeros((T, max_run + 1))
    run = np.array([1.0])
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])
    for t in range(T):
        xt = x[t]
        if not np.isfinite(xt):
            R[t, : len(run)] = run[: max_run + 1]
            continue
        # Student-t 預測分布
        scale2 = beta * (kappa + 1) / (alpha * kappa)
        nu = 2 * alpha
        z = (xt - mu) ** 2 / (nu * scale2)
        log_pred = (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(np.pi * nu * scale2)
            - (nu + 1) / 2 * np.log1p(z)
        )
        pred = np.exp(log_pred)
        growth = run * pred * (1 - hazard)
        cp = float(np.sum(run * pred * hazard))
        new_run = np.concatenate([[cp], growth])
        total = new_run.sum()
        new_run = new_run / total if total > 0 else np.r_[1.0, np.zeros(len(growth))]
        # 更新充分統計量
        mu_new = (kappa * mu + xt) / (kappa + 1)
        kappa_new = kappa + 1
        alpha_new = alpha + 0.5
        beta_new = beta + kappa * (xt - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa_new])
        alpha = np.concatenate([[alpha0], alpha_new])
        beta = np.concatenate([[beta0], beta_new])
        if len(new_run) > max_run + 1:
            # 截斷：把超出部分的機率併入最後一格
            tail = new_run[max_run:].sum()
            new_run = new_run[: max_run + 1]
            new_run[-1] = tail
            mu, kappa, alpha, beta = (
                mu[: max_run + 1],
                kappa[: max_run + 1],
                alpha[: max_run + 1],
                beta[: max_run + 1],
            )
        run = new_run
        R[t, : len(run)] = run
    return R


def changepoint_features(
    close: pd.Series,
    lookback: int = 21,
    hazard: float = 1.0 / 126,
    max_run: int = 252,
    vol_span: int = 63,
) -> pd.DataFrame:
    """以波動度標準化後的日報酬跑 BOCPD，輸出：

    - cp_location：期望 run length 的倒數位置（0=很久沒有變點，1=剛發生變點），對應 Wood et al. 的 location。
    - cp_severity：最近 lookback 日內發生變點的後驗機率，對應 severity。
    """
    r = np.log(close).diff()
    vol = r.ewm(span=vol_span, min_periods=20).std().shift(1)
    z = (r / vol).clip(-6, 6).to_numpy()
    R = bocpd(np.nan_to_num(z, nan=0.0), hazard=hazard, max_run=max_run)
    runs = np.arange(R.shape[1])
    expected = R @ runs
    loc = 1.0 - np.minimum(expected, lookback) / lookback
    severity = R[:, :lookback].sum(axis=1)
    out = pd.DataFrame({"cp_location": loc, "cp_severity": severity}, index=close.index)
    out[vol.isna().to_numpy()] = np.nan
    return out
