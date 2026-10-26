"""深度學習模型（需要 torch：`uv sync --extra deep`）。

1. LSTMClassifier —— Fischer & Krauss (2018), "Deep learning with long short-term memory networks for
   financial market predictions", European Journal of Operational Research 270(2)。
   原文：單層 LSTM（25 個隱藏單元）、dropout 0.1、RMSprop、early stopping（patience 10），
   以過去 240 日標準化報酬序列預測次日方向。這裡改為輸入多維特徵序列（預設 60 日）。

2. DeepMomentumNetwork —— Lim, Zohren & Roberts (2019), "Enhancing Time Series Momentum Strategies
   Using Deep Neural Networks", Journal of Financial Data Science。
   直接以「負 Sharpe ratio」為損失函數，讓 LSTM 同時學習趨勢判斷與部位大小；
   部位放在波動度目標化（volatility scaling）框架內：R_{t+1} = X_t · σ_tgt/σ_t · r_{t+1}，
   並以換手成本項正則化。可選擇加入 Wood et al. (2021) 的變點特徵。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("需要安裝 torch：uv sync --extra deep") from exc

from hermes.ml.features import momentum_features
from hermes.ml.models import Preprocessor


def _windows(Z: np.ndarray, positions: np.ndarray, length: int) -> np.ndarray:
    """取出以 positions 為結尾、長度 length 的序列（N × L × F）。"""
    offsets = np.arange(-length + 1, 1)
    return Z[positions[:, None] + offsets[None, :]]


class _LSTMNet(nn.Module):
    def __init__(
        self, n_features: int, hidden: int, dropout: float, out_act: str | None = None
    ):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        self.out_act = out_act

    def forward(self, x, all_steps: bool = False):
        h, _ = self.lstm(x)
        h = self.drop(h if all_steps else h[:, -1])
        out = self.head(h).squeeze(-1)
        if self.out_act == "tanh":
            return torch.tanh(out)
        if self.out_act == "sigmoid":
            return torch.sigmoid(out)
        return out


@dataclass
class LSTMClassifier:
    context: pd.DataFrame
    seq_len: int = 60
    hidden: int = 25
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 10
    val_frac: float = 0.15
    seed: int = 0
    net: object = field(init=False, default=None)

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None
    ) -> LSTMClassifier:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.prep = Preprocessor().fit(X)
        self.Z = self.prep.transform(self.context[X.columns]).astype(np.float32)
        pos = self.context.index.get_indexer(X.index)
        ok = pos >= self.seq_len - 1
        pos, yv = pos[ok], y.to_numpy()[ok].astype(np.float32)
        w = (
            np.ones(len(pos), dtype=np.float32)
            if sample_weight is None
            else sample_weight.to_numpy()[ok].astype(np.float32)
        )
        self.single_class = None
        if len(np.unique(yv)) < 2:
            self.single_class = float(yv[0]) if len(yv) else 0.5
            return self

        n_val = max(int(len(pos) * self.val_frac), 1)
        tr, va = slice(0, len(pos) - n_val), slice(len(pos) - n_val, len(pos))
        Xtr = torch.from_numpy(_windows(self.Z, pos[tr], self.seq_len))
        Xva = torch.from_numpy(_windows(self.Z, pos[va], self.seq_len))
        ytr, yva = torch.from_numpy(yv[tr]), torch.from_numpy(yv[va])
        wtr, wva = torch.from_numpy(w[tr]), torch.from_numpy(w[va])

        self.net = _LSTMNet(self.Z.shape[1], self.hidden, self.dropout)
        opt = torch.optim.RMSprop(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(perm), self.batch_size):
                b = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = (loss_fn(self.net(Xtr[b]), ytr[b]) * wtr[b]).mean()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = float((loss_fn(self.net(Xva), yva) * wva).mean())
            if val < best - 1e-5:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.single_class is not None:
            return np.full(len(X), self.single_class)
        pos = self.context.index.get_indexer(X.index)
        out = np.full(len(X), 0.5)
        ok = pos >= self.seq_len - 1
        if ok.any():
            self.net.eval()
            with torch.no_grad():
                logits = self.net(
                    torch.from_numpy(_windows(self.Z, pos[ok], self.seq_len))
                )
            out[ok] = torch.sigmoid(logits).numpy()
        return out

    def feature_importance(self):
        return None


# ---------------------------------------------------------------------------
# Deep Momentum Network
# ---------------------------------------------------------------------------
def _sharpe_loss(
    pos: torch.Tensor,
    scaled_ret: torch.Tensor,
    inv_vol: torch.Tensor,
    cost: float,
    vol_target: float,
) -> torch.Tensor:
    """負的年化 Sharpe；pos、scaled_ret、inv_vol 形狀為 (B, L)。"""
    captured = pos * scaled_ret
    if cost > 0:
        lev = pos * inv_vol * vol_target
        turnover = torch.abs(lev[:, 1:] - lev[:, :-1])
        captured = captured[:, 1:] - cost * turnover
    flat = captured.reshape(-1)
    return -(flat.mean() / (flat.std() + 1e-9)) * np.sqrt(252.0)


@dataclass
class DMNConfig:
    seq_len: int = 63
    hidden: int = 20
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    stride: int = 21
    vol_target: float = 0.15
    cost_bps: float = 10.0
    long_only: bool = True
    max_leverage: float = 1.0
    use_changepoint: bool = True
    val_frac: float = 0.15
    seeds: int = 1


def dmn_inputs(prices: pd.DataFrame, use_changepoint: bool = True) -> pd.DataFrame:
    X = momentum_features(prices)
    if use_changepoint:
        from hermes.ml.changepoint import changepoint_features

        X = X.join(changepoint_features(prices["close"]))
    return X


def dmn_walk_forward(
    prices: pd.DataFrame,
    config: DMNConfig | None = None,
    train_min: int = 1260,
    step: int = 252,
    progress=None,
) -> pd.DataFrame:
    """以擴張視窗逐年重新訓練 DMN，回傳樣本外的 position（已含波動度目標化）與原始輸出 signal。"""
    cfg = config or DMNConfig()
    X = dmn_inputs(prices, cfg.use_changepoint)
    r = prices["close"].pct_change()
    sigma = r.ewm(span=60, min_periods=20).std() * np.sqrt(252)
    next_ret = r.shift(-1)
    scaled = (cfg.vol_target / sigma * next_ret).clip(-0.5, 0.5)

    valid = X.notna().all(axis=1) & sigma.notna()
    first = int(np.argmax(valid.to_numpy()))
    idx = prices.index
    n = len(idx)
    out = pd.DataFrame({"signal": np.nan, "position": np.nan}, index=idx)
    starts = list(range(first + train_min, n, step))
    for k, s in enumerate(starts):
        e = min(s + step, n)
        train_idx = np.arange(first, s - 1)  # s−1 的 next_ret 屬於測試期，排除
        signals = []
        for seed in range(cfg.seeds):
            net, prep = _train_dmn(
                X.iloc[train_idx],
                scaled.iloc[train_idx],
                sigma.iloc[train_idx],
                cfg,
                seed,
            )
            Z = prep.transform(X).astype(np.float32)
            test_pos = np.arange(s, e)
            test_pos = test_pos[test_pos >= cfg.seq_len - 1]
            net.eval()
            with torch.no_grad():
                sig = net(torch.from_numpy(_windows(Z, test_pos, cfg.seq_len))).numpy()
            signals.append(pd.Series(sig, index=idx[test_pos]))
        sig = pd.concat(signals, axis=1).mean(axis=1)
        out.loc[sig.index, "signal"] = sig
        if progress:
            progress(k + 1, len(starts))
    lev = (cfg.vol_target / sigma).clip(upper=cfg.max_leverage)
    pos = out["signal"] * lev
    lo = 0.0 if cfg.long_only else -cfg.max_leverage
    out["position"] = pos.clip(lo, cfg.max_leverage)
    return out


def _train_dmn(
    X: pd.DataFrame, scaled: pd.Series, sigma: pd.Series, cfg: DMNConfig, seed: int
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ok = X.notna().all(axis=1) & scaled.notna() & sigma.notna()
    X, scaled, sigma = X[ok], scaled[ok], sigma[ok]
    prep = Preprocessor().fit(X)
    Z = prep.transform(X).astype(np.float32)
    y = scaled.to_numpy().astype(np.float32)
    inv = (1.0 / sigma.to_numpy()).astype(np.float32)

    ends = np.arange(cfg.seq_len - 1, len(Z), cfg.stride)
    n_val = max(int(len(ends) * cfg.val_frac), 1)
    tr_end, va_end = ends[:-n_val], ends[-n_val:]
    # 驗證集與訓練集之間保留一個序列長度的間隔，避免重疊
    tr_end = tr_end[tr_end < va_end[0] - cfg.seq_len]

    def batchify(end_pos):
        offs = np.arange(-cfg.seq_len + 1, 1)
        sel = end_pos[:, None] + offs[None, :]
        return (
            torch.from_numpy(Z[sel]),
            torch.from_numpy(y[sel]),
            torch.from_numpy(inv[sel]),
        )

    Xtr, ytr, itr = batchify(tr_end)
    Xva, yva, iva = batchify(va_end)
    act = "sigmoid" if cfg.long_only else "tanh"
    net = _LSTMNet(Z.shape[1], cfg.hidden, cfg.dropout, out_act=act)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    cost = cfg.cost_bps / 1e4
    best, best_state, bad = np.inf, None, 0
    for _ in range(cfg.max_epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i : i + cfg.batch_size]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = _sharpe_loss(
                net(Xtr[b], all_steps=True), ytr[b], itr[b], cost, cfg.vol_target
            )
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(
                _sharpe_loss(net(Xva, all_steps=True), yva, iva, cost, cfg.vol_target)
            )
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, prep
