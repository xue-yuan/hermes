"""模型庫。統一介面：fit(X, y, sample_weight) / predict_proba(X) → 上漲（或 meta-label=1）機率。

- logit：L2 正則化邏輯迴歸，線性基準。
- rf：隨機森林；依 López de Prado (2018) 建議，max_samples 設為樣本平均唯一性，降低重疊樣本造成的過擬合。
- et：Extremely Randomized Trees。
- hgb：梯度提升樹（Krauss, Do & Huck 2017；Gu, Kelly & Xiu 2020 中表現最好的模型之一）。
- mlp：Gu, Kelly & Xiu (2020) 的 NN3 架構（32-16-8，ReLU，early stopping）。
- lstm：Fischer & Krauss (2018) 的 LSTM 分類器（需要 torch）。
- ensemble：logit + rf + hgb 機率平均；Krauss et al. (2017)、Bielskis & Belovas (2026) 發現集成通常較穩定。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("logit", "rf", "et", "hgb", "mlp", "lstm", "ensemble")


class Preprocessor:
    """中位數補值 + 標準化（只用訓練集統計量）。"""

    def fit(self, X: pd.DataFrame) -> Preprocessor:
        self.columns = list(X.columns)
        # 訓練期間完全沒有資料的欄位（例如早期尚無籌碼資料）以 0 填補
        self.median = X.median().fillna(0.0)
        self.scaler = StandardScaler().fit(X.fillna(self.median).to_numpy())
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.scaler.transform(X[self.columns].fillna(self.median).to_numpy())
        return np.clip(np.nan_to_num(Z), -8, 8)


@dataclass
class SkModel:
    name: str
    seed: int = 0
    avg_uniqueness: float | None = None
    estimator: object = field(init=False, default=None)

    def _make(self):
        if self.name == "logit":
            return LogisticRegression(C=0.05, max_iter=2000)
        if self.name == "rf":
            max_samples = None
            if self.avg_uniqueness is not None:
                max_samples = float(np.clip(self.avg_uniqueness, 0.05, 1.0))
            return RandomForestClassifier(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=20,
                max_features="sqrt",
                class_weight="balanced_subsample",
                max_samples=max_samples,
                n_jobs=-1,
                random_state=self.seed,
            )
        if self.name == "et":
            return ExtraTreesClassifier(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=20,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.seed,
            )
        if self.name == "hgb":
            return HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_iter=300,
                max_depth=3,
                min_samples_leaf=40,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=self.seed,
            )
        if self.name == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(32, 16, 8),
                alpha=1e-3,
                learning_rate_init=1e-3,
                batch_size=128,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                random_state=self.seed,
            )
        raise ValueError(f"未知模型：{self.name}")

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None
    ) -> SkModel:
        self.prep = Preprocessor().fit(X)
        Z = self.prep.transform(X)
        self.estimator = self._make()
        self.single_class = None
        if y.nunique() < 2:
            self.single_class = float(y.iloc[0])
            return self
        w = None if sample_weight is None else sample_weight.to_numpy()
        if self.name in ("mlp", "hgb"):
            # MLPClassifier 不支援 sample_weight；HGB 在 sklearn ≥1.9 使用權重時會改用加權分位數分箱，
            # 速度慢約 7 倍。兩者都改以「依權重重抽樣」（weighted bootstrap）達到相同的加權效果。
            if w is not None:
                rng = np.random.default_rng(self.seed)
                idx = rng.choice(len(Z), size=len(Z), replace=True, p=w / w.sum())
                Z, yv = Z[idx], y.to_numpy()[idx]
            else:
                yv = y.to_numpy()
            self.estimator.fit(Z, yv)
        else:
            self.estimator.fit(Z, y.to_numpy(), sample_weight=w)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.single_class is not None:
            return np.full(len(X), self.single_class)
        p = self.estimator.predict_proba(self.prep.transform(X))
        classes = list(self.estimator.classes_)
        return p[:, classes.index(1)] if 1 in classes else np.zeros(len(X))

    def feature_importance(self) -> pd.Series | None:
        est = self.estimator
        if hasattr(est, "feature_importances_"):
            return pd.Series(est.feature_importances_, index=self.prep.columns)
        if hasattr(est, "coef_"):
            return pd.Series(np.abs(est.coef_[0]), index=self.prep.columns)
        return None


@dataclass
class EnsembleModel:
    members: tuple[str, ...] = ("logit", "rf", "hgb")
    seed: int = 0
    avg_uniqueness: float | None = None

    def fit(self, X, y, sample_weight=None) -> EnsembleModel:
        self.models = [
            SkModel(m, seed=self.seed, avg_uniqueness=self.avg_uniqueness).fit(
                X, y, sample_weight
            )
            for m in self.members
        ]
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

    def feature_importance(self) -> pd.Series | None:
        imps = [m.feature_importance() for m in self.models]
        imps = [i / i.sum() for i in imps if i is not None and i.sum() > 0]
        return sum(imps) / len(imps) if imps else None


def make_model(
    name: str,
    seed: int = 0,
    avg_uniqueness: float | None = None,
    context: pd.DataFrame | None = None,
    **kwargs,
):
    """kwargs 與 context 只用於 lstm（序列長度等），其他模型會忽略。"""
    if name not in MODEL_NAMES:
        raise ValueError(f"未知模型：{name}（可用：{', '.join(MODEL_NAMES)}）")
    if name == "ensemble":
        return EnsembleModel(seed=seed, avg_uniqueness=avg_uniqueness)
    if name == "lstm":
        from hermes.ml.deep import LSTMClassifier

        if context is None:
            raise ValueError("lstm 模型需要完整特徵表 context 以建立序列")
        return LSTMClassifier(context=context, seed=seed, **kwargs)
    return SkModel(name, seed=seed, avg_uniqueness=avg_uniqueness)
