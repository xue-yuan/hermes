"""matplotlib 圖表，輸出 PNG。之後做 Web GUI 時可以改用同一份資料畫互動圖。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hermes.backtest.metrics import drawdown
from hermes.services import chart_layout

plt.rcParams["font.sans-serif"] = [
    "PingFang TC",
    "Heiti TC",
    "Noto Sans CJK TC",
    "Microsoft JhengHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

UP, DOWN = "#d62728", "#2ca02c"  # 台股慣例：紅漲綠跌
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
]


def _finish(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _candles(ax, p: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    """畫 K 線，回傳每根 K 棒的顏色（給成交量等子圖沿用）。"""
    color = np.where(p["close"] >= p["open"], UP, DOWN)
    ax.vlines(x, p["low"], p["high"], color=color, linewidth=0.8)
    ax.bar(
        x,
        (p["close"] - p["open"]).abs().clip(lower=p["close"] * 1e-4),
        bottom=np.minimum(p["open"], p["close"]),
        color=color,
        width=0.7,
    )
    return color


def _date_ticks(ax, index: pd.DatetimeIndex) -> None:
    ticks = np.linspace(0, len(index) - 1, min(8, len(index))).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [index[t].strftime("%Y-%m-%d") for t in ticks], rotation=0, fontsize=8
    )


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """連續為 True 的區段（起點、終點的位置）。"""
    runs, start = [], None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def plot_signals(
    values: pd.DataFrame,
    buy_streak: pd.Series,
    sell_streak: pd.Series,
    strategy,
    title: str,
    path: Path,
) -> Path:
    """歷史買賣點：新觸發的那天用箭頭標示，條件連續成立的期間加上底色。

    均線、布林通道這類跟股價同單位的指標畫在價格圖上，其他指標各自一個子圖，
    並畫出策略用到的門檻（實線）和設定檔寫的參考線（虛線）。
    """
    layout = chart_layout(strategy)
    overlays, panels, used = layout.overlays, layout.panels, layout.thresholds
    refs = layout.reference_lines

    fig, axes = plt.subplots(
        1 + len(panels),
        1,
        figsize=(13, 5.5 + 2.2 * len(panels)),
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": [3] + [1.2] * len(panels)},
    )
    axes = axes[:, 0]
    x = np.arange(len(values))
    ax = axes[0]
    _candles(ax, values, x)
    for k, col in enumerate(overlays):
        ax.plot(x, values[col], label=col, color=COLORS[k % len(COLORS)], linewidth=1)
    for lvl in used.get("price", ()):
        ax.axhline(lvl, color=COLORS[1], linewidth=0.9, label=f"門檻 {lvl:g}")
    for lvl in refs.get("price", ()):
        ax.axhline(
            lvl, color="gray", linewidth=0.8, linestyle="--", label=f"參考線 {lvl:g}"
        )

    for s, color, marker, y, label in (
        (buy_streak, UP, "^", values["low"] * 0.98, "買進"),
        (sell_streak, DOWN, "v", values["high"] * 1.02, "賣出"),
    ):
        starts = (s == 1).to_numpy()
        if starts.any():
            ax.scatter(
                x[starts],
                y.to_numpy()[starts],
                marker=marker,
                color=color,
                s=70,
                zorder=5,
                edgecolors="black",
                linewidths=0.4,
                label=f"{label}訊號（新觸發）",
            )
        long_runs = [(a, b) for a, b in _runs((s > 0).to_numpy()) if b > a]
        for k, (a, b) in enumerate(long_runs):
            for axis in axes:
                axis.axvspan(
                    a - 0.5,
                    b + 0.5,
                    color=color,
                    alpha=0.08,
                    label=f"{label}條件持續成立" if (k == 0 and axis is ax) else None,
                )
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    for axis, (name, cols) in zip(axes[1:], panels.items()):
        for k, col in enumerate(cols):
            axis.plot(
                x, values[col], label=col, color=COLORS[k % len(COLORS)], linewidth=1
            )
        for lvl in used.get(name, ()):
            axis.axhline(lvl, color=COLORS[1], linewidth=0.9, label=f"門檻 {lvl:g}")
        for lvl in refs.get(name, ()):
            axis.axhline(
                lvl, color="gray", linewidth=0.8, linestyle="--", label=f"參考線 {lvl:g}"
            )
        axis.set_ylabel(name)
        axis.legend(loc="upper left", fontsize=8)
        axis.grid(alpha=0.3)
    _date_ticks(axes[-1], values.index)
    return _finish(fig, path)


def plot_backtests(results: list, title: str, path: Path) -> Path:
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1]}
    )
    bench = results[0].benchmark_equity
    axes[0].plot(
        bench.index, bench, color="black", linewidth=1.2, label="買進持有", alpha=0.7
    )
    for k, r in enumerate(results):
        c = COLORS[k % len(COLORS)]
        axes[0].plot(r.equity.index, r.equity, color=c, linewidth=1.2, label=r.name)
        dd = drawdown(r.equity)
        axes[1].plot(dd.index, dd * 100, color=c, linewidth=0.9)
        axes[2].plot(r.position.index, r.position, color=c, linewidth=0.8, alpha=0.8)
    bdd = drawdown(bench)
    axes[1].fill_between(bdd.index, bdd * 100, 0, color="black", alpha=0.08)
    axes[0].set_yscale("log")
    axes[0].set_title(title)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("淨值（對數）")
    axes[1].set_ylabel("回撤 %")
    axes[2].set_ylabel("部位")
    for ax in axes:
        ax.grid(alpha=0.3)
    return _finish(fig, path)


def plot_ml(
    prices: pd.DataFrame, proba: pd.Series, result, title: str, path: Path
) -> Path:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2, 1]},
    )
    axes[0].plot(result.equity.index, result.equity, color=COLORS[0], label=result.name)
    axes[0].plot(
        result.benchmark_equity.index,
        result.benchmark_equity,
        color="black",
        alpha=0.6,
        label="買進持有",
    )
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title(title)
    p = proba.dropna()
    axes[1].plot(p.index, p, color=COLORS[2], linewidth=0.7)
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
    axes[1].set_ylabel("上漲機率")
    axes[2].plot(result.position.index, result.position, color=COLORS[1], linewidth=0.8)
    axes[2].set_ylabel("部位")
    for ax in axes:
        ax.grid(alpha=0.3)
    return _finish(fig, path)
