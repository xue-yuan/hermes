"""以 rich 在終端機輸出表格。"""

from __future__ import annotations

import math

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def pct(x, digits: int = 2, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    s = f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"
    return s


def num(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:,.{digits}f}"


def colored(x, text: str | None = None) -> str:
    """台股慣例：紅色為正、綠色為負。"""
    text = text if text is not None else num(x)
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return text
    return (
        f"[red]{text}[/red]" if x > 0 else f"[green]{text}[/green]" if x < 0 else text
    )


METRIC_ROWS = [
    ("total_return", "總報酬", lambda v: pct(v)),
    ("cagr", "年化報酬 CAGR", lambda v: pct(v)),
    ("ann_vol", "年化波動", lambda v: pct(v)),
    ("sharpe", "Sharpe", lambda v: num(v)),
    ("sortino", "Sortino", lambda v: num(v)),
    ("max_drawdown", "最大回撤", lambda v: pct(v)),
    ("calmar", "Calmar", lambda v: num(v)),
    ("mdd_duration", "最長水下天數", lambda v: f"{v:,}"),
    ("psr", "PSR(SR>0)", lambda v: pct(v, 1)),
    ("exposure", "持倉時間比例", lambda v: pct(v, 1)),
    ("turnover", "年換手（倍）", lambda v: num(v, 1)),
    ("trades", "交易次數", lambda v: f"{v:,}"),
    ("win_rate", "勝率", lambda v: pct(v, 1)),
    ("avg_trade", "平均每筆報酬", lambda v: pct(v)),
    ("profit_factor", "獲利因子", lambda v: num(v)),
    ("avg_hold", "平均持有天數", lambda v: num(v, 1)),
    ("alpha", "Alpha（年化）", lambda v: pct(v)),
    ("beta", "Beta", lambda v: num(v)),
    ("info_ratio", "資訊比率", lambda v: num(v)),
]


def metrics_table(results: list, title: str, include_benchmark: bool = True) -> Table:
    table = Table(title=title, show_lines=False)
    table.add_column("指標", style="bold")
    cols = []
    if include_benchmark and results:
        cols.append(("買進持有", results[0].benchmark_metrics))
    cols += [(r.name, r.metrics) for r in results]
    for name, _ in cols:
        table.add_column(name, justify="right")
    for key, label, fmt in METRIC_ROWS:
        if not any(key in m for _, m in cols):
            continue
        row = []
        for _, m in cols:
            v = m.get(key)
            row.append("—" if v is None else fmt(v))
        table.add_row(label, *row)
    return table


def ranking_table(rows: list[dict], title: str) -> Table:
    table = Table(title=title)
    for col, just in (
        ("策略", "left"),
        ("CAGR", "right"),
        ("Sharpe", "right"),
        ("最大回撤", "right"),
        ("Calmar", "right"),
        ("勝率", "right"),
        ("交易次數", "right"),
        ("持倉比例", "right"),
        ("PSR", "right"),
    ):
        table.add_column(col, justify=just)
    for r in rows:
        m = r["metrics"]
        table.add_row(
            r["name"],
            colored(m["cagr"], pct(m["cagr"])),
            num(m["sharpe"]),
            pct(m["max_drawdown"]),
            num(m["calmar"]),
            pct(m.get("win_rate"), 1),
            f"{m.get('trades', 0):,}",
            pct(m.get("exposure"), 1),
            pct(m["psr"], 1),
        )
    return table


def frame_table(
    df: pd.DataFrame, title: str, formats: dict | None = None, index_name: str = ""
) -> Table:
    formats = formats or {}
    table = Table(title=title)
    table.add_column(index_name)
    for c in df.columns:
        table.add_column(str(c), justify="right")
    for idx, row in df.iterrows():
        label = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)
        cells = []
        for c in df.columns:
            v = row[c]
            f = formats.get(c)
            cells.append(
                f(v) if f else (num(v) if isinstance(v, (int, float)) else str(v))
            )
        table.add_row(label, *cells)
    return table
