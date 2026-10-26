"""網頁的資料 API：全部是 GET，只讀本機資料，不下載也不寫入。

每個端點都只呼叫 services，再把結果轉成 JSON；數字都在 Python 算好，前端只負責顯示。
用實際成交價判斷；adjusted=true 是還原模式，用還原價格判斷。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import Path as PathParam

from hermes.api import serialize as js
from hermes.api.auth import current_user
from hermes.backtest import BacktestResult, CostModel
from hermes.chips import ChipReport
from hermes.config import indicators_dir, strategies_dir
from hermes.data import DataLoader
from hermes.services import (
    SORT_KEYS,
    MarketDataReader,
    StrategyNotFoundError,
    chart_layout,
    chip_report,
    column_labels,
    compare_strategies,
    gap_warnings,
    indicator_label,
    indicator_set_files,
    indicator_snapshot,
    list_rule_strategies,
    operand_labels,
    rule_backtest,
    screen,
    signal_history,
    strategy_files,
)
from hermes.strategies import (
    REGISTRY,
    RuleStrategy,
    StrategyConfigError,
    describe,
    operand_label,
    operand_text,
    rule_operands,
)
from hermes.strategies.rules import load_indicator_set
from hermes.technical import CATALOG

router = APIRouter(prefix="/api", dependencies=[Depends(current_user)])

StockId = Annotated[str, PathParam(pattern=r"^[0-9A-Za-z]{4,6}$")]
Execution = Literal["open", "close"]
# 還原模式：用還原價格判斷（預設用實際成交價）
Adjusted = Annotated[bool, Query()]


def get_loader() -> DataLoader:
    return DataLoader()


def get_reader(loader: DataLoader = Depends(get_loader)):
    reader = MarketDataReader(loader)
    try:
        yield reader
    finally:
        reader.close()


def get_strategies_dir() -> Path:
    return strategies_dir()


def get_indicators_dir() -> Path:
    return indicators_dir()


def _rule_strategy(ref: str, directory: Path) -> str:
    """只接受 strategies/ 裡的檔名，不接受路徑，也不接受和內建策略同名的檔案。"""
    if ref not in {p.stem for p in strategy_files(directory)}:
        raise StrategyNotFoundError(f"找不到策略 {ref}")
    if ref in REGISTRY:
        raise StrategyConfigError(
            f"{ref} 同時是策略設定檔和內建策略的名稱，請把設定檔改名"
        )
    return ref


def _strategy_info(ref: str, st: RuleStrategy) -> dict:
    return {
        "id": ref,
        "name": st.name,
        "description": st.description,
        "buy": describe(st.buy, operand_labels(st)),
        "sell": describe(st.sell, operand_labels(st)),
        "forming": describe(st.forming, operand_labels(st)) if st.forming else None,
        "indicators": [
            {"name": n, "label": indicator_label(c)} for n, c in st.indicators.items()
        ],
    }


def _name(reader: MarketDataReader, stock_id: str) -> str:
    return reader.stock_name(stock_id)


# ---------------------------------------------------------------------------
@router.get("/overview")
def overview(
    reader: MarketDataReader = Depends(get_reader),
    sdir: Path = Depends(get_strategies_dir),
    idir: Path = Depends(get_indicators_dir),
) -> dict:
    """自選股、策略設定檔、指標組的清單。"""
    strategies = []
    for path, item in list_rule_strategies(sdir):
        if isinstance(item, StrategyConfigError):
            strategies.append({"id": path.stem, "error": str(item)})
        else:
            strategies.append({**_strategy_info(path.stem, item), "error": None})
    sets = []
    for path in indicator_set_files(idir):
        try:
            s = load_indicator_set(path)
            sets.append(
                {"id": path.stem, "name": s.name, "description": s.description, "error": None}
            )
        except StrategyConfigError as exc:
            sets.append({"id": path.stem, "error": str(exc)})
    return {
        "watchlist": [
            {
                "stock_id": item.stock_id,
                "name": item.name,
                "added_at": js.value(item.added_at),
                "last_price_date": js.value(reader.last_price_date(item.stock_id)),
            }
            for item in reader.list_watchlist()
        ],
        "strategies": strategies,
        "indicator_sets": sets,
        "sort_keys": list(SORT_KEYS),
    }


@router.get("/screen")
def screen_report(
    strategy: Annotated[list[str] | None, Query()] = None,
    adjusted: Adjusted = False,
    reader: MarketDataReader = Depends(get_reader),
    sdir: Path = Depends(get_strategies_dir),
) -> dict:
    """每日選股報告：自選股 × 策略。"""
    refs = [_rule_strategy(s, sdir) for s in strategy] if strategy else None
    ids = [item.stock_id for item in reader.list_watchlist()]
    report = screen(ids, refs, reader.loader, sdir, adjusted)

    def base(r) -> dict:
        return {
            "stock_id": r.stock_id,
            "name": r.name,
            "strategy_id": r.strategy_id,
            "strategy": r.strategy.name,
        }

    return {
        "adjusted": adjusted,
        "latest_date": js.value(report.latest_date),
        "stocks": len({r.stock_id for r in report.rows}),
        "strategies": len({r.strategy_id for r in report.rows}),
        "stale": [{"stock_id": k, "date": js.value(v)} for k, v in report.stale().items()],
        "triggered": [
            {**base(r), "side": side, "close": r.close} for r, side in report.triggered_today()
        ],
        "ongoing": [
            {**base(r), "side": side, "days": days}
            for r, side, days in sorted(report.ongoing(), key=lambda x: x[2])
        ],
        # 有沒有策略寫了形成中的條件；都沒有時網頁不顯示這一段
        "has_forming": any(r.strategy.forming for r in report.rows),
        "forming": [
            {
                **base(r),
                "days": days,
                "values": [
                    {
                        "label": operand_text(o, operand_labels(r.strategy)),
                        "value": js.value(r.values[operand_label(o)]),
                    }
                    for o in rule_operands(r.strategy.forming)
                ],
            }
            for r, days in report.forming()
        ],
        "warnings": [{"who": w, "message": m} for w, m in report.warnings],
        "errors": [{"who": w, "message": m} for w, m in report.errors],
    }


# ---------------------------------------------------------------------------
@router.get("/stocks/{stock_id}")
def stock(stock_id: StockId, reader: MarketDataReader = Depends(get_reader)) -> dict:
    prices = reader.price_history(stock_id, adjust=False)
    return {
        "stock_id": stock_id,
        "name": _name(reader, stock_id),
        "first_date": js.value(prices.index[0]),
        "last_date": js.value(prices.index[-1]),
        "close": js.value(prices["close"].iloc[-1]),
        "in_watchlist": stock_id in {i.stock_id for i in reader.list_watchlist()},
    }


def _period(days: pd.DatetimeIndex, start: date | None, end: date | None) -> tuple[date, date]:
    """表格、統計和圖一開始顯示的區間：預設最近一年，和 CLI 相同。"""
    last_day = days[-1]
    s = start or (last_day - pd.DateOffset(years=1)).date()
    e = end or last_day.date()
    if s > e:
        raise HTTPException(422, "起始日不能晚於結束日")
    if not ((days >= pd.Timestamp(s)) & (days <= pd.Timestamp(e))).any():
        raise HTTPException(422, f"{s} ～ {e} 之間沒有交易資料")
    return s, e


@router.get("/stocks/{stock_id}/prices")
def prices(
    stock_id: StockId,
    start: date | None = None,
    end: date | None = None,
    adjusted: Adjusted = False,
    reader: MarketDataReader = Depends(get_reader),
) -> dict:
    """股價圖：全部歷史的日 K 和成交量（實際成交價；還原模式是還原價格），不用選策略；
    range 是圖一開始顯示的區間，和 /signals 的表格相同。"""
    full = reader.price_history(stock_id, adjust=adjusted)
    s, e = _period(full.index, start, end)
    shown = full.loc[pd.Timestamp(s) : pd.Timestamp(e)]
    return {
        "stock_id": stock_id,
        "adjusted": adjusted,
        "range": {"start": js.value(shown.index[0]), "end": js.value(shown.index[-1])},
        "series": {
            "date": js.days(full.index),
            **{c: js.values(full[c]) for c in ("open", "high", "low", "close")},
            "volume_lots": js.values(full["volume"] / 1000),  # 成交量（張）
        },
    }


@router.get("/stocks/{stock_id}/signals")
def signals(
    stock_id: StockId,
    strategy: str,
    start: date | None = None,
    end: date | None = None,
    adjusted: Adjusted = False,
    reader: MarketDataReader = Depends(get_reader),
    sdir: Path = Depends(get_strategies_dir),
) -> dict:
    """歷史買賣點：圖用全部歷史，表格和統計用 start ～ end（預設最近一年，和 CLI 相同）。"""
    ref = _rule_strategy(strategy, sdir)
    hist = signal_history(stock_id, ref, adjusted, reader.loader, sdir)
    full = hist.table(every_day=True)
    last_day = full.index[-1]
    s, e = _period(full.index, start, end)
    period = hist.table(s, e, every_day=True)
    cols = hist.indicator_columns
    labels = operand_labels(hist.strategy)
    layout = chart_layout(hist.strategy)
    rows = period[period["buy"] | period["sell"]]
    streaks = ["buy_streak", "sell_streak"]
    latest = full.iloc[-1]
    last_buy, last_sell = hist.last_trigger("buy"), hist.last_trigger("sell")
    return {
        "stock_id": stock_id,
        "name": _name(reader, stock_id),
        "strategy": _strategy_info(ref, hist.strategy),
        "adjusted": adjusted,
        "warnings": gap_warnings(reader.loader, stock_id, adjusted) + hist.warnings,
        "range": {"start": js.value(period.index[0]), "end": js.value(period.index[-1])},
        "layout": {
            "overlays": layout.overlays,
            "panels": layout.panels,
            # 圖上的成交量是張，策略裡的成交量門檻是股，換成張才畫得在同一個刻度
            "thresholds": {
                k: [v / 1000 for v in lv] if k == "volume" else lv
                for k, lv in layout.thresholds.items()
            },
            "reference_lines": layout.reference_lines,
        },
        "indicator_columns": cols,
        "column_labels": {c: labels.get(c, c) for c in cols},
        "series": {
            "date": js.days(full.index),
            **{c: js.values(full[c]) for c in ("open", "high", "low", "close")},
            "volume_lots": js.values(full["volume"] / 1000),  # 成交量（張）
            **{c: js.values(full[c]) for c in cols},
            **{c: js.values(full[c]) for c in streaks},
        },
        "rows": [
            {
                "date": js.value(d),
                "close": js.value(r["close"]),
                "values": {c: js.value(r[c]) for c in cols},
                **{c: js.value(r[c]) for c in streaks},
            }
            for d, r in rows.iloc[::-1].iterrows()
        ],
        "summary": {
            "buy_signals": int((period["buy_streak"] == 1).sum()),
            "buy_days": int(period["buy"].sum()),
            "sell_signals": int((period["sell_streak"] == 1).sum()),
            "sell_days": int(period["sell"].sum()),
        },
        "latest": {
            "date": js.value(last_day),
            **{c: js.value(latest[c]) for c in streaks},
            "last_buy": js.value(last_buy),
            "last_sell": js.value(last_sell),
        },
    }


@router.get("/stocks/{stock_id}/tech")
def tech(
    stock_id: StockId,
    indicator_set: Annotated[str, Query(alias="set")] = "common",
    day: Annotated[date | None, Query(alias="date")] = None,
    adjusted: Adjusted = False,
    reader: MarketDataReader = Depends(get_reader),
    idir: Path = Depends(get_indicators_dir),
) -> dict:
    """指標組在某一天的數值（實際成交價；還原模式是還原價格）。"""
    if indicator_set not in {p.stem for p in indicator_set_files(idir)}:
        raise HTTPException(404, f"找不到指標組 {indicator_set}")
    snap = indicator_snapshot(stock_id, indicator_set, day, reader.loader, idir, adjusted)
    rows = []
    for name, cfg in snap.indicator_set.indicators.items():
        spec = CATALOG[cfg.type]
        labels = column_labels({name: cfg})
        for line in spec.outputs or (None,):
            col = f"{name}.{line}" if line else name
            rows.append(
                {
                    "column": col,
                    "label": labels[col],
                    # 多條線的指標：網頁先列一次指標名稱（indicator），下面每列只寫線的名稱（line）
                    "indicator": indicator_label(cfg),
                    "line": spec.line_label(line) if line else None,
                    "value": js.value(snap.values[col]),
                    "unavailable": bool(snap.unavailable[col]),
                }
            )
    return {
        "stock_id": stock_id,
        "name": _name(reader, stock_id),
        "set": {
            "id": indicator_set,
            "name": snap.indicator_set.name,
            "description": snap.indicator_set.description,
        },
        "date": js.value(snap.date),
        "adjusted": adjusted,
        "close": snap.close,
        "rows": rows,
        "warnings": gap_warnings(reader.loader, stock_id, adjusted) + snap.warnings,
    }


def _chips(rep: ChipReport, windows: list[int]) -> dict:
    def changes(info: dict, prefix: str) -> dict:
        return {str(w): js.value(info.get(f"{prefix}{w}")) for w in windows}

    fr, mg = rep.foreign_ratio, rep.margin
    return {
        "as_of": js.value(rep.as_of),
        "windows": windows,
        "flows": [
            {"who": who, "values": js.values(row)} for who, row in rep.flows.iterrows()
        ],
        "flow_columns": [str(c) for c in rep.flows.columns],
        "streaks": {k: int(v) for k, v in rep.streaks.items()},
        "recent_columns": [str(c) for c in rep.recent.columns],
        "recent": [
            {"date": js.value(d), "values": js.values(row)}
            for d, row in rep.recent.iloc[::-1].iterrows()
        ],
        "foreign_ratio": {
            "date": js.value(fr["date"]),
            "now": js.value(fr["now"]),
            "changes": changes(fr, "chg_"),
        }
        if fr
        else None,
        "margin": {
            **{
                k: js.value(mg[k])
                for k in (
                    "date",
                    "margin_balance",
                    "margin_usage",
                    "short_balance",
                    "short_margin_ratio",
                )
            },
            "changes": changes(mg, "margin_chg_"),
        }
        if mg
        else None,
    }


@router.get("/stocks/{stock_id}/chips")
def chips(
    stock_id: StockId,
    window: Annotated[list[int] | None, Query()] = None,
    days: Annotated[int, Query(gt=0, le=250)] = 10,
    reader: MarketDataReader = Depends(get_reader),
) -> dict:
    """籌碼摘要；N 日由使用者指定。"""
    windows = list(dict.fromkeys(window or []))
    try:
        rep = chip_report(stock_id, windows, days, reader.loader)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    return {"stock_id": stock_id, "name": _name(reader, stock_id), **_chips(rep, windows)}


# ---------------------------------------------------------------------------
def _costs(fee_discount: float, slippage: float, tax: float) -> CostModel:
    return CostModel(discount=fee_discount, slippage=slippage, tax=tax)


def _cost_info(costs: CostModel) -> dict:
    return {
        "commission": costs.commission,
        "discount": costs.discount,
        "tax": costs.tax,
        "slippage": costs.slippage,
    }


def _cost_warnings(stock_id: str, costs: CostModel) -> list[str]:
    if stock_id.startswith("00") and costs.tax == CostModel().tax:
        return [f"{stock_id} 是 ETF，證交稅率和股票不同，請另外指定（目前用股票的 {costs.tax:.1%}）"]
    return []


def _curves(results: list[tuple[str, BacktestResult]], benchmark: pd.Series) -> dict:
    """淨值、回撤、部位；買進持有的淨值放在 benchmark。"""
    index = benchmark.index

    def dd(equity: pd.Series) -> pd.Series:
        return equity / equity.cummax() - 1

    return {
        "date": js.days(index),
        "benchmark": {"equity": js.values(benchmark), "drawdown": js.values(dd(benchmark))},
        "series": [
            {
                "id": key,
                "name": r.name,
                "equity": js.values(r.equity.reindex(index)),
                "drawdown": js.values(dd(r.equity).reindex(index)),
                "position": js.values(r.position.reindex(index)),
            }
            for key, r in results
        ],
    }


Fee = Annotated[float, Query(gt=0, le=1)]
Rate = Annotated[float, Query(ge=0, lt=0.1)]


@router.get("/stocks/{stock_id}/backtest")
def backtest(
    stock_id: StockId,
    strategy: str,
    start: date | None = None,
    end: date | None = None,
    execution: Execution = "open",
    fee_discount: Fee = 1.0,
    slippage: Rate = 0.0,
    tax: Rate = 0.003,
    adjusted: Adjusted = False,
    reader: MarketDataReader = Depends(get_reader),
    sdir: Path = Depends(get_strategies_dir),
) -> dict:
    """設定檔策略的回測：訊號用實際成交價判斷（還原模式用還原價格），報酬用還原價格計算（包含股利）。"""
    ref = _rule_strategy(strategy, sdir)
    costs = _costs(fee_discount, slippage, tax)
    hist = signal_history(stock_id, ref, adjusted, reader.loader, sdir)
    rb = rule_backtest(hist, start, end, costs, execution, reader.loader)
    res = rb.result
    trades = rb.trades()
    latest = None
    if end is None:
        last, holding, action = rb.latest()
        latest = {"date": js.value(last), "holding": holding, "action": action}
    return {
        "stock_id": stock_id,
        "name": _name(reader, stock_id),
        "strategy": _strategy_info(ref, rb.strategy),
        "start": js.value(rb.start),
        "end": js.value(res.returns.index[-1]),
        "warmup_end": js.value(rb.warmup_end),
        "start_given": start is not None,
        "adjusted": adjusted,
        "execution": execution,
        "costs": _cost_info(costs),
        "metrics": [
            {"id": "benchmark", "name": "買進持有", "metrics": js.mapping(res.benchmark_metrics)},
            {"id": "strategy", "name": res.name, "metrics": js.mapping(res.metrics)},
        ],
        "curves": _curves([("strategy", res)], res.benchmark_equity),
        "trades": js.records(trades.iloc[::-1]),
        "latest": latest,
        "warnings": gap_warnings(reader.loader, stock_id, adjusted, returns=True)
        + hist.warnings
        + _cost_warnings(stock_id, costs),
    }


@router.get("/stocks/{stock_id}/compare")
def compare(
    stock_id: StockId,
    strategy: Annotated[list[str], Query()],
    start: date | None = None,
    end: date | None = None,
    execution: Execution = "open",
    fee_discount: Fee = 1.0,
    slippage: Rate = 0.0,
    tax: Rate = 0.003,
    sort: Literal["sharpe", "cagr", "calmar", "max_drawdown"] = "sharpe",
    adjusted: Adjusted = False,
    reader: MarketDataReader = Depends(get_reader),
    sdir: Path = Depends(get_strategies_dir),
) -> dict:
    """並排比較設定檔策略，附上買進持有。機器學習策略要長時間訓練，網頁不提供。"""
    refs = [_rule_strategy(s, sdir) for s in dict.fromkeys(strategy)]
    costs = _costs(fee_discount, slippage, tax)
    cmp = compare_strategies(
        stock_id, refs, start, end, costs, execution, sort, reader.loader, sdir,
        adjusted=adjusted,
    )
    bench = cmp.benchmark
    return {
        "stock_id": stock_id,
        "name": _name(reader, stock_id),
        "sort": sort,
        "adjusted": adjusted,
        "start": js.value(bench.returns.index[0]),
        "end": js.value(bench.returns.index[-1]),
        "execution": execution,
        "costs": _cost_info(costs),
        "rows": [
            {"id": i, "name": r.name, "metrics": js.mapping(r.metrics)}
            for i, r in zip(cmp.ids, cmp.results)
        ],
        "benchmark": {"name": "買進持有", "metrics": js.mapping(bench.metrics)},
        "curves": _curves(list(zip(cmp.ids, cmp.results)), bench.equity),
        "warnings": gap_warnings(reader.loader, stock_id, adjusted, returns=True)
        + cmp.warnings
        + _cost_warnings(stock_id, costs),
    }
