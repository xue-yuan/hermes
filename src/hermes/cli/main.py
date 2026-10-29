"""hermes 命令列介面。"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from hermes import technical as ta
from hermes.backtest import CostModel, run_backtest
from hermes.cli.console import (
    colored,
    console,
    frame_table,
    metrics_table,
    num,
    pct,
    ranking_table,
)
from hermes.config import indicators_dir, reports_dir, strategies_dir
from hermes.data import DATASET_LABELS, DataError, DataLoader
from hermes.services import (
    PRICE_NAMES,
    FetchReport,
    IndicatorSetNotFoundError,
    MarketDataService,
    RuleBacktest,
    StrategyNotFoundError,
    adjusted_volume_note,
    builtin_backtest,
    chip_report,
    compare_strategies,
    gap_warnings,
    indicator_label,
    indicator_set_files,
    indicator_snapshot,
    is_rule_strategy,
    list_rule_strategies,
    rule_backtest,
    signal_history,
    strategy_context,
)
from hermes.services import screen as screen_stocks
from hermes.strategies import (
    DATA_FIELDS,
    REGISTRY,
    StrategyConfigError,
    describe,
    get_strategy,
    operand_label,
    parse_params,
    rule_operands,
)

app = typer.Typer(
    help="hermes：台股分析工具（策略設定檔的買賣點、每日選股、回測；籌碼面；機器學習）",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
watchlist_app = typer.Typer(
    help="自選股清單：hermes update 會更新清單中的股票。", no_args_is_help=True
)
app.add_typer(watchlist_app, name="watchlist")

DISCLAIMER = (
    "[dim]※ 本工具僅供研究與教育用途，回測績效不代表未來報酬，不構成投資建議。[/dim]"
)
ADJUST_HELP = "還原模式：指標和買賣判斷改用還原價格算（預設用實際成交價）"

def _loader() -> DataLoader:
    """分析指令只讀本機資料；下載請用 fetch 或 update。"""
    return DataLoader()


def _gap_warnings(stock: str, adjusted: bool, returns: bool = False) -> None:
    """沒有官方紀錄的跳空：不估算還原，只提醒。"""
    for msg in gap_warnings(_loader(), stock, adjusted, returns):
        console.print(f"[yellow]⚠ {stock} {msg}（hermes verify 查看）[/yellow]")


def _field_warnings(stock: str, warnings: list[str]) -> None:
    """籌碼、本益比缺資料：不補值，只提醒。"""
    for msg in warnings:
        console.print(f"[yellow]⚠ {stock} {msg}[/yellow]")


def _name(loader: DataLoader, stock: str) -> str:
    try:
        return f"{stock} {loader.info(stock)['stock_name']}"
    except DataError:
        return stock


@contextmanager
def _progress(desc: str):
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task(desc, total=None)

        def update(done: int, total: int):
            prog.update(task, completed=done, total=total)

        yield update


@contextmanager
def _ml_progress():
    """機器學習訓練的進度條：services 回報進度時才出現（設定檔策略不會用到）。"""
    prog = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    tasks: dict[str, int] = {}

    def update(title: str, done: int, total: int) -> None:
        if not tasks:
            prog.start()
        if title not in tasks:
            tasks[title] = prog.add_task(f"{title}：walk-forward 訓練中", total=total)
        prog.update(tasks[title], completed=done, total=total)

    try:
        yield update
    finally:
        if tasks:
            prog.stop()


def _costs(fee_discount: float, slippage: float, tax: float) -> CostModel:
    try:
        return CostModel(discount=fee_discount, slippage=slippage, tax=tax)
    except ValueError as exc:
        _fail(str(exc))


def _etf_tax_note(stock: str, tax: float) -> None:
    if stock.startswith("00") and tax == CostModel().tax:
        console.print(
            f"[yellow]{stock} 是 ETF，證交稅率和股票不同，請用 --tax 指定（目前用股票的 {tax:.1%}）[/yellow]"
        )


def _fail(msg: str):
    console.print(f"[red]錯誤：{msg}[/red]")
    raise typer.Exit(1)


def _date(text: str | None, option: str) -> date | None:
    """YYYY-MM-DD 轉成日期；格式不對時顯示錯誤。"""
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        _fail(f"{option} 要是 YYYY-MM-DD 格式的日期，例如 2024-01-02（收到 {text}）")


def _dates(start: str | None, end: str | None) -> tuple[date | None, date | None]:
    """--start、--end 轉成日期；起始日不能晚於結束日。"""
    s, e = _date(start, "--start"), _date(end, "--end")
    if s and e and s > e:
        _fail("--start 不能晚於 --end")
    return s, e


def _at_least_one(value: int, option: str) -> None:
    if value < 1:
        _fail(f"{option} 要是正整數")


# ---------------------------------------------------------------------------
# 資料：下載、每日更新、自選股、查看。只有 fetch、update、watchlist add 會連網。


def _fetch_with_progress(
    stock_ids: list[str], run: Callable[[Callable[[str, str], None]], FetchReport]
) -> FetchReport:
    order = {sid: i + 1 for i, sid in enumerate(dict.fromkeys(stock_ids))}
    with console.status("下載中…") as status:

        def progress(sid: str, dataset: str) -> None:
            pos = f"（{order[sid]}/{len(order)}）" if sid in order else ""
            status.update(f"{sid}{pos} 下載{DATASET_LABELS.get(dataset, dataset)}…")

        return run(progress)


def _print_fetch_report(svc: MarketDataService, report: FetchReport) -> None:
    if report.results:
        t = Table(title="下載結果")
        for c in ("代號", "名稱", "股價資料區間", "筆數", "本次新增"):
            t.add_column(c, justify="right" if c in ("筆數", "本次新增") else "left")
        for r in report.results:
            span = f"{r.price_start} ～ {r.price_end}" if r.price_start else "—"
            t.add_row(
                r.stock_id,
                svc.stock_name(r.stock_id),
                span,
                f"{r.price_rows:,}",
                f"{r.added_price_rows:+,}",
            )
        console.print(t)
    for r in report.results:
        for msg in r.missing_share_ratios:
            console.print(
                f"[yellow]⚠ {r.stock_id} {msg}；還原價格在這天以前的成交量算不出來[/yellow]"
            )
    for sid, msg in report.failed.items():
        console.print(f"[red]{sid}：{msg}[/red]")
    if report.skipped:
        console.print(
            f"[yellow]請求次數已用完，以下股票尚未更新：{'、'.join(report.skipped)}。"
            f"約一小時後再執行一次，會從各檔最後一筆接續下載。[/yellow]"
        )
    if report.benchmark_rows is not None:
        console.print(
            f"[dim]大盤指數（TAIEX）共 {report.benchmark_rows:,} 筆。資料位置：{svc.sync.cache.root}[/dim]"
        )


@app.command()
def fetch(
    stocks: Annotated[
        list[str], typer.Argument(help="股票代號，可一次多個，例如 2330 2317")
    ],
    start: Annotated[
        str | None,
        typer.Option(
            help="重新下載的起始日 YYYY-MM-DD；不指定時從本機最後一筆接續，沒有資料就下載完整歷史",
        ),
    ] = None,
    end: Annotated[
        str | None, typer.Option(help="重新下載的結束日，預設到今天")
    ] = None,
):
    """下載個股的所有資料（股價、籌碼、本益比、除權息…），並一併更新大盤指數。只接受上市股票。"""
    s, e = _dates(start, end)
    svc = MarketDataService()
    report = _fetch_with_progress(
        stocks, lambda cb: svc.fetch(stocks, s, e, on_progress=cb)
    )
    _print_fetch_report(svc, report)
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def update():
    """依自選股清單接續下載所有資料，給每日排程使用。"""
    svc = MarketDataService()
    ids = [item.stock_id for item in svc.list_watchlist()]
    if not ids:
        _fail("自選股清單是空的，請先用 hermes watchlist add 2330 加入股票")
    report = _fetch_with_progress(ids, lambda cb: svc.update(on_progress=cb))
    _print_fetch_report(svc, report)
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def verify(
    stocks: Annotated[
        list[str] | None, typer.Argument(help="股票代號；不指定時檢查自選股清單")
    ] = None,
    months: Annotated[
        int,
        typer.Option(
            help="股價、本益比和證交所比對最近幾個月（每檔每個月 2 次請求，間隔 3 秒）"
        ),
    ] = 3,
    chip_days: Annotated[
        int,
        typer.Option(
            help="籌碼和證交所比對最近幾個交易日（每天 3 次請求，多檔股票共用）"
        ),
    ] = 3,
):
    """檢查本機資料是否正確：股價、籌碼、本益比和證交所官方資料比對，還原權息、升降單位與漲跌停。"""
    from hermes.data.twse import TwseClient
    from hermes.services.verify import verify_stock

    _at_least_one(months, "--months")
    ids = stocks or [item.stock_id for item in MarketDataService().list_watchlist()]
    if not ids:
        _fail("請指定股票代號，或先用 hermes watchlist add 加入自選股")
    failed = False
    mark = {
        "ok": "[green]通過[/green]",
        "fail": "[red]有問題[/red]",
        "warn": "[yellow]注意[/yellow]",
        "skip": "[dim]略過[/dim]",
    }
    twse = TwseClient()
    for sid in ids:
        try:
            with console.status(f"檢查 {sid}…"):
                checks = verify_stock(sid, months, chip_days, twse=twse)
        except DataError as exc:
            console.print(f"[red]{sid}：{exc}[/red]")
            failed = True
            continue
        t = Table(title=f"{_name(_loader(), sid)} 資料檢查")
        for c in ("項目", "結果", "說明"):
            t.add_column(c)
        for c in checks:
            t.add_row(c.name, mark[c.status], c.summary)
        console.print(t)
        for c in checks:
            color = "yellow" if c.status == "warn" else "red"
            for line in c.problems[:10]:
                console.print(f"  [{color}]{c.name}：{line}[/{color}]")
            if len(c.problems) > 10:
                console.print(
                    f"  [{color}]{c.name}：另有 {len(c.problems) - 10} 筆[/{color}]"
                )
        failed |= any(c.status == "fail" for c in checks)
    if failed:
        raise typer.Exit(1)


@app.command()
def show(
    stock: Annotated[str, typer.Argument(help="股票代號")],
    last: Annotated[int, typer.Option(help="顯示最近幾筆")] = 20,
    adjust: Annotated[
        bool, typer.Option("--adjust", help="顯示還原權息後的價格（預設為實際成交價）")
    ] = False,
):
    """查看本機的日 K 資料。"""
    _at_least_one(last, "--last")
    svc = MarketDataService()
    try:
        prices = svc.price_history(stock, adjust=adjust)
    except DataError as exc:
        _fail(str(exc))
    kind = "還原權息" if adjust else "實際成交價，未還原權息"
    t = Table(
        title=f"{stock} {svc.stock_name(stock)}（共 {len(prices):,} 筆，"
        f"{prices.index[0]:%Y-%m-%d} ～ {prices.index[-1]:%Y-%m-%d}，{kind}）"
    )
    for c in ("日期", "開盤", "最高", "最低", "收盤", "成交量(張)"):
        t.add_column(c, justify="right")
    for d, row in prices.iloc[-last:].iterrows():
        t.add_row(
            f"{d:%Y-%m-%d}",
            num(row["open"]),
            num(row["high"]),
            num(row["low"]),
            num(row["close"]),
            "無法計算" if pd.isna(row["volume"]) else f"{row['volume'] / 1000:,.0f}",
        )
    console.print(t)
    if prices["volume"].iloc[-last:].isna().any():
        console.print(
            f"[yellow]⚠ {stock} {adjusted_volume_note(svc.loader, stock)}[/yellow]"
        )


@app.command("screen")
def screen_report(
    stocks: Annotated[
        list[str] | None, typer.Argument(help="股票代號；不指定時用自選股清單")
    ] = None,
    strategy: Annotated[
        list[str] | None,
        typer.Option(
            "-s", "--strategy", help="策略設定檔，可重複；不指定時用 strategies/ 裡全部"
        ),
    ] = None,
    adjust: Annotated[bool, typer.Option("--adjust", help=ADJUST_HELP)] = False,
):
    """每日選股報告：新觸發的訊號、條件仍成立的訊號、形成中。收盤後執行，決定下一個交易日開盤要不要買賣。"""
    ids = stocks or [item.stock_id for item in MarketDataService().list_watchlist()]
    if not ids:
        _fail("請指定股票代號，或先用 hermes watchlist add 加入自選股")
    with console.status("計算中…"):
        report = screen_stocks(ids, strategy or None, adjusted=adjust)
    if not report.rows:
        for who, msg in report.errors:
            console.print(f"[red]{who}：{msg}[/red]")
        _fail("沒有可以計算的股票和策略")

    strategies = {r.strategy_id for r in report.rows}
    stocks_done = {r.stock_id for r in report.rows}
    console.print(
        Panel.fit(
            f"資料日期：[bold]{report.latest_date:%Y-%m-%d}[/bold]　{len(stocks_done)} 檔股票 × {len(strategies)} 個策略\n"
            f"價格：{PRICE_NAMES[adjust]}",
            title="每日選股報告",
        )
    )
    for sid, d in report.stale().items():
        console.print(
            f"[yellow]{sid} 的資料只到 {d:%Y-%m-%d}，比其他股票舊，請先執行 hermes update[/yellow]"
        )
    for who, msg in report.warnings:
        console.print(f"[yellow]⚠ {who}：{msg}[/yellow]")

    label = {
        "buy": "[bold red]買進[/bold red]",
        "sell": "[bold green]賣出[/bold green]",
    }
    new = report.triggered_today()
    if new:
        t = Table(title=f"① 新觸發（{report.latest_date:%Y-%m-%d}）")
        for c in ("代號", "名稱", "策略", "訊號", "收盤"):
            t.add_column(c, justify="right" if c == "收盤" else "left")
        for r, side in new:
            t.add_row(
                r.stock_id,
                r.name,
                r.strategy.name,
                label[side],
                _price(r.close),
            )
        console.print(t)
    else:
        console.print("① 這天沒有新觸發的訊號。")

    ongoing = report.ongoing()
    if ongoing:
        t = Table(title="② 條件仍成立")
        for c in ("代號", "名稱", "策略", "訊號", "已持續"):
            t.add_column(c, justify="right" if c == "已持續" else "left")
        for r, side, days in sorted(ongoing, key=lambda x: x[2]):
            t.add_row(
                r.stock_id,
                r.name,
                r.strategy.name,
                label[side],
                f"{days} 天",
            )
        console.print(t)
    else:
        console.print("② 沒有持續成立中的條件。")

    forming = report.forming()
    if forming:
        t = Table(title="③ 形成中（還沒觸發買進）")
        for c in ("代號", "名稱", "策略", "數值", "已持續"):
            t.add_column(c, justify="right" if c == "已持續" else "left")
        for r, days in forming:
            values = "、".join(
                f"{operand_label(o)} {_value(r.values[operand_label(o)])}"
                for o in rule_operands(r.strategy.forming)
            )
            t.add_row(r.stock_id, r.name, r.strategy.name, values, f"{days} 天")
        console.print(t)
    elif any(r.strategy.forming for r in report.rows):
        console.print("③ 沒有形成中的型態。")

    for who, msg in report.errors:
        console.print(f"[red]{who}：{msg}[/red]")
    console.print(DISCLAIMER)
    if report.errors:
        raise typer.Exit(1)


@watchlist_app.command("add")
def watchlist_add(
    stocks: Annotated[list[str], typer.Argument(help="股票代號，可一次多個")],
):
    """加入自選股（會向 FinMind 確認代號是否存在）。"""
    svc = MarketDataService()
    with console.status("確認股票代號…"):
        changes = svc.add_to_watchlist(stocks)
    for c in changes:
        if c.status == "added":
            console.print(f"已加入 {c.stock_id} {c.name}")
        elif c.status == "exists":
            console.print(f"[dim]{c.stock_id} {c.name} 已經在清單中[/dim]")
        else:
            console.print(f"[red]{c.stock_id}：{c.message}[/red]")
    if any(c.status == "added" for c in changes):
        console.print("執行 hermes update 下載新加入股票的資料。")
    if any(c.status in ("not_found", "error") for c in changes):
        raise typer.Exit(1)


@watchlist_app.command("remove")
def watchlist_remove(
    stocks: Annotated[list[str], typer.Argument(help="股票代號，可一次多個")],
):
    """從自選股移除（本機已下載的資料會保留）。"""
    for c in MarketDataService().remove_from_watchlist(stocks):
        console.print(
            f"已移除 {c.stock_id}"
            if c.status == "removed"
            else f"[yellow]{c.stock_id} 不在清單中[/yellow]"
        )


@watchlist_app.command("list")
def watchlist_list():
    """列出自選股，以及本機股價資料的最新日期。"""
    svc = MarketDataService()
    items = svc.list_watchlist()
    if not items:
        console.print("自選股清單是空的，用 hermes watchlist add 2330 加入。")
        return
    t = Table(title=f"自選股（{len(items)} 檔）")
    for c in ("代號", "名稱", "加入日期", "本機股價最新日"):
        t.add_column(c)
    for item in items:
        last = svc.last_price_date(item.stock_id)
        t.add_row(
            item.stock_id,
            item.name,
            f"{item.added_at:%Y-%m-%d}",
            str(last) if last else "[yellow]尚未下載[/yellow]",
        )
    console.print(t)


# ---------------------------------------------------------------------------
@app.command()
def tech(
    stock: Annotated[str, typer.Argument(help="股票代號")],
    indicator_set: Annotated[
        str,
        typer.Option(
            "-i", "--set", help="indicators/ 裡的指標組（不含 .yaml），預設是常用版"
        ),
    ] = "common",
    day: Annotated[
        str | None,
        typer.Option("--date", help="看哪一天的數值 YYYY-MM-DD，預設是最新一天"),
    ] = None,
    adjust: Annotated[bool, typer.Option("--adjust", help=ADJUST_HELP)] = False,
):
    """技術指標數值：預設是常用版（indicators/common.yaml），用實際成交價算。"""
    loader = _loader()
    try:
        snap = indicator_snapshot(
            stock, indicator_set, _date(day, "--date"), loader, adjusted=adjust
        )
    except (
        IndicatorSetNotFoundError,
        StrategyConfigError,
        DataError,
        ValueError,
    ) as exc:
        _fail(str(exc))
    console.print(
        Panel.fit(
            f"[bold]{_name(loader, stock)}[/bold]  {snap.date:%Y-%m-%d}  收盤 {num(snap.close)}"
            f"（{PRICE_NAMES[adjust]}）\n指標組：{snap.indicator_set.name}",
            title="技術指標",
        )
    )
    t = Table()
    for c, justify in (("名稱", "left"), ("指標（參數）", "left"), ("數值", "right")):
        t.add_column(c, justify=justify)
    for name, cfg in snap.indicator_set.indicators.items():
        spec = ta.CATALOG[cfg.type]
        label = indicator_label(cfg)
        for k, line in enumerate(spec.outputs or (None,)):
            col = f"{name}.{line}" if line else name
            value = "無法計算" if snap.unavailable[col] else _indicator_num(snap.values[col])
            t.add_row(col, label if k == 0 else "", value)
    console.print(t)
    console.print(
        "[dim]只列出數值，不判斷買賣。要用自己的參數，在 indicators/ 加新的名稱或新的檔案。[/dim]"
    )
    _gap_warnings(stock, adjust)
    _field_warnings(stock, snap.warnings)


def _indicator_num(v: float) -> str:
    return num(v, 2 if pd.isna(v) or abs(v) >= 1 else 4)


@app.command()
def chips(
    stock: Annotated[str, typer.Argument(help="股票代號")],
    window: Annotated[
        list[int] | None,
        typer.Option(
            "-w",
            "--window",
            help="要看近幾個交易日的合計和變化，可重複，例如 -w 5 -w 20",
        ),
    ] = None,
    days: Annotated[int, typer.Option(help="明細顯示最近幾個交易日")] = 10,
):
    """籌碼面：三大法人買賣超、外資持股比例、融資融券。"""
    window = window or []
    if any(w <= 0 for w in window):
        _fail("-w 要是正整數")
    loader = _loader()
    try:
        rep = chip_report(stock, window, days, loader)
    except (DataError, ValueError) as exc:
        _fail(str(exc))
    name = _name(loader, stock)
    console.print(
        Panel.fit(
            f"[bold]{name}[/bold]  資料日期 {rep.as_of:%Y-%m-%d}"
            if rep.as_of is not None
            else name,
            title="籌碼面",
        )
    )
    if not rep.flows.empty:
        fm = {c: (lambda v: colored(v, f"{v:+,.0f}")) for c in rep.flows.columns}
        console.print(frame_table(rep.flows, "三大法人買賣超合計（張）", fm, "法人"))
    if not rep.recent.empty:
        s = "、".join(
            f"{k} 連{'買' if v > 0 else '賣'} {abs(v)} 日"
            for k, v in rep.streaks.items()
            if v
        )
        if s:
            console.print(f"連續買賣超：{s}")
    if not rep.recent.empty:
        fm = {c: (lambda v: colored(v, f"{v:+,.0f}")) for c in rep.recent.columns}
        console.print(
            frame_table(
                rep.recent, f"最近 {len(rep.recent)} 個交易日明細（張）", fm, "日期"
            )
        )

    def signed(v: float, digits: int) -> str:
        return colored(v, "—" if pd.isna(v) else f"{v:+,.{digits}f}")

    def changes(info: dict, prefix: str, digits: int) -> str:
        parts = [f"{w} 日 {signed(info[f'{prefix}{w}'], digits)}" for w in window]
        return f"（和 N 個交易日前相比：{'、'.join(parts)}）" if parts else ""

    def as_of(d: pd.Timestamp) -> str:
        return f"（{d:%Y-%m-%d}）" if rep.as_of is not None and d != rep.as_of else ""

    if rep.foreign_ratio:
        f = rep.foreign_ratio
        console.print(
            f"外資持股比例{as_of(f['date'])}：[bold]{f['now']:.2f}%[/bold]{changes(f, 'chg_', 2)}"
        )
    if rep.margin:
        mg = rep.margin
        console.print(
            f"融資餘額{as_of(mg['date'])}：{mg['margin_balance']:,.0f} 張{changes(mg, 'margin_chg_', 0)}；"
            f"使用率 {pct(mg['margin_usage'])}\n"
            f"融券餘額：{mg['short_balance']:,.0f} 張；券資比 {pct(mg['short_margin_ratio'])}"
        )
    if not window:
        console.print(
            "[dim]要看近 N 個交易日的合計和變化，加上 -w，例如 -w 5 -w 20。[/dim]"
        )
    console.print(DISCLAIMER)


# ---------------------------------------------------------------------------
@app.command("indicators")
def list_indicators():
    """列出策略設定檔可以用的指標類型、參數、輸出的線。"""
    t = Table(title="指標目錄（策略設定檔的 type）")
    for c in ("type", "名稱", "參數（都要指定）", "輸出的線"):
        t.add_column(c, overflow="fold")
    for spec in ta.CATALOG.values():
        params = "、".join(f"{k}（{p.label}）" for k, p in spec.params.items())
        t.add_row(spec.type, spec.title, params or "—", "、".join(spec.outputs) or "—")
    console.print(t)
    console.print(
        "參數沒有預設值，每個都要寫在設定檔裡，例如 {type: cci, period: 88}。\n"
        "多條線的指標在條件裡寫成「名稱.線」，例如 kd.k、macd.hist。\n"
        "要比前 N 天的值，寫成 {ref: 名稱, ago: N}，例如前一天的唐奇安上緣：{ref: dc.upper, ago: 1}。"
    )
    ft = Table(title="籌碼、本益比欄位（條件裡直接寫名稱，不用放在 indicators）")
    for c in ("名稱", "說明", "單位"):
        ft.add_column(c)
    for name, (title, unit, _) in DATA_FIELDS.items():
        ft.add_row(name, title, unit)
    console.print(ft)
    console.print("這些都是當天收盤後才公布：回測只能隔天開盤成交。")
    sets = "、".join(p.stem for p in indicator_set_files()) or "沒有"
    console.print(
        f"\n指標組（{indicators_dir()}）：{sets}。用 hermes tech 2330 -i 名稱 查看數值。"
    )


@app.command("strategies")
def list_strategies():
    """列出策略設定檔，以及機器學習策略和買進持有。"""
    t = Table(
        title="機器學習策略和買進持有（機器學習的參數是程式內建的，還沒改成寫在設定檔裡）"
    )
    for c in ("名稱", "中文", "類型", "預設參數", "說明"):
        t.add_column(c, overflow="fold")
    for spec in REGISTRY.values():
        params = ", ".join(f"{k}={v}" for k, v in spec.defaults.items()) or "—"
        t.add_row(spec.name, spec.title, spec.category, params, spec.description)
    console.print(t)
    console.print(
        "用法：hermes backtest 2330 -s hmm -p states=3；meta 要指定主要策略：-p primary=策略設定檔名稱"
    )

    rules = list_rule_strategies()
    if not rules:
        console.print(f"\n[dim]{strategies_dir()} 裡還沒有策略設定檔。[/dim]")
        return
    rt = Table(title=f"設定檔策略（{strategies_dir()}）")
    for c in ("檔名", "策略名稱", "說明"):
        rt.add_column(c, overflow="fold")
    for path, item in rules:
        if isinstance(item, StrategyConfigError):
            rt.add_row(path.stem, "[red]設定有誤[/red]", f"[red]{item}[/red]")
        else:
            rt.add_row(path.stem, item.name, item.description or "—")
    console.print(rt)
    console.print(
        f"用法：hermes signals 2330 -s {rules[0][0].stem}、hermes backtest 2330 -s {rules[0][0].stem}"
    )


def _signal_label(row: pd.Series) -> str:
    parts = []
    for side, label, color in (("buy", "買", "red"), ("sell", "賣", "green")):
        n = int(row[f"{side}_streak"])
        if n == 1:
            parts.append(f"[bold {color}]{label}[/bold {color}]")
        elif n > 1:
            parts.append(f"[{color}]{label}（第 {n} 天）[/{color}]")
    return "、".join(parts)


def _condition_state(streak: int) -> str:
    if streak == 1:
        return "成立（新觸發）"
    return f"成立（已持續 {streak} 天）" if streak > 1 else "不成立"


def _price(p: float) -> str:
    return f"{p:,.2f}".rstrip("0").rstrip(".")


def _value(x: float) -> str:
    return "—" if pd.isna(x) else _price(x)


@app.command()
def signals(
    stock: Annotated[str, typer.Argument(help="股票代號")],
    strategy: Annotated[
        str,
        typer.Option(
            "-s",
            "--strategy",
            help="策略設定檔：strategies/ 裡的檔名（不含 .yaml）或檔案路徑",
        ),
    ],
    start: Annotated[
        str | None,
        typer.Option(
            help="顯示的起始日 YYYY-MM-DD，預設為結束日的一年前（計算一律使用全部資料）",
        ),
    ] = None,
    end: Annotated[
        str | None, typer.Option(help="顯示的結束日，預設為最新一天")
    ] = None,
    every_day: Annotated[
        bool,
        typer.Option(
            "--every-day", help="列出每一天的指標數值（預設只列出有訊號的日子）"
        ),
    ] = False,
    plot: Annotated[
        bool, typer.Option("--plot/--no-plot", help="輸出買賣點圖 PNG")
    ] = True,
    adjust: Annotated[bool, typer.Option("--adjust", help=ADJUST_HELP)] = False,
):
    """歷史買賣點：依策略設定檔逐日判斷，列出過去的買進、賣出訊號。"""
    try:
        hist = signal_history(stock, strategy, adjust)
    except (DataError, StrategyConfigError, StrategyNotFoundError) as exc:
        _fail(str(exc))
    last_day = hist.result.values.index[-1]
    s, e = _dates(start, end)
    e = e or last_day.date()
    s = s or (pd.Timestamp(e) - pd.DateOffset(years=1)).date()
    if s > e:
        _fail(f"--start 不能晚於結束日 {e}")
    period = hist.table(s, e, every_day=True)
    if period.empty:
        _fail(f"{s} ～ {e} 之間沒有交易資料")

    st = hist.strategy
    name = _name(_loader(), stock)
    indicators = "、".join(
        f"{n} = {ta.CATALOG[c.type].title}"
        f"({', '.join(f'{k}={v}' for k, v in ta.CATALOG[c.type].resolve_params(c.params).items())})"
        for n, c in st.indicators.items()
    )
    console.print(
        Panel.fit(
            f"[bold]{name}[/bold]  策略：{st.name}\n"
            f"買進：{describe(st.buy)}\n賣出：{describe(st.sell)}\n"
            + (f"形成中：{describe(st.forming)}\n" if st.forming else "")
            + f"指標：{indicators or '—'}\n價格：{PRICE_NAMES[adjust]}",
            title="歷史買賣點",
        )
    )
    _gap_warnings(stock, adjust)
    _field_warnings(stock, hist.warnings)

    rows = period if every_day else period[period["buy"] | period["sell"]]
    cols = hist.indicator_columns
    t = Table(
        title=f"{s} ～ {e}（{'每一天' if every_day else '只列出有訊號的日子'}）"
    )
    for c in ("日期", "收盤", *cols, "訊號"):
        t.add_column(c, justify="left" if c == "訊號" else "right")
    for d, r in rows.iterrows():
        t.add_row(f"{d:%Y-%m-%d}", num(r["close"]), *(num(r[c]) for c in cols), _signal_label(r))
    if rows.empty:
        console.print("這段期間沒有任何訊號。")
    else:
        console.print(t)

    n_buy, n_sell = (
        int((period["buy_streak"] == 1).sum()),
        int((period["sell_streak"] == 1).sum()),
    )
    console.print(
        f"{s} ～ {e}：買進訊號 {n_buy} 次（條件成立 {int(period['buy'].sum())} 天）、"
        f"賣出訊號 {n_sell} 次（條件成立 {int(period['sell'].sum())} 天）"
    )
    latest = hist.table(every_day=True).iloc[-1]
    last_buy, last_sell = hist.last_trigger("buy"), hist.last_trigger("sell")
    console.print(
        f"最新一天（{last_day:%Y-%m-%d}）：買進條件{_condition_state(int(latest['buy_streak']))}；"
        f"賣出條件{_condition_state(int(latest['sell_streak']))}\n"
        f"最近一次買進訊號：{f'{last_buy:%Y-%m-%d}' if last_buy is not None else '—'}；"
        f"最近一次賣出訊號：{f'{last_sell:%Y-%m-%d}' if last_sell is not None else '—'}"
    )
    if plot:
        from hermes.cli.plots import plot_signals

        path = plot_signals(
            period,
            period["buy_streak"],
            period["sell_streak"],
            st,
            f"{name} {st.name}（{PRICE_NAMES[adjust]}）",
            reports_dir() / f"{stock}_signals_{Path(strategy).stem}.png",
        )
        console.print(f"圖表：{path}")
    console.print(DISCLAIMER)


def _is_rule_strategy(ref: str) -> bool:
    try:
        return is_rule_strategy(ref)
    except StrategyConfigError as exc:
        _fail(str(exc))


def _trade_rows(rb: RuleBacktest, trades: pd.DataFrame) -> Table:
    t = Table()
    for c in (
        "買進訊號",
        "買進日",
        "買進價",
        "賣出訊號",
        "賣出日",
        "賣出價",
        "報酬",
        "持有天數",
    ):
        t.add_column(c, justify="right")
    for _, r in trades.iterrows():
        buy_price = _price(r["entry_price"]) + (
            " [yellow]⚠ 漲停[/yellow]" if r["buy_at_limit"] else ""
        )
        if pd.isna(r["exit"]):
            sell = ("[dim]持有中[/dim]", "—", "—")
        else:
            sell_price = _price(r["exit_price"]) + (
                " [yellow]⚠ 跌停[/yellow]" if r["sell_at_limit"] else ""
            )
            sell = (f"{r['sell_signal']:%Y-%m-%d}", f"{r['exit']:%Y-%m-%d}", sell_price)
        t.add_row(
            f"{r['buy_signal']:%Y-%m-%d}",
            f"{r['entry']:%Y-%m-%d}",
            buy_price,
            *sell,
            colored(r["return"], pct(r["return"], sign=True)),
            str(r["bars"]),
        )
    return t


def _backtest_rules(
    stock: str,
    ref: str,
    start: date | None,
    end: date | None,
    execution: str,
    costs: CostModel,
    n_trades: int,
    plot: bool,
    adjust: bool = False,
) -> None:
    """設定檔策略的回測：訊號用實際成交價判斷（還原模式用還原價格），報酬包含股利。"""
    loader = _loader()
    try:
        hist = signal_history(stock, ref, adjust, loader)
        rb = rule_backtest(hist, start, end, costs, execution, loader)
    except (DataError, StrategyConfigError, StrategyNotFoundError) as exc:
        _fail(str(exc))
    st, res = rb.strategy, rb.result
    name = _name(loader, stock)
    if start is None:
        begin_note = "（指標都算得出來的第一天）"
    elif rb.warmup_end > rb.start:
        begin_note = (
            f"（指標到 {rb.warmup_end:%Y-%m-%d} 才都算得出來，在這之前不會有訊號）"
        )
    else:
        begin_note = ""
    when = "隔天開盤" if execution == "open" else "當天收盤"
    console.print(
        Panel.fit(
            f"[bold]{name}[/bold]  策略：{st.name}\n"
            f"買進：{describe(st.buy)}\n賣出：{describe(st.sell)}\n"
            f"期間：{rb.start:%Y-%m-%d}{begin_note} ～ {res.returns.index[-1]:%Y-%m-%d}，從空手開始\n"
            f"做法：買進條件成立後{when}買進，持有到賣出條件成立後{when}賣出；不加碼、不放空\n"
            f"價格：訊號用{PRICE_NAMES[adjust]}判斷；報酬包含股利\n"
            f"成本：手續費 {costs.commission:.4%}（折扣 {costs.discount:g}）、證交稅 {costs.tax:.1%}、"
            f"單邊滑價 {costs.slippage:.2%}",
            title="設定檔策略回測",
        )
    )
    _gap_warnings(stock, adjust, returns=True)
    _field_warnings(stock, hist.warnings)
    _etf_tax_note(stock, costs.tax)

    console.print(metrics_table([res], f"{name} 回測：{st.name}"))
    trades = rb.trades()
    if trades.empty:
        console.print("這段期間沒有任何交易。")
    else:
        t = _trade_rows(rb, trades.iloc[-n_trades:])
        t.title = f"交易明細（最近 {min(n_trades, len(trades))} 筆，共 {len(trades)} 筆；價格是當天的成交價，報酬包含股利和成本）"
        console.print(t)
    if trades["buy_at_limit"].any() or trades["sell_at_limit"].any():
        console.print(
            "[yellow]⚠ 漲停：用漲停價買進；跌停：用跌停價賣出。實際上可能排不到，回測結果會偏樂觀[/yellow]"
        )

    if end is None:
        last, now, action = rb.latest()
        action = {"buy": "買進", "sell": "賣出"}.get(action, "")
        if execution == "open":
            action = f"，下一個交易日開盤{action}" if action else ""
        else:
            action = f"，當天收盤{action}" if action else ""
        console.print(
            f"最新（{last:%Y-%m-%d} 收盤後）：照策略應該{'持有' if now else '空手'}{action}"
        )
    if plot:
        from hermes.cli.plots import plot_backtests

        path = plot_backtests(
            [res],
            f"{name} {st.name}",
            reports_dir() / f"{stock}_backtest_{Path(ref).stem}.png",
        )
        console.print(f"圖表：{path}")
    console.print(DISCLAIMER)


# ---------------------------------------------------------------------------
@app.command()
def backtest(
    stock: Annotated[str, typer.Argument(help="股票代號")],
    strategy: Annotated[
        str,
        typer.Option(
            "-s",
            "--strategy",
            help="策略：strategies/ 裡的設定檔（不含 .yaml），或機器學習策略、buy_hold（hermes strategies 查看）",
        ),
    ],
    param: Annotated[
        list[str] | None,
        typer.Option("-p", "--param", help="機器學習策略的參數 key=value，可重複"),
    ] = None,
    start: Annotated[str | None, typer.Option(help="回測起始日 YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="回測結束日")] = None,
    execution: Annotated[
        str,
        typer.Option("--exec", help="成交方式：open=次日開盤（預設）、close=當日收盤"),
    ] = "open",
    fee_discount: Annotated[
        float, typer.Option(help="手續費折扣，例如 0.28 代表 2.8 折")
    ] = 1.0,
    slippage: Annotated[
        float, typer.Option(help="單邊滑價；開盤、收盤都是集合競價，預設 0")
    ] = 0.0,
    tax: Annotated[float, typer.Option(help="證交稅（ETF 請設 0.001）")] = 0.003,
    trades: Annotated[int, typer.Option(help="顯示最近幾筆交易")] = 10,
    plot: Annotated[bool, typer.Option("--plot/--no-plot")] = True,
    adjust: Annotated[bool, typer.Option("--adjust", help=ADJUST_HELP)] = False,
):
    """回測單一策略。"""
    if execution not in ("open", "close"):
        _fail("--exec 只能是 open 或 close")
    start, end = _dates(start, end)
    _at_least_one(trades, "--trades")
    if _is_rule_strategy(strategy):
        if param:
            _fail("設定檔策略的參數寫在設定檔裡，不能用 -p")
        _backtest_rules(
            stock,
            strategy,
            start,
            end,
            execution,
            _costs(fee_discount, slippage, tax),
            trades,
            plot,
            adjust,
        )
        return
    if adjust:
        _fail("--adjust 只用在策略設定檔；機器學習策略和買進持有本來就用還原價格")
    _etf_tax_note(stock, tax)
    loader = _loader()
    try:
        params = parse_params(get_strategy(strategy), param or [])
    except (KeyError, ValueError) as exc:
        _fail(str(exc))
    try:
        with _ml_progress() as on_progress:
            bt = builtin_backtest(
                stock,
                strategy,
                params,
                start,
                end,
                _costs(fee_discount, slippage, tax),
                execution,
                loader,
                on_progress,
            )
    except (ValueError, DataError, StrategyNotFoundError, StrategyConfigError) as exc:
        _fail(str(exc))
    res, target, label = bt.result, bt.target, bt.label
    name = _name(loader, stock)
    period = f"{res.returns.index[0]:%Y-%m-%d} ～ {res.returns.index[-1]:%Y-%m-%d}"
    console.print(metrics_table([res], f"{name} 回測：{label}（{period}）"))

    if not res.trades.empty:
        tr = res.trades.iloc[-trades:].copy()
        tt = Table(title=f"最近 {len(tr)} 筆交易")
        for c in ("進場", "出場", "方向", "報酬", "持有天數"):
            tt.add_column(c, justify="right")
        for _, row in tr.iterrows():
            ex = "持有中" if pd.isna(row["exit"]) else f"{row['exit']:%Y-%m-%d}"
            tt.add_row(
                f"{row['entry']:%Y-%m-%d}",
                ex,
                "多" if row["side"] > 0 else "空",
                colored(row["return"], pct(row["return"], sign=True)),
                str(int(row["bars"])),
            )
        console.print(tt)
    latest = target.dropna()
    if not latest.empty:
        console.print(
            f"最新訊號（{latest.index[-1]:%Y-%m-%d} 收盤後）：目標部位 [bold]{latest.iloc[-1]:.2f}[/bold]"
            f"（下一交易日{'開盤' if execution == 'open' else '收盤'}執行）"
        )
    if plot:
        from hermes.cli.plots import plot_backtests

        path = plot_backtests(
            [res], f"{name} {label}", reports_dir() / f"{stock}_backtest_{strategy}.png"
        )
        console.print(f"圖表：{path}")
    console.print(DISCLAIMER)


# ---------------------------------------------------------------------------
@app.command()
def compare(
    stock: Annotated[str, typer.Argument(help="股票代號")],
    strategies: Annotated[
        list[str] | None,
        typer.Option(
            "-s",
            "--strategy",
            help="要比較的策略：strategies/ 裡的設定檔，或機器學習策略，可重複；買進持有一定會列出",
        ),
    ] = None,
    ml: Annotated[
        bool,
        typer.Option("--ml", help="一併比較機器學習策略（ml、hmm、dmn），需要較長時間"),
    ] = False,
    start: Annotated[
        str | None, typer.Option(help="起始日（預設為所有策略都有訊號的第一天）")
    ] = None,
    end: Annotated[str | None, typer.Option(help="結束日")] = None,
    execution: Annotated[
        str,
        typer.Option("--exec", help="成交方式：open=次日開盤（預設）、close=當日收盤"),
    ] = "open",
    fee_discount: Annotated[
        float, typer.Option(help="手續費折扣，例如 0.28 代表 2.8 折")
    ] = 1.0,
    slippage: Annotated[
        float, typer.Option(help="單邊滑價；開盤、收盤都是集合競價，預設 0")
    ] = 0.0,
    tax: Annotated[float, typer.Option(help="證交稅（ETF 請設 0.001）")] = 0.003,
    sort: Annotated[
        str, typer.Option(help="排序依據：sharpe、cagr、calmar、max_drawdown")
    ] = "sharpe",
    plot: Annotated[bool, typer.Option("--plot/--no-plot")] = True,
    top: Annotated[int, typer.Option(help="圖表中顯示前幾名")] = 6,
    adjust: Annotated[bool, typer.Option("--adjust", help=ADJUST_HELP)] = False,
):
    """並排比較多個策略。"""
    names = list(strategies or [])
    if ml:
        names += ["ml", "hmm"]
        try:
            import torch  # noqa: F401

            names.append("dmn")
        except ImportError:
            console.print(
                "[yellow]未安裝 torch，略過 dmn（uv sync --extra deep）[/yellow]"
            )
    if not names:
        _fail("請用 -s 指定要比較的策略，例如 -s connors_rsi2")
    if execution not in ("open", "close"):
        _fail("--exec 只能是 open 或 close")
    start, end = _dates(start, end)
    _at_least_one(top, "--top")
    _etf_tax_note(stock, tax)
    loader = _loader()
    try:
        with _ml_progress() as on_progress:
            cmp = compare_strategies(
                stock,
                names,
                start,
                end,
                _costs(fee_discount, slippage, tax),
                execution,
                sort,
                loader,
                on_progress=on_progress,
                adjusted=adjust,
            )
    except (
        KeyError,
        ValueError,
        DataError,
        StrategyNotFoundError,
        StrategyConfigError,
    ) as exc:
        _fail(str(exc))
    rows = [
        {"name": r.name, "metrics": r.metrics}
        for r in cmp.results
    ]
    rows.append({"name": "[bold]買進持有[/bold]", "metrics": cmp.benchmark.metrics})
    name = _name(loader, stock)
    index = cmp.benchmark.returns.index
    period = f"{index[0]:%Y-%m-%d} ～ {index[-1]:%Y-%m-%d}"
    console.print(ranking_table(rows, f"{name} 策略比較（{period}，依 {sort} 排序）"))
    _gap_warnings(stock, adjust, returns=True)  # 報酬都用還原價格算
    if cmp.rules:
        _field_warnings(stock, cmp.warnings)
        console.print(
            f"[dim]設定檔策略：訊號用{PRICE_NAMES[adjust]}判斷，報酬包含股利。[/dim]"
        )
    if plot:
        from hermes.cli.plots import plot_backtests

        path = plot_backtests(
            cmp.results[:top],
            f"{name} 策略比較",
            reports_dir() / f"{stock}_compare.png",
        )
        console.print(f"圖表：{path}")
    console.print(
        "[dim]比較多個策略後挑最好的，本身就是一種多重檢定，最好的那個可能只是運氣好。[/dim]"
    )
    console.print(DISCLAIMER)


# ---------------------------------------------------------------------------
@app.command("ml")
def ml_command(
    stock: Annotated[str, typer.Argument(help="股票代號")],
    model: Annotated[
        str, typer.Option(help="logit / rf / et / hgb / mlp / lstm / ensemble")
    ] = "hgb",
    horizon: Annotated[
        int, typer.Option(help="triple-barrier 垂直柵欄（交易日）")
    ] = 10,
    pt: Annotated[
        float, typer.Option(help="停利柵欄倍數（× 波動度 × √horizon）")
    ] = 1.0,
    sl: Annotated[float, typer.Option(help="停損柵欄倍數")] = 1.0,
    meta: Annotated[
        str | None,
        typer.Option(help="meta-labeling 的主要策略：strategies/ 裡的設定檔名稱"),
    ] = None,
    cusum: Annotated[bool, typer.Option(help="以 CUSUM 濾波器挑選事件")] = False,
    exogenous: Annotated[
        bool,
        typer.Option("--exogenous/--no-exogenous", help="加入籌碼、估值、大盤特徵"),
    ] = True,
    changepoint: Annotated[
        bool, typer.Option("--changepoint/--no-changepoint", help="加入 BOCPD 變點特徵")
    ] = True,
    fracdiff: Annotated[float, typer.Option(help="分數差分階數（0 表示不使用）")] = 0.4,
    train_min: Annotated[int, typer.Option(help="第一次訓練前至少需要的交易日")] = 1000,
    step: Annotated[int, typer.Option(help="每隔幾個交易日重新訓練")] = 63,
    cv: Annotated[
        bool, typer.Option("--cv/--no-cv", help="執行 Purged K-Fold 交叉驗證")
    ] = True,
    sizing: Annotated[
        str,
        typer.Option(
            help="部位決定方式：threshold（機率門檻）或 afml（2Φ(z)−1 下注大小）"
        ),
    ] = "threshold",
    threshold: Annotated[
        float, typer.Option(help="sizing=threshold 時的機率門檻")
    ] = 0.5,
    cpcv: Annotated[
        bool, typer.Option("--cpcv", help="執行 CPCV，估計 Sharpe 分布（較耗時）")
    ] = False,
    mda: Annotated[
        bool, typer.Option("--mda", help="計算樣本外 MDA 特徵重要性（較耗時）")
    ] = False,
    start: Annotated[str | None, typer.Option(help="回測起始日")] = None,
    seed: int = 0,
    plot: Annotated[bool, typer.Option("--plot/--no-plot")] = True,
):
    """機器學習：triple-barrier 標記 + purged CV + walk-forward 回測 + 最新預測。"""
    from hermes.ml.features import build_features
    from hermes.ml.pipeline import (
        MLConfig,
        cpcv_paths,
        fit_latest,
        latest_position,
        make_dataset,
        mda_importance,
        proba_to_position,
        purged_cv,
        walk_forward,
    )

    start = _date(start, "--start")
    loader = _loader()
    try:
        prices = loader.prices(stock)
    except DataError as exc:
        _fail(str(exc))
    name = _name(loader, stock)
    cfg = MLConfig(
        model=model,
        horizon=horizon,
        pt=pt,
        sl=sl,
        cusum=cusum,
        train_min=train_min,
        step=step,
        seed=seed,
        sizing=sizing,
        threshold=threshold,
    )
    ctx = strategy_context(prices, stock, loader)
    with console.status("建立特徵與標籤…"):
        try:
            extra = ctx.exogenous() if exogenous else None
            X = build_features(
                prices,
                extra,
                fracdiff_d=fracdiff if fracdiff > 0 else None,
                changepoint=changepoint,
            )
        except ValueError as exc:
            _fail(str(exc))
        side = None
        if meta:
            try:
                side = np.sign(ctx.rule_position(meta).fillna(0.0))
            except (StrategyNotFoundError, StrategyConfigError, DataError) as exc:
                _fail(str(exc))
        ds = make_dataset(prices, X, cfg, side=side)
    ev = ds.events
    console.print(
        Panel.fit(
            f"[bold]{name}[/bold]  模型：{model}  {'meta-labeling（主要策略：' + meta + '）' if meta else '方向預測'}\n"
            f"特徵 {X.shape[1]} 個；事件 {len(ev):,} 筆（{ev.index[0]:%Y-%m-%d} ～ {ev.index[-1]:%Y-%m-%d}）；"
            f"正類比例 {ds.y.mean():.1%}；平均唯一性 {_avg_u(ds):.2f}\n"
            f"柵欄：停利 {pt}σ√h、停損 {sl}σ√h、到期 {horizon} 日（到期時依報酬正負標記）",
            title="機器學習",
        )
    )

    if cv:
        with console.status("Purged K-Fold 交叉驗證中…"):
            cv_df, _ = purged_cv(ds, cfg)
        cv_show = cv_df.set_index("fold")
        cv_show.loc["平均"] = cv_show.mean()
        fm = {
            "n_train": lambda v: f"{v:,.0f}",
            "n_test": lambda v: f"{v:,.0f}",
            "base_rate": lambda v: pct(v, 1),
            "accuracy": lambda v: pct(v, 1),
            "auc": lambda v: num(v, 3),
            "log_loss": lambda v: num(v, 4),
            "f1": lambda v: num(v, 3),
        }
        console.print(
            frame_table(
                cv_show,
                f"Purged {cfg.n_splits}-Fold CV（embargo {cfg.embargo_pct:.0%}）",
                fm,
                "折",
            )
        )
        console.print("[dim]AUC 0.5 等於隨機猜測；準確率請與「正類比例」比較。[/dim]")

    with _progress("walk-forward 訓練中") as cb:
        proba = walk_forward(ds, cfg, progress=cb)
    position = proba_to_position(proba, cfg, side=side)
    start = start or position.first_valid_index()
    res = run_backtest(prices, position, CostModel(), "open", f"ML-{model}", {}, start)
    console.print(
        metrics_table([res], f"Walk-forward 樣本外回測（每 {step} 日重新訓練）")
    )

    if cpcv:
        with console.status("CPCV（6 組取 2，5 條路徑）計算中…"):
            paths = cpcv_paths(ds, cfg)
        srs = []
        for path in paths:
            pos = proba_to_position(path.reindex(prices.index), cfg, side=side)
            r = run_backtest(
                prices, pos, CostModel(), "open", "", {}, path.index[0], path.index[-1]
            )
            srs.append(r.metrics["sharpe"])
        srs = np.array(srs)
        console.print(
            f"CPCV Sharpe 分布（{len(srs)} 條路徑）：平均 {srs.mean():.2f}、標準差 {srs.std(ddof=1):.2f}、"
            f"最小 {srs.min():.2f}、最大 {srs.max():.2f}、為正比例 {(srs > 0).mean():.0%}"
        )

    with console.status("以全部資料訓練最終模型…"):
        latest, imp = fit_latest(ds, cfg)
    p = latest["proba"]
    cur_side = float(side.iloc[-1]) if side is not None else None
    size = latest_position(p, cfg, cur_side)
    if side is not None:
        desc = f"主要策略目前方向 {cur_side:+.0f}，訊號成功機率 {p:.1%}"
    else:
        desc = f"未來 {horizon} 日先觸及停利柵欄的機率 {p:.1%}"
    console.print(
        Panel.fit(
            f"{latest['date']:%Y-%m-%d} 收盤後：{desc}\n建議部位（{sizing}）：[bold]{size:.2f}[/bold]",
            title="最新預測（以全部資料訓練的最終模型）",
        )
    )

    if mda:
        with console.status("計算 MDA 特徵重要性…"):
            imp = mda_importance(ds, cfg)
        title = "MDA 特徵重要性（log loss 增加量，前 15）"
    else:
        title = "特徵重要性（模型內建，前 15）"
    if imp is not None:
        t = Table(title=title)
        t.add_column("特徵")
        t.add_column("重要性", justify="right")
        for k, v in imp.head(15).items():
            t.add_row(k, f"{v:.4f}")
        console.print(t)
    else:
        console.print(
            "[dim]此模型沒有內建特徵重要性，可加上 --mda 計算樣本外重要性。[/dim]"
        )
    if plot:
        from hermes.cli.plots import plot_ml

        path = plot_ml(
            prices,
            proba,
            res,
            f"{name} 機器學習（{model}）樣本外",
            reports_dir() / f"{stock}_ml_{model}.png",
        )
        console.print(f"圖表：{path}")
    console.print(DISCLAIMER)


def _avg_u(ds) -> float:
    from hermes.ml.weights import average_uniqueness

    return float(average_uniqueness(ds.X.index, ds.events["t1"]).mean())


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
