"""資料正確性檢查：和證交所官方資料比對（股價、籌碼、本益比）、檢查還原權息、
檢查價格是否落在升降單位和漲跌停範圍內。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from hermes.config import HISTORY_START
from hermes.data import DataError, DataLoader
from hermes.data.loader import CORPORATE_ACTIONS
from hermes.data.twse import (
    FOREIGN_HOLDING_FIELDS,
    INSTITUTIONAL_FIELDS,
    MARGIN_COLUMNS,
    TwseClient,
)
from hermes.strategies import candidate_closes, tick_size

PRICE_COLUMNS = ["open", "max", "min", "close"]
COMPARE_COLUMNS = PRICE_COLUMNS + [
    "Trading_Volume",
    "Trading_money",
    "Trading_turnover",
]


@dataclass
class Check:
    name: str
    status: str  # ok、fail、warn（提醒，不是資料錯誤）、skip
    summary: str
    problems: list[str] = field(default_factory=list)


def _months_back(last: date, months: int) -> list[date]:
    first = pd.Timestamp(last).to_period("M") - (months - 1)
    return [
        p.to_timestamp().date()
        for p in pd.period_range(first, periods=months, freq="M")
    ]


def check_official(
    stock_id: str, raw: pd.DataFrame, market: str, months: int, twse: TwseClient
) -> Check:
    name = "股價和證交所比對"
    if market != "twse":
        return Check(name, "skip", f"市場別是 {market or '未知'}，目前只能比對上市股票")
    finmind = raw.assign(date=raw["date"].astype(str).str[:10]).set_index("date")
    last = finmind.index.max()
    official = pd.concat(
        [twse.stock_day(stock_id, m) for m in _months_back(date.fromisoformat(last), months)]
    )
    # 只比到本機最後一天：之後的日子是還沒下載，不是 FinMind 缺資料
    official = official[official["date"] <= last].set_index("date")
    problems = [
        f"{d} 官方有資料，FinMind 沒有"
        for d in official.index.difference(finmind.index)
    ]
    in_range = finmind.index[
        (finmind.index >= official.index.min())
        & (finmind.index <= official.index.max())
    ]
    # FinMind 把沒有成交的日子（例如暫停交易）記成全 0，官方則沒有這天；兩邊一致，DataLoader 也會略過這天
    no_trade = finmind.index[(finmind[PRICE_COLUMNS].astype(float) == 0).all(axis=1)]
    problems += [
        f"{d} FinMind 有資料，官方沒有"
        for d in in_range.difference(official.index).difference(no_trade)
    ]
    both = official.index.intersection(finmind.index)
    diff = (
        finmind.loc[both, COMPARE_COLUMNS].astype(float)
        - official.loc[both, COMPARE_COLUMNS].astype(float)
    ).abs()
    for d, row in diff[(diff > 1e-6).any(axis=1)].iterrows():
        cols = "、".join(c for c in COMPARE_COLUMNS if row[c] > 1e-6)
        problems.append(f"{d} 欄位不同：{cols}")
    summary = f"最近 {months} 個月共 {len(both)} 個交易日（到本機最後一天 {last}）"
    skipped = in_range.intersection(no_trade).difference(official.index)
    if len(skipped):
        summary += f"；{'、'.join(skipped)} 兩邊都沒有成交"
    return Check(name, "fail" if problems else "ok", summary, problems)


def _differs(a, b) -> bool:
    """官方寫「-」（None）和 FinMind 的 0 或空值視為相同。"""
    a = 0.0 if a is None or pd.isna(a) else float(a)
    b = 0.0 if b is None or pd.isna(b) else float(b)
    return abs(a - b) > 1e-6


def _by_date(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.assign(date=raw["date"].astype(str).str[:10])
        .drop_duplicates("date", keep="last")
        .set_index("date")
    )


def check_chips(
    stock_id: str, loader: DataLoader, market: str, days: int, twse: TwseClient
) -> Check:
    """最近 days 個交易日的三大法人買賣超、融資融券、外資持股，和證交所逐欄比對。

    三大法人比對五個類別的買進、賣出股數，以及合計買賣超；官方沒列出的股票代表當天三大法人都沒有買賣。
    """
    name = "籌碼和證交所比對"
    if market != "twse":
        return Check(name, "skip", f"市場別是 {market or '未知'}，目前只能比對上市股票")
    if days <= 0:
        return Check(name, "skip", "沒有指定要比對的天數")
    dates = loader.prices(stock_id, adjust=False).index[-days:]
    inst = loader.read("TaiwanStockInstitutionalInvestorsBuySell", stock_id)
    inst = (
        inst.assign(date=inst["date"].astype(str).str[:10])
        .drop_duplicates(["date", "name"], keep="last")
        .set_index(["date", "name"])[["buy", "sell"]]
    )
    margin = _by_date(loader.read("TaiwanStockMarginPurchaseShortSale", stock_id))
    holding = _by_date(loader.read("TaiwanStockShareholding", stock_id))
    problems = []
    for d in dates:
        day = f"{d:%Y-%m-%d}"
        official = twse.institutional(d.date())
        row = (
            official.loc[stock_id]
            if stock_id in official.index
            else pd.Series(0.0, index=official.columns)
        )
        total = 0.0
        for n in INSTITUTIONAL_FIELDS:
            ours = (
                inst.loc[(day, n)]
                if (day, n) in inst.index
                else pd.Series({"buy": None, "sell": None})
            )
            total += float(ours["buy"] or 0) - float(ours["sell"] or 0)
            for side in ("buy", "sell"):
                if _differs(ours[side], row[f"{n}_{side}"]):
                    problems.append(
                        f"{day} 三大法人 {n} {side}：FinMind {ours[side]}，官方 {row[f'{n}_{side}']:g}"
                    )
        if _differs(total, row["total"]):
            problems.append(
                f"{day} 三大法人合計買賣超：FinMind {total:g}，官方 {row['total']:g}"
            )

        for label, ours_table, official_table, columns in (
            ("融資融券", margin, twse.margin(d.date()), MARGIN_COLUMNS),
            (
                "外資持股",
                holding,
                twse.foreign_holding(d.date()),
                list(FOREIGN_HOLDING_FIELDS.values()),
            ),
        ):
            has_ours, has_official = (
                day in ours_table.index,
                stock_id in official_table.index,
            )
            if has_ours != has_official:
                problems.append(
                    f"{day} {label}：{'官方' if has_ours else 'FinMind'}沒有資料"
                )
                continue
            if not has_ours:
                continue
            for c in columns:
                if _differs(ours_table.at[day, c], official_table.at[stock_id, c]):
                    problems.append(
                        f"{day} {label} {c}：FinMind {ours_table.at[day, c]}，"
                        f"官方 {official_table.at[stock_id, c]:g}"
                    )
    summary = f"三大法人、融資融券、外資持股，最近 {len(dates)} 個交易日（{dates[0]:%Y-%m-%d} 起）"
    return Check(name, "fail" if problems else "ok", summary, problems)


def check_valuation(
    stock_id: str, loader: DataLoader, market: str, months: int, twse: TwseClient
) -> Check:
    """最近 months 個月每天的本益比、股價淨值比、殖利率，和證交所逐日比對。"""
    name = "本益比和證交所比對"
    if market != "twse":
        return Check(name, "skip", f"市場別是 {market or '未知'}，目前只能比對上市股票")
    ours = _by_date(loader.read("TaiwanStockPER", stock_id))
    last = ours.index.max()
    official = pd.concat(
        [twse.valuation(stock_id, m) for m in _months_back(date.fromisoformat(last), months)]
    )
    # 只比到本機最後一天：之後的日子是還沒下載，不是 FinMind 缺資料
    official = official[official["date"] <= last].set_index("date")
    problems = [
        f"{d} 官方有資料，FinMind 沒有" for d in official.index.difference(ours.index)
    ]
    in_range = ours.index[
        (ours.index >= official.index.min()) & (ours.index <= official.index.max())
    ]
    problems += [
        f"{d} FinMind 有資料，官方沒有" for d in in_range.difference(official.index)
    ]
    both = official.index.intersection(ours.index)
    for d in both:
        for c in ("PER", "PBR", "dividend_yield"):
            if _differs(ours.at[d, c], official.at[d, c]):
                problems.append(
                    f"{d} {c}：FinMind {ours.at[d, c]}，官方 {official.at[d, c]}"
                )
    summary = f"本益比、股價淨值比、殖利率，最近 {months} 個月共 {len(both)} 個交易日（到本機最後一天 {last}）"
    return Check(name, "fail" if problems else "ok", summary, problems)


def check_adjustment(stock_id: str, loader: DataLoader) -> Check:
    """除權息、減資、分割當天的還原報酬，應等於「收盤價 ÷ 官方參考價 − 1」。"""
    name = "還原（除權息、減資、分割）"
    missing = [
        label
        for dataset, label, *_ in CORPORATE_ACTIONS
        if loader.cache.read(dataset, stock_id) is None
    ]
    if missing:
        return Check(
            name,
            "fail",
            f"尚未下載{'、'.join(missing)}紀錄，請執行 hermes fetch {stock_id}",
        )
    adj = loader.prices(stock_id, adjust=True)
    ret = adj["close"].pct_change()
    problems, checked = [], 0
    for _, ev in loader.corporate_actions(stock_id).iterrows():
        d = ev["date"]
        if d not in adj.index or d == adj.index[0]:
            continue
        expected = adj.at[d, "raw_close"] / ev["reference"] - 1
        checked += 1
        if abs(ret.at[d] - expected) > 1e-9:
            problems.append(
                f"{d:%Y-%m-%d} {ev['source']}：還原後報酬 {ret.at[d]:.4%}，依官方參考價應為 {expected:.4%}"
            )
    for _, g in loader.unexplained_gaps(stock_id).iterrows():
        problems.append(
            f"{g['date']:%Y-%m-%d} 前收 {g['prev_close']:g}、收盤 {g['close']:g}（{g['gap']:+.1%}），"
            f"超過漲跌幅限制卻沒有官方除權息／減資／分割紀錄；不會估算還原，這段期間的還原價格可能不正確"
        )
    return Check(
        name,
        "fail" if problems else "ok",
        f"檢查 {checked} 次官方紀錄，以及有沒有無法解釋的跳空",
        problems,
    )


# 官方參考價到小數第二位（實際資料是無條件捨去，例如 14.5789 公布 14.57），算回的參考價差不到 0.01
PRICE_PRECISION = 0.01
# 除權息明細的每千股無償配股只到小數第一位，每股變成幾股最多差 0.00005（見 hermes.data.twse）
RIGHTS_RATIO_PRECISION = 0.05 / 1000


def check_share_changes(stock_id: str, loader: DataLoader) -> Check:
    """股數改變的事件（除權、減資、變更面額）都要有證交所的股數比例，而且照官方公式用這個比例算回的參考價，
    要等於官方參考價（FinMind 的除權息、減資、分割紀錄），誤差只能來自兩邊公布的位數。

    沒有比例的事件（例如 ETF 分割，證交所沒有公布比例）以前，還原成交量算不出來：不是資料錯誤，只提醒。
    """
    name = "成交量還原（股數變動）"
    changes = loader.share_changes(stock_id)
    if changes.empty:
        return Check(name, "ok", "沒有除權、減資、分割，成交量不用還原")
    problems, missing = [], []
    for _, ev in changes.iterrows():
        label = f"{ev['date']:%Y-%m-%d} {ev['source']}"
        if np.isnan(ev["ratio"]):
            missing.append(f"{label}：沒有證交所的股數比例")
            continue
        if np.isnan(ev["rights_ratio"]):
            missing.append(
                f"{label}：有現金增資、但沒有原股東認購，算不出現金增資配股率，沒辦法用公式對照參考價"
            )
            continue
        num = ev["before"] - ev["cash"] + ev["rights_price"] * ev["rights_ratio"]
        den = ev["ratio"] + ev["rights_ratio"]
        expected = num / den
        # 比例少 RIGHTS_RATIO_PRECISION 時，算回的參考價最多高多少
        slack = RIGHTS_RATIO_PRECISION if ev["source"] == "除權" else 0.0
        tolerance = PRICE_PRECISION + num * slack / (den * (den - slack))
        if abs(expected - ev["reference"]) >= tolerance:
            problems.append(
                f"{label}：每股變成 {ev['ratio']:g} 股，依官方公式算回的參考價 {expected:.4f}，"
                f"官方參考價 {ev['reference']:g}"
            )
    no_ratio = changes["ratio"].isna()
    parts = [f"{len(changes)} 次股數改變"]
    if (~no_ratio).any():
        checked = int((~no_ratio & changes["rights_ratio"].notna()).sum())
        parts.append(
            f"{int((~no_ratio).sum())} 次有證交所的股數比例，其中 {checked} 次依官方公式算回的參考價"
            + ("有不同" if problems else "都和官方相符")
        )
    if no_ratio.any():
        parts.append(
            f"{int(no_ratio.sum())} 次沒有比例，還原價格在 "
            f"{changes.loc[no_ratio, 'date'].max():%Y-%m-%d} 以前的成交量算不出來"
        )
    summary = "；".join(parts)
    status = "fail" if problems else "warn" if missing else "ok"
    return Check(name, status, summary, problems + missing)


def _on_tick(price: float, etf: bool) -> bool:
    """價格對齊升降單位（以「分」計算，避免浮點誤差）。"""
    cents = price * 100
    whole = round(cents)
    return abs(cents - whole) < 1e-6 and whole % tick_size(whole, etf) == 0


def check_tick_grid(stock_id: str, loader: DataLoader) -> Check:
    """每天（HISTORY_START 起）的開高低收都要在參考價的漲跌停範圍內，且對齊升降單位。

    參考價：除權息、減資、分割當天用官方參考價，其他日子用前一個交易日的收盤價（暫停交易的日子不算）。
    新上市股票前幾個交易日沒有漲跌幅限制，只檢查升降單位。
    官方有公布當天漲跌停價的日子，也比對程式算出來的漲跌停價。
    """
    name = "升降單位與漲跌停"
    p = loader.prices(stock_id, adjust=False)
    actions = (
        loader.corporate_actions(stock_id)
        .drop_duplicates("date", keep="last")
        .set_index("date")
    )
    etf = stock_id.startswith("00")
    no_limit = loader.no_limit_days(stock_id)
    problems, checked, official = [], 0, 0
    prev_close = p["close"].shift(1)
    for d, row in p.iterrows():
        if d in actions.index:
            ref, label = (
                float(actions.at[d, "reference"]),
                f"{actions.at[d, 'source']}參考價",
            )
        elif not np.isnan(prev_close[d]):
            ref, label = float(prev_close[d]), "前收"
        else:
            continue
        grid = candidate_closes(ref, etf)
        checked += 1
        if d in no_limit:
            bad = [
                c
                for c in ("open", "high", "low", "close")
                if not _on_tick(float(row[c]), etf)
            ]
            where = "不在升降單位的格點上（新上市沒有漲跌幅限制）"
        else:
            bad = [
                c
                for c in ("open", "high", "low", "close")
                if float(row[c]) not in set(grid)
            ]
            where = "不在範圍或格點上"
        if bad:
            problems.append(
                f"{d:%Y-%m-%d} {label} {ref:g}：{'、'.join(f'{c}={row[c]:g}' for c in bad)} {where}"
            )
        if (
            d in actions.index
            and actions.at[d, "limit_up"] > 0
            and actions.at[d, "limit_down"] > 0
        ):
            official += 1
            ours, theirs = (
                (grid[-1], grid[0]),
                (float(actions.at[d, "limit_up"]), float(actions.at[d, "limit_down"])),
            )
            if ours != theirs:
                problems.append(
                    f"{d:%Y-%m-%d} 漲跌停價：程式算 {ours[0]:g}／{ours[1]:g}，官方 {theirs[0]:g}／{theirs[1]:g}"
                )
    summary = f"檢查 {checked} 個交易日（{HISTORY_START:%Y-%m-%d} 起），其中 {official} 天和官方公布的漲跌停價比對"
    return Check(name, "fail" if problems else "ok", summary, problems)


def verify_stock(
    stock_id: str,
    months: int = 3,
    chip_days: int = 3,
    loader: DataLoader | None = None,
    twse: TwseClient | None = None,
) -> list[Check]:
    """twse 可以由呼叫端傳入，多檔股票共用同一天的三大法人、融資融券、外資持股資料。"""
    loader = loader or DataLoader()
    twse = twse or TwseClient()
    raw = loader.read("TaiwanStockPrice", stock_id)
    if raw.empty:
        raise DataError(
            f"本機沒有 {stock_id} 的股價資料，請先執行 hermes fetch {stock_id}"
        )
    market = loader.info(stock_id).get("market", "")
    return [
        check_official(stock_id, raw, market, months, twse),
        check_valuation(stock_id, loader, market, months, twse),
        check_chips(stock_id, loader, market, chip_days, twse),
        check_adjustment(stock_id, loader),
        check_share_changes(stock_id, loader),
        check_tick_grid(stock_id, loader),
    ]
