"""歷史買賣點：讀策略設定檔，用本機資料逐日判斷買賣條件。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from hermes import chips, fundamental
from hermes.config import strategies_dir
from hermes.data import DataError, DataLoader
from hermes.strategies import (
    DATA_FIELDS,
    PRICE_FIELDS,
    RuleResult,
    RuleStrategy,
    StrategyConfigError,
    load_strategy,
    ref_parts,
    run_rules,
)
from hermes.technical import CATALOG

STRATEGY_SUFFIXES = (".yaml", ".yml")
# 判斷用的價格的白話名稱：還原模式（adjusted=True）用還原價格
PRICE_NAMES = {False: "實際成交價", True: "還原價格"}
FIELD_SOURCES = {
    "institutional": "三大法人買賣超",
    "margin": "融資融券",
    "shareholding": "外資持股",
    "valuation": "本益比、淨值比、殖利率",
}


class StrategyNotFoundError(LookupError):
    pass


def strategy_files(directory: Path | None = None) -> list[Path]:
    directory = directory or strategies_dir()
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix in STRATEGY_SUFFIXES)


def list_rule_strategies(
    directory: Path | None = None,
) -> list[tuple[Path, RuleStrategy | StrategyConfigError]]:
    """列出資料夾裡的策略設定檔；設定有誤的檔案回傳錯誤，不中斷整份清單。"""
    out = []
    for path in strategy_files(directory):
        try:
            out.append((path, load_strategy(path)))
        except StrategyConfigError as exc:
            out.append((path, exc))
    return out


def resolve_strategy_path(ref: str, directory: Path | None = None) -> Path:
    """ref 可以是策略資料夾裡的檔名（不含副檔名），也可以是設定檔路徑。"""
    path = Path(ref).expanduser()
    if path.suffix in STRATEGY_SUFFIXES and path.is_file():
        return path
    directory = directory or strategies_dir()
    for suffix in STRATEGY_SUFFIXES:
        candidate = directory / f"{ref}{suffix}"
        if candidate.is_file():
            return candidate
    names = "、".join(p.stem for p in strategy_files(directory)) or "沒有任何策略設定檔"
    raise StrategyNotFoundError(f"找不到策略 {ref}（{directory} 裡有：{names}）")


def load_fields(
    loader: DataLoader, stock_id: str, names: set[str], index: pd.DatetimeIndex
) -> tuple[pd.DataFrame | None, list[str]]:
    """策略用到的籌碼、本益比欄位，對齊到交易日；還有缺資料的提醒。

    沒有紀錄的交易日是空值，用到它的條件不成立，不補值。常見的情況是當天收盤後還沒公布完，
    下次 hermes update 會補上。
    """
    if not names:
        return None, []
    sources = sorted({DATA_FIELDS[n][2] for n in names})
    frames = {src: getattr(loader, src)(stock_id) for src in sources}
    warnings = []
    for src, df in frames.items():
        if df.empty:
            warnings.append(f"沒有{FIELD_SOURCES[src]}資料，用到它的條件都不會成立")
            continue
        missing = index.difference(df.index)
        if len(missing):
            recent = "、".join(f"{d:%Y-%m-%d}" for d in missing[-3:])
            warnings.append(
                f"{FIELD_SOURCES[src]}有 {len(missing)} 個交易日沒有資料（最近的：{recent}），"
                f"這些日子用到它的條件不會成立"
            )
    merged = pd.concat(
        [
            chips.daily_fields(
                *(frames.get(k) for k in ("institutional", "margin", "shareholding"))
            ),
            fundamental.daily_fields(frames.get("valuation")),
        ],
        axis=1,
    )
    return merged.reindex(index=index, columns=sorted(names)), warnings


@dataclass
class SignalHistory:
    stock_id: str
    strategy: RuleStrategy
    adjusted: bool  # True：還原價格（還原模式）；False：實際成交價
    prices: pd.DataFrame
    result: RuleResult  # 用全部歷史計算，指標才有足夠的暖機天數
    fields: pd.DataFrame | None = None  # 策略用到的籌碼、本益比欄位
    warnings: list[str] = field(default_factory=list)  # 缺資料的提醒

    @property
    def indicator_columns(self) -> list[str]:
        return [c for c in self.result.values.columns if c not in PRICE_FIELDS]

    def table(
        self,
        start: date | None = None,
        end: date | None = None,
        every_day: bool = False,
    ) -> pd.DataFrame:
        """區間內每天的價格、指標數值、買賣條件與連續天數；預設只留下有訊號的日子。"""
        out = self.result.values.copy()
        out["buy"], out["buy_streak"] = self.result.buy, self.result.buy_streak
        out["sell"], out["sell_streak"] = self.result.sell, self.result.sell_streak
        out = out.loc[
            pd.Timestamp(start) if start else None : pd.Timestamp(end) if end else None
        ]
        return out if every_day else out[out["buy"] | out["sell"]]

    def last_trigger(self, side: str) -> pd.Timestamp | None:
        """最近一次新觸發（連續天數為 1）的日期。"""
        s = self.result.buy_streak if side == "buy" else self.result.sell_streak
        hits = s.index[s == 1]
        return hits[-1] if len(hits) else None


def signal_history(
    stock_id: str,
    strategy_ref: str,
    adjust: bool = False,
    loader: DataLoader | None = None,
    directory: Path | None = None,
) -> SignalHistory:
    """歷史買賣點：預設用實際成交價判斷；adjust=True 是還原模式，用還原價格判斷。"""
    strategy = load_strategy(resolve_strategy_path(strategy_ref, directory))
    loader = loader or DataLoader()
    prices = loader.prices(stock_id, adjust=adjust)
    fields, warnings = load_fields(loader, stock_id, strategy.data_fields, prices.index)
    hist = history(stock_id, strategy, adjust, prices, fields)
    hist.warnings += warnings + missing_volume_warnings(loader, hist)
    return hist


def history(
    stock_id: str,
    strategy: RuleStrategy,
    adjusted: bool,
    prices: pd.DataFrame,
    fields: pd.DataFrame | None = None,
) -> SignalHistory:
    return SignalHistory(
        stock_id,
        strategy,
        adjusted,
        prices,
        run_rules(strategy, prices, fields),
        fields,
    )


def missing_volume_warnings(loader: DataLoader, hist: SignalHistory) -> list[str]:
    """還原價格的成交量有算不出來的日子（股數改變、但沒有證交所股數比例，例如 ETF 分割），
    而策略的指標或條件用到它：這段時間的指標算不出來，用到它的條件不成立，回測也要等它算得出來才開始，要提醒。

    「用到」的判斷：和實際成交價比，有哪個策略用到的值，實際成交價算得出來、還原價格算不出來。
    指標沒用在條件裡也算，因為暖機天數（warmup_end）也看它。
    「算得出來」看 RuleResult.ready：型態類指標（例如 Darvas 箱）沒有值是「當時沒有箱子」，不是算不出來。
    """
    if not hist.adjusted or not hist.prices["volume"].isna().any():
        return []
    strategy = hist.strategy
    raw = run_rules(strategy, loader.prices(hist.stock_id, adjust=False), hist.fields)
    used = {
        ref_parts(o)[0] for cond in strategy.conditions() for o in (cond.left, cond.right)
    }
    cols = [c for c in raw.values.columns if c not in PRICE_FIELDS or c in used]
    if not (raw.ready[cols] & ~hist.result.ready[cols]).any().any():
        return []
    return [
        f"{adjusted_volume_note(loader, hist.stock_id)}；{strategy.name} 用到成交量，"
        f"這天以前的指標算不出來，用到它的條件不會成立"
    ]


def gap_warnings(
    loader: DataLoader, stock_id: str, adjusted: bool, returns: bool = False
) -> list[str]:
    """沒有官方紀錄的跳空：不估算還原，只提醒。

    影響的是還原價格：還原模式的判斷，以及回測報酬（returns=True，兩種模式都用還原價格算報酬）。
    實際成交價模式的訊號、選股、指標用不到還原價格，不提醒。
    """
    if not adjusted and not returns:
        return []
    try:
        gaps = loader.unexplained_gaps(stock_id)
    except DataError:
        return []
    affected = "還原價格" if adjusted else "回測報酬"
    return [
        f"{g['date']:%Y-%m-%d} 跳空 {g['gap']:+.1%} 超過漲跌幅限制，"
        f"但沒有官方除權息／減資／分割紀錄，{affected}可能不正確"
        for _, g in gaps.iterrows()
    ]


def adjusted_volume_note(loader: DataLoader, stock_id: str) -> str:
    """還原價格算不出來的原因，給警告訊息用。"""
    changes = loader.share_changes(stock_id)
    missing = changes[changes["ratio"].isna()]
    if missing.empty:
        return "還原價格有些值算不出來"
    last = missing.iloc[-1]
    return (
        f"{last['date']:%Y-%m-%d} {last['source']}，股數改變，沒有證交所的股數比例，"
        f"還原價格在這天以前的成交量算不出來"
    )


@dataclass
class ChartLayout:
    """歷史買賣點圖要畫什麼：CLI 的 PNG 和網頁的互動圖用同一份。"""

    overlays: list[str]  # 跟股價同單位的指標欄位（均線、布林通道…），畫在價格圖上
    panels: dict[str, list[str]]  # 其他指標、籌碼和本益比欄位，各自一個子圖：{子圖名稱: [欄位]}
    # 策略條件裡和數字比較的門檻：{"price": 價格圖, "volume": 成交量, 子圖名稱: 該子圖}；沒有寫在策略裡的不畫
    thresholds: dict[str, list[float]]
    # 策略設定檔 reference_lines 寫的參考線，只畫線、不影響買賣；鍵和 thresholds 一樣
    reference_lines: dict[str, list[float]]


def chart_layout(strategy: RuleStrategy) -> ChartLayout:
    """多條線的指標只畫買賣條件用到的線（例如海龜只畫 dc55.upper、dc20.lower）；一條都沒用到時全部畫。

    形成中的條件不影響買賣，不看它。
    """
    overlays: list[str] = []
    panels: dict[str, list[str]] = {}
    used = {
        ref_parts(operand)[0]
        for cond in strategy.conditions(forming=False)
        for operand in (cond.left, cond.right)
    }
    for name, cfg in strategy.indicators.items():
        spec = CATALOG[cfg.type]
        lines = [line for line in spec.outputs if f"{name}.{line}" in used]
        for line in lines or spec.outputs or (None,):
            col = f"{name}.{line}" if line else name
            if spec.on_price_chart(line):
                overlays.append(col)
            else:
                panels.setdefault(name, []).append(col)
    for name in sorted(used & DATA_FIELDS.keys()):  # 籌碼、本益比各自一個子圖
        panels[name] = [name]
    on_price = {"open", "high", "low", "close"} | {c.split(".")[0] for c in overlays}
    levels: dict[str, set[float]] = {}
    for cond in strategy.conditions(forming=False):
        if isinstance(cond.right, float):
            base = ref_parts(cond.left)[0].split(".")[0]
            key = "price" if base in on_price else base
            levels.setdefault(key, set()).add(cond.right)
    refs: dict[str, set[float]] = {}
    for name, values in strategy.reference_lines.items():
        refs.setdefault("price" if name in on_price else name, set()).update(values)
    return ChartLayout(
        overlays,
        panels,
        {k: sorted(v) for k, v in levels.items()},
        {k: sorted(v) for k, v in refs.items()},
    )
