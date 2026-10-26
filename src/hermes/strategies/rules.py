"""規則型策略的設定檔格式：定義要用的指標，再用結構化條件描述買賣點。

    name: CCI 超賣反轉
    indicators:
      cci:  {type: cci, period: 88}
      ma60: {type: sma, period: 60}
    buy:
      all:
        - {left: cci,   op: crosses_above, right: -100}
        - {left: close, op: gt,            right: ma60}
    sell:
      any:
        - {left: cci,   op: crosses_below, right: 100}
        - {left: close, op: lt, right: ma60, days: 3}

要比前 N 天的值時，把名稱寫成 {ref: 名稱, ago: N}，例如收盤突破前一天的唐奇安通道上緣：
    - {left: close, op: crosses_above, right: {ref: dc.upper, ago: 1}}

籌碼、本益比這些數值（DATA_FIELDS）可以直接寫名稱，例如外資連續 3 天買超：
    - {left: foreign_net, op: gt, right: 0, days: 3}

forming（選填）是「形成中」的條件，寫法和 buy 一樣：成立、而且當天沒有觸發買進時，
每日選股列在「形成中」，例如 Darvas 箱已經成形、價格還在箱子裡。只用來顯示，不影響買賣和回測。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    ValidationError,
    model_validator,
)

from hermes.chips import FIELDS as CHIP_FIELDS
from hermes.fundamental import FIELDS as VALUATION_FIELDS
from hermes.technical import CATALOG

PRICE_FIELDS = ("open", "high", "low", "close", "volume")
# 籌碼、本益比：名稱 →（說明, 單位, 來源資料）。都是當天收盤後才公布
DATA_FIELDS = {**CHIP_FIELDS, **VALUATION_FIELDS}
STATE_OPS = ("gt", "gte", "lt", "lte")  # 條件成立的每一天都算
EVENT_OPS = ("crosses_above", "crosses_below")  # 只算剛穿越的那一天
OP_LABELS = {
    "gt": ">",
    "gte": "≥",
    "lt": "<",
    "lte": "≤",
    "crosses_above": "由下往上穿越",
    "crosses_below": "由上往下穿越",
}


class StrategyConfigError(ValueError):
    """策略設定檔有誤；訊息列出每個錯誤的位置，給使用者看。"""


class Ago(BaseModel):
    """前 N 天的值：{ref: dc.upper, ago: 1} 是前一天的 dc.upper；ago: 0 等於直接寫名稱。"""

    model_config = ConfigDict(extra="forbid")

    ref: str
    ago: int = Field(default=0, ge=0)


def _operand_kind(value: object) -> str | None:
    if isinstance(value, (dict, Ago)):
        return "previous"
    if isinstance(value, str):
        return "name"
    return (
        "number"
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


# 條件兩邊可以寫的東西：名稱（指標、價格欄位）、前 N 天的值；右邊還可以是數字
Ref = Annotated[
    Annotated[str, Tag("name")] | Annotated[Ago, Tag("previous")],
    Discriminator(_operand_kind),
]
Operand = Annotated[
    Annotated[float, Tag("number")]
    | Annotated[str, Tag("name")]
    | Annotated[Ago, Tag("previous")],
    Discriminator(_operand_kind),
]


def ref_parts(operand: float | str | Ago) -> tuple[str | None, int]:
    """（名稱, 前幾天）；數字回傳 (None, 0)。"""
    if isinstance(operand, Ago):
        return operand.ref, operand.ago
    return (operand, 0) if isinstance(operand, str) else (None, 0)


def operand_label(operand: float | str | Ago) -> str:
    """條件裡的值在 RuleResult.values 的欄位名稱，例如 dc.upper（前1天）；數字回傳數字本身。"""
    name, ago = ref_parts(operand)
    if name is None:
        return f"{operand:g}"
    return f"{name}（前{ago}天）" if ago else name


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: Ref
    op: Literal["gt", "gte", "lt", "lte", "crosses_above", "crosses_below"]
    right: Operand
    days: int = Field(default=1, ge=1)  # 連續幾天成立才算

    @model_validator(mode="after")
    def _days_only_for_state(self) -> Condition:
        if self.days > 1 and self.op in EVENT_OPS:
            raise ValueError(
                "穿越是單日事件，不能搭配 days；要「連續 N 天在上方／下方」請用 gt、lt"
            )
        return self


class Group(BaseModel):
    model_config = ConfigDict(extra="forbid")

    all: list[Rule] | None = None  # 全部成立（而且）
    any: list[Rule] | None = None  # 任一成立（或者）

    @model_validator(mode="after")
    def _exactly_one(self) -> Group:
        if (self.all is None) == (self.any is None):
            raise ValueError("all 和 any 要剛好寫一個")
        if not self.rules:
            raise ValueError("條件清單不能是空的")
        return self

    @property
    def rules(self) -> list[Rule]:
        return self.all if self.all is not None else self.any


def _rule_kind(value: object) -> str:
    if isinstance(value, dict):
        return "group" if ("all" in value or "any" in value) else "condition"
    return "group" if isinstance(value, Group) else "condition"


Rule = Annotated[
    Annotated[Condition, Tag("condition")] | Annotated[Group, Tag("group")],
    Discriminator(_rule_kind),
]
Group.model_rebuild()


class IndicatorConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # 其餘欄位都是指標參數

    type: str

    @property
    def params(self) -> dict:
        return dict(self.model_extra or {})


def _check_indicators(indicators: dict[str, IndicatorConfig]) -> None:
    for name, cfg in indicators.items():
        if name in PRICE_FIELDS or name in DATA_FIELDS or "." in name:
            raise ValueError(
                f"指標名稱 {name} 不能和價格欄位、籌碼或本益比欄位相同，也不能有點號"
            )
        spec = CATALOG.get(cfg.type)
        if spec is None:
            raise ValueError(
                f"指標 {name} 的類型 {cfg.type} 不存在（hermes indicators 可以查看所有類型）"
            )
        try:
            spec.resolve_params(cfg.params)
        except ValueError as exc:
            raise ValueError(f"指標 {name}：{exc}") from None


class IndicatorSet(BaseModel):
    """指標組（indicators/*.yaml）：一組有名稱的指標和參數，例如常用版 common.yaml。

    同一種指標要用不同參數時，取不同名稱即可，例如 rsi14 和 rsi_mine。
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    indicators: dict[str, IndicatorConfig]

    @model_validator(mode="after")
    def _check(self) -> IndicatorSet:
        if not self.indicators:
            raise ValueError("indicators 不能是空的")
        _check_indicators(self.indicators)
        return self


class RuleStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    indicators: dict[str, IndicatorConfig] = Field(default_factory=dict)
    buy: Rule
    sell: Rule | None = None
    forming: Rule | None = None  # 形成中：成立、而且當天沒有觸發買進時，列在每日選股的「形成中」
    # 圖上另外畫的參考線，只畫線、不影響買賣：{指標名稱: [數值]}，例如 {cci88: [-100, 100]}
    reference_lines: dict[str, list[float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_references(self) -> RuleStrategy:
        _check_indicators(self.indicators)
        for name in self.reference_lines:
            if name not in self.indicators:
                raise ValueError(
                    f"reference_lines 用到 {name}，但 indicators 裡沒有定義它"
                )

        operands = self.operands()
        for side, rule in (
            ("buy", self.buy),
            ("sell", self.sell),
            ("forming", self.forming),
        ):
            for cond in walk(rule):
                for operand in (cond.left, cond.right):
                    ref, _ = ref_parts(operand)
                    if ref is not None and ref not in operands:
                        raise ValueError(f"{side} 用到 {ref}，但{self._hint(ref)}")
        return self

    def operands(self) -> set[str]:
        """條件裡可以使用的名稱：價格欄位、籌碼和本益比欄位、單線指標名稱，以及多線指標的「名稱.線」。"""
        names = set(PRICE_FIELDS) | set(DATA_FIELDS)
        for name, cfg in self.indicators.items():
            outputs = CATALOG[cfg.type].outputs
            if outputs:
                names.update(f"{name}.{o}" for o in outputs)
            else:
                names.add(name)
        return names

    def conditions(self, forming: bool = True) -> list[Condition]:
        """買進、賣出的每一個條件；forming=True 時也包含形成中的條件。"""
        rules = (self.buy, self.sell, self.forming if forming else None)
        return [cond for rule in rules for cond in walk(rule)]

    def _refs(self, forming: bool = True) -> Iterator[tuple[str, int]]:
        for cond in self.conditions(forming):
            for operand in (cond.left, cond.right):
                name, ago = ref_parts(operand)
                if name is not None:
                    yield name, ago

    @property
    def data_fields(self) -> set[str]:
        """條件用到的籌碼、本益比欄位（包含形成中的條件，計算時要讀進來）。"""
        return {name for name, _ in self._refs() if name in DATA_FIELDS}

    @property
    def same_day_fields(self) -> set[str]:
        """買賣條件用到「當天」值的籌碼、本益比欄位；這些要收盤後才公布，盤中和收盤成交時都還不知道。

        形成中的條件不影響買賣，不算在內。
        """
        return {
            name
            for name, ago in self._refs(forming=False)
            if name in DATA_FIELDS and ago == 0
        }

    def _hint(self, ref: str) -> str:
        base = ref.split(".", 1)[0]
        cfg = self.indicators.get(base)
        if cfg is not None and CATALOG[cfg.type].outputs:
            lines = "、".join(f"{base}.{o}" for o in CATALOG[cfg.type].outputs)
            return f"{base} 有多條線，請指定其中一條：{lines}"
        return (
            f"indicators 裡沒有定義它，價格欄位、籌碼和本益比欄位也沒有這個名稱"
            f"（價格欄位：{'、'.join(PRICE_FIELDS)}；其他欄位用 hermes indicators 查看）"
        )


def describe(rule: Rule | None, names: Mapping[str, str] | None = None) -> str:
    """把規則轉成一行中文，例如「cci 由下往上穿越 -100 且 close > ma60」。

    names 把設定檔裡的名稱換成白話（給網頁用），例如 {"close": "收盤價", "ma60": "簡單移動平均（60 日）"}。
    """
    if rule is None:
        return "—"
    if isinstance(rule, Condition):
        days = f"（連續 {rule.days} 天）" if rule.days > 1 else ""
        left, right = (operand_text(o, names) for o in (rule.left, rule.right))
        return f"{left} {OP_LABELS[rule.op]} {right}{days}"
    joiner = " 且 " if rule.all is not None else " 或 "
    return joiner.join(
        describe(r, names) if isinstance(r, Condition) else f"（{describe(r, names)}）"
        for r in rule.rules
    )


def rule_operands(rule: Rule | None) -> list[str | Ago]:
    """規則裡用到的名稱和前 N 天的值（不含數字），依出現順序、不重複。"""
    out: dict[str, str | Ago] = {}
    for cond in walk(rule):
        for operand in (cond.left, cond.right):
            if ref_parts(operand)[0] is not None:
                out.setdefault(operand_label(operand), operand)
    return list(out.values())


def operand_text(
    operand: float | str | Ago, names: Mapping[str, str] | None = None
) -> str:
    """條件裡的一個值的寫法，例如「前 1 天的 dc.upper」；names 把名稱換成白話。"""
    name, ago = ref_parts(operand)
    if name is None:
        return f"{operand:g}"
    shown = (names or {}).get(name, name)
    return f"前 {ago} 天的 {shown}" if ago else shown


def walk(rule: Rule | None) -> Iterator[Condition]:
    """依序列出規則裡的每一個條件。"""
    if rule is None:
        return
    if isinstance(rule, Condition):
        yield rule
    else:
        for r in rule.rules:
            yield from walk(r)


_ERROR_MESSAGES = {
    "missing": "缺少這個欄位",
    "extra_forbidden": "不認得這個欄位",
    "union_tag_invalid": "要寫指標或價格欄位的名稱，不能是數字",
    "union_tag_not_found": "要寫名稱、數字，或 {ref: 名稱, ago: 天數}（左邊不能是數字）",
}


def _format_error(err: dict) -> str:
    loc = ""
    for part in err["loc"]:
        if part in ("condition", "group", "number", "name", "previous"):
            continue
        loc += (
            f"[{part}]" if isinstance(part, int) else (f".{part}" if loc else str(part))
        )
    if err["type"] == "value_error":
        msg = str(err["ctx"]["error"])
    elif err["type"] == "literal_error":
        msg = f"只能是 {err['ctx']['expected']}"
    elif err["type"] == "greater_than_equal":
        msg = f"要大於或等於 {err['ctx']['ge']}"
    else:
        msg = _ERROR_MESSAGES.get(err["type"], err["msg"])
    return f"{loc}：{msg}" if loc else msg


def parse_strategy(data: object) -> RuleStrategy:
    try:
        return RuleStrategy.model_validate(data)
    except ValidationError as exc:
        raise StrategyConfigError(
            "\n".join(_format_error(e) for e in exc.errors())
        ) from None


def parse_indicator_set(data: object) -> IndicatorSet:
    try:
        return IndicatorSet.model_validate(data)
    except ValidationError as exc:
        raise StrategyConfigError(
            "\n".join(_format_error(e) for e in exc.errors())
        ) from None


class _UniqueKeyLoader(yaml.SafeLoader):
    """同一層有重複的名稱就報錯；YAML 預設會讓後面的默默蓋掉前面的。"""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"名稱 {key} 重複", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _load(path: Path, parse):
    try:
        data = yaml.load(
            Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
        )
    except yaml.YAMLError as exc:
        raise StrategyConfigError(f"{Path(path).name} 不是正確的 YAML：{exc}") from None
    try:
        return parse(data)
    except StrategyConfigError as exc:
        raise StrategyConfigError(f"{Path(path).name} 設定有誤：\n{exc}") from None


def load_strategy(path: Path) -> RuleStrategy:
    return _load(path, parse_strategy)


def load_indicator_set(path: Path) -> IndicatorSet:
    return _load(path, parse_indicator_set)
