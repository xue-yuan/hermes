"""策略：組合技術面、籌碼面、本益比、機器學習算出的數值，判斷買賣點。"""

from hermes.strategies import benchmark, ml  # noqa: F401  註冊策略
from hermes.strategies.base import REGISTRY, StrategyContext, StrategySpec, get_strategy, parse_params
from hermes.strategies.engine import RuleResult, indicator_frame, ready_frame, run_rules, streak
from hermes.strategies.rules import (
    DATA_FIELDS,
    PRICE_FIELDS,
    Ago,
    IndicatorSet,
    RuleStrategy,
    StrategyConfigError,
    describe,
    load_indicator_set,
    load_strategy,
    operand_label,
    operand_text,
    parse_strategy,
    ref_parts,
    rule_operands,
    walk,
)
from hermes.strategies.ticks import candidate_closes, tick_size

__all__ = [
    "DATA_FIELDS",
    "PRICE_FIELDS",
    "Ago",
    "IndicatorSet",
    "REGISTRY",
    "RuleResult",
    "RuleStrategy",
    "StrategyConfigError",
    "StrategyContext",
    "StrategySpec",
    "candidate_closes",
    "describe",
    "get_strategy",
    "indicator_frame",
    "load_indicator_set",
    "load_strategy",
    "operand_label",
    "operand_text",
    "parse_params",
    "parse_strategy",
    "ready_frame",
    "ref_parts",
    "rule_operands",
    "run_rules",
    "streak",
    "tick_size",
    "walk",
]
