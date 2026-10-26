"""檢查模組之間的依賴方向。

只允許上層依賴下層；計算數值的模組（technical、chips、fundamental、ml）不能依賴
data（資料由上層傳入），也不能依賴 strategies（買賣判斷屬於策略）。
"""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "hermes"

ALLOWED = {
    "config": set(),
    "data": {"config"},
    "technical": set(),
    "chips": set(),
    "fundamental": set(),
    "ml": {"technical"},
    "backtest": {"config"},
    "strategies": {"technical", "chips", "fundamental", "ml"},
    "services": {"config", "data", "technical", "chips", "fundamental", "ml", "backtest", "strategies"},
    "cli": {"config", "data", "technical", "chips", "fundamental", "ml", "backtest", "strategies", "services"},
    "api": {"config", "data", "technical", "chips", "fundamental", "ml", "backtest", "strategies", "services"},
}


def _module_of(path: Path) -> str:
    rel = path.relative_to(PACKAGE_ROOT)
    return rel.parts[0].removesuffix(".py")


def _hermes_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module] if node.module != "hermes" else [f"hermes.{a.name}" for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "hermes" and len(parts) > 1:
                found.add(parts[1])
    return found


def test_every_module_has_a_rule():
    modules = {_module_of(p) for p in PACKAGE_ROOT.rglob("*.py")} - {"__init__"}
    assert modules <= set(ALLOWED), f"新模組需要在 ALLOWED 定義依賴規則：{modules - set(ALLOWED)}"


def test_dependency_direction():
    violations = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        module = _module_of(path)
        if module == "__init__":
            continue
        for dep in _hermes_imports(path) - ALLOWED[module] - {module}:
            violations.append(f"{path.relative_to(PACKAGE_ROOT)} 不能依賴 hermes.{dep}")
    assert not violations, "\n".join(violations)
