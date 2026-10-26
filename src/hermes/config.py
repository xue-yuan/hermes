"""全域設定：資料目錄、資料起始日等常數。"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# 資料只保留這天之後：下載從這天開始，本機較舊的資料讀取時也會略過
HISTORY_START = date(2020, 1, 1)


def home_dir() -> Path:
    """資料根目錄，預設 ~/.hermes，可用環境變數 HERMES_HOME 覆寫。"""
    path = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    """應用狀態（自選股等）的 SQLite 資料庫位置。"""
    return home_dir() / "hermes.db"


def strategies_dir() -> Path:
    """策略設定檔資料夾，預設為目前工作目錄下的 strategies/（跟著專案用 git 管理）。"""
    return Path(
        os.environ.get("HERMES_STRATEGIES", Path.cwd() / "strategies")
    ).expanduser()


def indicators_dir() -> Path:
    """指標組設定檔資料夾，預設為目前工作目錄下的 indicators/，例如常用版 common.yaml。"""
    return Path(
        os.environ.get("HERMES_INDICATORS", Path.cwd() / "indicators")
    ).expanduser()


def reports_dir() -> Path:
    """圖表與報表輸出目錄，預設為目前工作目錄下的 reports/。"""
    path = Path(os.environ.get("HERMES_REPORTS", Path.cwd() / "reports")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path
