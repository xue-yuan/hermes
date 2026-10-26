"""自選股清單，存在 SQLite（預設 $HERMES_HOME/hermes.db）。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

from hermes.config import db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    stock_id TEXT PRIMARY KEY,
    name     TEXT NOT NULL DEFAULT '',
    added_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class WatchItem:
    stock_id: str
    name: str
    added_at: datetime


class Watchlist:
    def __init__(self, path: Path | None = None):
        self._conn = sqlite3.connect(path or db_path())
        self._conn.executescript(SCHEMA)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def add(self, stock_id: str, name: str = "") -> bool:
        """加入清單；已經在清單中時回傳 False。"""
        with self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO watchlist (stock_id, name, added_at) VALUES (?, ?, ?)",
                (stock_id, name, datetime.now().isoformat(timespec="seconds")),
            )
        return cur.rowcount == 1

    def remove(self, stock_id: str) -> bool:
        """移出清單；本來就不在清單中時回傳 False。"""
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM watchlist WHERE stock_id = ?", (stock_id,)
            )
        return cur.rowcount == 1

    def items(self) -> list[WatchItem]:
        rows = self._conn.execute(
            "SELECT stock_id, name, added_at FROM watchlist ORDER BY stock_id"
        )
        return [
            WatchItem(sid, name, datetime.fromisoformat(added))
            for sid, name, added in rows
        ]

    def stock_ids(self) -> list[str]:
        return [item.stock_id for item in self.items()]
