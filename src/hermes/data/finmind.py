"""FinMind Open Data API 客戶端（https://finmindtrade.com）。

免費等級不需 token（每小時約 300 次請求）；設定環境變數 FINMIND_TOKEN 可提高額度。
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests

from hermes.data.errors import DataError

API_URL = "https://api.finmindtrade.com/api/v4/data"


class FinMindError(DataError):
    pass


class FinMindRateLimitError(FinMindError):
    """請求次數已達上限；繼續請求也只會失敗，呼叫端應該停止。"""


class FinMindClient:
    def __init__(
        self, token: str | None = None, timeout: float = 60.0, max_retries: int = 3
    ):
        self.token = token if token is not None else os.environ.get("FINMIND_TOKEN")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def get(
        self,
        dataset: str,
        data_id: str | None = None,
        start_date: str = "1990-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        params = {"dataset": dataset, "start_date": start_date}
        if data_id:
            params["data_id"] = data_id
        if end_date:
            params["end_date"] = end_date
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(
                    API_URL, params=params, headers=headers, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(2**attempt)
                continue
            if resp.status_code in (402, 429):
                raise FinMindRateLimitError(
                    "FinMind 請求次數已達上限，請稍後再試，或設定 FINMIND_TOKEN 環境變數提高額度。"
                )
            try:
                payload = resp.json()
            except ValueError as exc:
                last_error = exc
                time.sleep(2**attempt)
                continue
            if payload.get("status") != 200:
                raise FinMindError(f"{dataset}({data_id}): {payload.get('msg')}")
            return pd.DataFrame(payload.get("data", []))
        raise FinMindError(f"{dataset}({data_id}) 連線失敗：{last_error}")
