"""啟動網頁：uv run hermes-web（設定寫在專案根目錄的 .env，範本是 .env.example）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def load_settings(directory: Path | None = None):
    """先載入 directory（預設目前目錄）裡的 .env，再讀環境變數；已經設定的環境變數不會被 .env 蓋掉。"""
    from dotenv import load_dotenv

    from hermes.api.settings import WebSettings

    load_dotenv((directory or Path.cwd()) / ".env", override=False)
    return WebSettings.from_env()


def main() -> None:
    import uvicorn

    from hermes.api.app import create_app
    from hermes.api.settings import SettingsError

    parser = argparse.ArgumentParser(description="hermes 網頁（只能看，不會下載或修改資料）")
    # 預設用 localhost：啟動時印出的網址要和 Google「已授權的 JavaScript 來源」一致，用 127.0.0.1 會被 Google 擋掉
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        settings = load_settings()
    except SettingsError as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        sys.exit(1)
    if settings.static_dir is None:
        print("找不到前端檔案（web/dist），只提供 API；先執行 cd web && pnpm install && pnpm build", file=sys.stderr)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, proxy_headers=True)
