"""網頁的設定，從環境變數讀（hermes-web 會先載入專案根目錄的 .env）；缺必要的設定就不啟動。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ACCESS_MINUTES = 15  # access token 的有效分鐘數，可用 HERMES_ACCESS_MINUTES 改
REFRESH_HOURS = 168  # refresh token 閒置多久過期（7 天沒用就要重新登入），可用 HERMES_REFRESH_HOURS 改


class SettingsError(RuntimeError):
    pass


@dataclass(frozen=True)
class WebSettings:
    google_client_id: str  # Google Cloud 主控台建立的 OAuth 用戶端 ID（網頁應用程式）
    allowed_emails: frozenset[str]  # 可以登入的 Google 帳號；其他帳號一律拒絕
    session_secret: str  # 簽署 access token（JWT）、雜湊 refresh token 用
    access_minutes: float = ACCESS_MINUTES
    refresh_hours: float = REFRESH_HOURS
    secure_cookie: bool = True  # refresh token 的 cookie 只在 HTTPS 送出；本機用 http 開發時才關掉
    static_dir: Path | None = None  # 前端 build 出來的檔案（web/dist）；None 時只提供 API
    sessions_path: Path | None = None  # refresh token 的紀錄；None 時是 $HERMES_HOME/web_sessions.db

    @classmethod
    def from_env(cls) -> WebSettings:
        env = os.environ
        missing = [
            name
            for name in (
                "HERMES_GOOGLE_CLIENT_ID",
                "HERMES_ALLOWED_EMAILS",
                "HERMES_SESSION_SECRET",
            )
            if not env.get(name, "").strip()
        ]
        if missing:
            raise SettingsError(
                f"網頁設定不完整，請在專案根目錄的 .env 填上：{'、'.join(missing)}（說明見 .env.example）"
            )
        emails = frozenset(
            e.strip().lower() for e in env["HERMES_ALLOWED_EMAILS"].split(",") if e.strip()
        )
        secret = env["HERMES_SESSION_SECRET"].strip()
        if len(secret) < 32:
            raise SettingsError("HERMES_SESSION_SECRET 至少要 32 個字元")
        minutes = _positive(env, "HERMES_ACCESS_MINUTES", ACCESS_MINUTES)
        hours = _positive(env, "HERMES_REFRESH_HOURS", REFRESH_HOURS)
        static = Path(env.get("HERMES_WEB_DIST", Path.cwd() / "web" / "dist")).expanduser()
        return cls(
            google_client_id=env["HERMES_GOOGLE_CLIENT_ID"].strip(),
            allowed_emails=emails,
            session_secret=secret,
            access_minutes=minutes,
            refresh_hours=hours,
            secure_cookie=env.get("HERMES_INSECURE_COOKIE", "") != "1",
            static_dir=static if (static / "index.html").is_file() else None,
        )


def _positive(env, name: str, default: float) -> float:
    try:
        value = float(env.get(name, default))
    except ValueError:
        raise SettingsError(f"{name} 要是數字") from None
    if value <= 0:
        raise SettingsError(f"{name} 要大於 0")
    return value
