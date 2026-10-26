"""refresh token 的紀錄：輪替、重複使用時撤銷整串、閒置太久過期（RFC 9700 §4.14）。

refresh token 是隨機字串，資料庫只存它的 HMAC（用 session_secret），資料庫外流也拿不到可用的 token；
換掉 session_secret 後，舊的 refresh token 全部失效。

同一次登入發出去的 refresh token 屬於同一串（family）。每次換發都把舊的標成已換過，發一個新的；
已換過的 token 又被拿來用，代表 token 可能被偷了，整串都撤銷，兩邊都要重新登入。

例外：換過之後 30 秒內還可以再用（寬限期）。換發的回應可能沒送到瀏覽器，例如換發途中重新整理頁面，
瀏覽器手上還是舊的 token；沒有寬限期的話，這種情況會被當成被偷而登出。做法和預設值照 Okta 的
「Grace period for token rotation」（預設 30 秒，可設 0～60 秒）。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path

ROTATION_GRACE_SECONDS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS refresh_tokens (
    token_hash TEXT PRIMARY KEY,
    family     TEXT NOT NULL,
    email      TEXT NOT NULL,
    issued_at  REAL NOT NULL,
    expires_at REAL NOT NULL,
    replaced   INTEGER NOT NULL DEFAULT 0,
    replaced_at REAL,
    revoked    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS refresh_tokens_family ON refresh_tokens (family);
"""


class RefreshError(Exception):
    """refresh token 不能用：不存在、過期、撤銷，或重複使用。訊息給使用者看。"""


@dataclass(frozen=True)
class Issued:
    email: str
    token: str
    expires_at: float


class RefreshStore:
    def __init__(self, path: Path, secret: str):
        self.path = path
        self._key = secret.encode()

    def _hash(self, token: str) -> str:
        return hmac.new(self._key, token.encode(), hashlib.sha256).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.executescript(SCHEMA)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(refresh_tokens)")}
        if "replaced_at" not in columns:  # 加上寬限期以前建立的資料庫
            conn.execute("ALTER TABLE refresh_tokens ADD COLUMN replaced_at REAL")
        return conn

    def _insert(self, conn, family: str, email: str, now: float, lifetime: float) -> Issued:
        token = secrets.token_urlsafe(32)
        conn.execute(
            "INSERT INTO refresh_tokens (token_hash, family, email, issued_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (self._hash(token), family, email, now, now + lifetime),
        )
        return Issued(email, token, now + lifetime)

    def create(self, email: str, now: float, lifetime: float) -> Issued:
        """登入：開一串新的 refresh token。順便清掉已經過期的紀錄。"""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM refresh_tokens WHERE expires_at <= ?", (now,))
            issued = self._insert(conn, secrets.token_urlsafe(16), email, now, lifetime)
            conn.execute("COMMIT")
            return issued
        finally:
            conn.close()

    def rotate(self, token: str, now: float, lifetime: float, grace: float = ROTATION_GRACE_SECONDS) -> Issued:
        """換發：舊的作廢，發一個同一串的新 token，有效期限從現在重新算（閒置超過 lifetime 才過期）。

        舊的在作廢後 grace 秒內還可以再換一次（見上面的寬限期），超過才當成重複使用，整串撤銷。
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")  # 同時有兩個請求拿同一個 token 來換，只有一個會成功
            row = conn.execute(
                "SELECT family, email, expires_at, replaced, replaced_at, revoked FROM refresh_tokens "
                "WHERE token_hash = ?",
                (self._hash(token),),
            ).fetchone()
            if row is None or row[5]:
                conn.execute("COMMIT")
                raise RefreshError("請重新登入")
            family, email, expires_at, replaced, replaced_at, _ = row
            if replaced and not (replaced_at is not None and now - replaced_at <= grace):
                conn.execute("UPDATE refresh_tokens SET revoked = 1 WHERE family = ?", (family,))
                conn.execute("COMMIT")
                raise RefreshError("登入憑證被重複使用，為了安全已全部登出，請重新登入")
            if expires_at <= now:
                conn.execute("COMMIT")
                raise RefreshError("太久沒有使用，請重新登入")
            if not replaced:  # 寬限期內再換一次時，作廢的時間不變，寬限期不會被延長
                conn.execute(
                    "UPDATE refresh_tokens SET replaced = 1, replaced_at = ? WHERE token_hash = ?",
                    (now, self._hash(token)),
                )
            issued = self._insert(conn, family, email, now, lifetime)
            conn.execute("COMMIT")
            return issued
        finally:
            conn.close()

    def revoke(self, token: str) -> None:
        """登出：撤銷這個 token 所在的整串。"""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE refresh_tokens SET revoked = 1 WHERE family = "
                "(SELECT family FROM refresh_tokens WHERE token_hash = ?)",
                (self._hash(token),),
            )
        finally:
            conn.close()

    def revoke_email(self, email: str) -> None:
        """不在允許名單了：撤銷這個帳號所有的 refresh token。"""
        conn = self._connect()
        try:
            conn.execute("UPDATE refresh_tokens SET revoked = 1 WHERE email = ?", (email,))
        finally:
            conn.close()
