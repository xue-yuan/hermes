"""登入：Google 登入只需要用戶端 ID；登入後用 JWT access token 加 refresh token。

1. 瀏覽器拿到 Google 簽發的 ID token 交給後端。後端用 Google 的公鑰驗證簽章、發給誰（aud 是自己的用戶端 ID）、
   有沒有過期，再確認 email 已驗證而且在允許名單裡。
2. 驗證通過後發兩個 token：
   - access token：短效的 JWT（RFC 7519），放在回應內容裡，前端只存在記憶體，呼叫 API 時放在
     Authorization: Bearer。驗證時只接受 HS256，檢查 typ、iss、aud、exp（RFC 8725 §3.1、§3.9、§3.11）。
   - refresh token：隨機字串，放在 HttpOnly、SameSite=Strict、只送到 /api/auth 的 cookie，JavaScript 讀不到。
     用來換新的 access token，每次換發都輪替；舊的被重複使用時整串撤銷（RFC 9700 §4.14，見 sessions.py）。
3. 每次請求、每次換發都重新檢查允許名單。
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from hermes.api.sessions import Issued, RefreshError, RefreshStore
from hermes.api.settings import WebSettings
from hermes.config import home_dir

ALGORITHM = "HS256"
ACCESS_TYP = "at+jwt"  # JWT 格式的 access token（RFC 9068 §2.1）
ISSUER = "hermes"
AUDIENCE = "hermes-web"
REFRESH_COOKIE = "hermes_refresh"
REFRESH_PATH = "/api/auth"  # refresh token 只送到換發、登出，不會跟著一般的 API 請求出去

# 驗證 Google ID token，回傳裡面的資料（email、email_verified…）；token 不對時丟出 ValueError
Verifier = Callable[[str], dict]


def google_verifier(client_id: str) -> Verifier:
    from google.auth.exceptions import GoogleAuthError
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    request = google_requests.Request()

    def verify(credential: str) -> dict:
        try:
            return id_token.verify_oauth2_token(credential, request, client_id)
        except GoogleAuthError as exc:
            raise ValueError(str(exc)) from None

    return verify


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# access token


def issue_access(email: str, settings: WebSettings, now: float) -> str:
    payload = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": email,
        "iat": int(now),
        "exp": int(now + settings.access_minutes * 60),
        "jti": secrets.token_urlsafe(12),
    }
    return jwt.encode(payload, settings.session_secret, algorithm=ALGORITHM, headers={"typ": ACCESS_TYP})


def verify_access(token: str, settings: WebSettings) -> str:
    """回傳 email；簽章、演算法、typ、發給誰、過期，任何一項不對都丟出 jwt.InvalidTokenError。"""
    if jwt.get_unverified_header(token).get("typ") != ACCESS_TYP:
        raise jwt.InvalidTokenError("不是 access token")
    claims = jwt.decode(
        token,
        settings.session_secret,
        algorithms=[ALGORITHM],
        audience=AUDIENCE,
        issuer=ISSUER,
        options={"require": ["exp", "iat", "iss", "aud", "sub"]},
    )
    return str(claims["sub"])


def settings_of(request: Request) -> WebSettings:
    return request.app.state.settings


def _unauthorized(message: str) -> HTTPException:
    return HTTPException(401, message, headers={"WWW-Authenticate": "Bearer"})


def current_user(request: Request, settings: WebSettings = Depends(settings_of)) -> str:
    """Authorization: Bearer 的 access token 驗證通過，而且還在允許名單裡，回傳 email；否則 401。"""
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise _unauthorized("請先登入")
    try:
        email = verify_access(token.strip(), settings)
    except jwt.ExpiredSignatureError:
        raise _unauthorized("登入已過期") from None
    except jwt.InvalidTokenError:
        raise _unauthorized("請先登入") from None
    if email not in settings.allowed_emails:
        raise _unauthorized("這個帳號沒有權限")
    return email


# ---------------------------------------------------------------------------
# refresh token


def _store(settings: WebSettings) -> RefreshStore:
    return RefreshStore(settings.sessions_path or home_dir() / "web_sessions.db", settings.session_secret)


def _set_refresh(response: Response, issued: Issued, settings: WebSettings, now: float) -> None:
    response.set_cookie(
        REFRESH_COOKIE,
        issued.token,
        max_age=max(int(issued.expires_at - now), 0),
        httponly=True,
        secure=settings.secure_cookie,
        samesite="strict",
        path=REFRESH_PATH,
    )


def _clear_refresh(response: Response, settings: WebSettings) -> None:
    response.delete_cookie(
        REFRESH_COOKIE, path=REFRESH_PATH, secure=settings.secure_cookie, httponly=True, samesite="strict"
    )


def _tokens(response: Response, issued: Issued, settings: WebSettings, now: float) -> dict:
    _set_refresh(response, issued, settings, now)
    return {
        "email": issued.email,
        "access_token": issue_access(issued.email, settings, now),
        "token_type": "Bearer",
        "expires_in": int(settings.access_minutes * 60),
    }


class Credential(BaseModel):
    credential: str  # Google 回傳的 ID token


router = APIRouter(prefix="/api")


@router.get("/config")
def config(settings: WebSettings = Depends(settings_of)) -> dict:
    """登入頁要用的公開設定。"""
    return {"google_client_id": settings.google_client_id}


@router.post("/auth/google")
def login(
    body: Credential,
    request: Request,
    response: Response,
    settings: WebSettings = Depends(settings_of),
) -> dict:
    try:
        claims = request.app.state.verify(body.credential)
    except ValueError:
        raise HTTPException(401, "Google 登入驗證失敗，請重新登入") from None
    email = str(claims.get("email", "")).lower()
    if not claims.get("email_verified") or email not in settings.allowed_emails:
        raise HTTPException(403, f"{email or '這個帳號'} 沒有權限")
    now = _now()
    issued = _store(settings).create(email, now, settings.refresh_hours * 3600)
    return _tokens(response, issued, settings, now)


@router.post("/auth/refresh")
def refresh(request: Request, response: Response, settings: WebSettings = Depends(settings_of)) -> dict:
    """用 refresh token 換新的 access token，refresh token 也同時換新（輪替）。"""
    token = request.cookies.get(REFRESH_COOKIE)
    if not token:
        raise _unauthorized("請先登入")
    store, now = _store(settings), _now()
    try:
        issued = store.rotate(token, now, settings.refresh_hours * 3600)
    except RefreshError as exc:
        error = _unauthorized(str(exc))
        _clear_on(error, settings)
        raise error from None
    if issued.email not in settings.allowed_emails:
        store.revoke_email(issued.email)
        error = _unauthorized("這個帳號沒有權限")
        _clear_on(error, settings)
        raise error
    return _tokens(response, issued, settings, now)


def _clear_on(error: HTTPException, settings: WebSettings) -> None:
    """換發失敗時，順便叫瀏覽器刪掉不能用的 cookie。"""
    response = Response()
    _clear_refresh(response, settings)
    error.headers = {**(error.headers or {}), "set-cookie": response.headers["set-cookie"]}


@router.post("/auth/logout")
def logout(request: Request, response: Response, settings: WebSettings = Depends(settings_of)) -> dict:
    """撤銷 refresh token；access token 在前端丟掉，最多再有效 access_minutes 分鐘。"""
    token = request.cookies.get(REFRESH_COOKIE)
    if token:
        _store(settings).revoke(token)
    _clear_refresh(response, settings)
    return {}


@router.get("/me")
def me(email: str = Depends(current_user)) -> dict:
    return {"email": email}

