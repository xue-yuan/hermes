"""FastAPI 應用：登入、資料 API，以及前端 build 出來的檔案（web/dist）。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from hermes.api import auth, routes
from hermes.api.auth import Verifier, google_verifier
from hermes.api.settings import WebSettings
from hermes.data import DataError
from hermes.services import IndicatorSetNotFoundError, StrategyNotFoundError
from hermes.strategies import StrategyConfigError

# Google 登入按鈕要載入 accounts.google.com 的程式和 iframe
CSP = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' https://accounts.google.com/gsi/client",
        "frame-src https://accounts.google.com/gsi/",
        "connect-src 'self' https://accounts.google.com/gsi/",
        "style-src 'self' 'unsafe-inline' https://accounts.google.com/gsi/style",
        "img-src 'self' data:",
        "font-src 'self'",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self'",
    ]
)


def _error(status: int):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=status)

    return handler


# 網址參數格式不對時，網頁顯示的名稱和說明（FastAPI 預設是英文，而且用程式的參數名稱）
PARAM_LABELS = {
    "stock_id": "股票代號",
    "strategy": "策略",
    "start": "起始日",
    "end": "結束日",
    "date": "日期",
    "set": "指標組",
    "execution": "成交方式",
    "fee_discount": "手續費折扣",
    "slippage": "單邊滑價",
    "tax": "證交稅",
    "sort": "排序依據",
    "window": "N 日",
    "days": "明細天數",
    "adjusted": "還原模式",
    "credential": "登入資料",
}


def _invalid_param(err: dict) -> str:
    name = str(err["loc"][-1]) if err.get("loc") else ""
    label = PARAM_LABELS.get(name, name)
    ctx = err.get("ctx") or {}
    kind = err["type"]
    if kind == "missing":
        return f"沒有指定{label}"
    if kind in ("greater_than", "greater_than_equal", "less_than", "less_than_equal"):
        op = {"greater_than": "大於", "greater_than_equal": "大於或等於", "less_than": "小於",
              "less_than_equal": "小於或等於"}[kind]
        bound = next(iter(ctx.values()), "")
        return f"{label}要{op} {bound:g}" if isinstance(bound, (int, float)) else f"{label}超出範圍"
    if kind in ("float_parsing", "float_type", "int_parsing", "int_type"):
        return f"{label}要是{'整數' if kind.startswith('int') else '數字'}"
    if kind.startswith("date") or kind.startswith("datetime"):
        return f"{label}要是正確的日期（YYYY-MM-DD）"
    return f"{label}的格式不對"


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    messages = dict.fromkeys(_invalid_param(e) for e in exc.errors())
    return JSONResponse({"detail": "；".join(messages)}, status_code=422)


def create_app(settings: WebSettings | None = None, verify: Verifier | None = None) -> FastAPI:
    settings = settings or WebSettings.from_env()
    # 不公開 API 文件：網站在網路上，只給登入的人用
    app = FastAPI(title="hermes", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.verify = verify or google_verifier(settings.google_client_id)
    app.include_router(auth.router)
    app.include_router(routes.router)
    # 使用者看得懂的錯誤：缺資料、找不到策略、設定檔有誤
    app.add_exception_handler(DataError, _error(404))
    app.add_exception_handler(StrategyNotFoundError, _error(404))
    app.add_exception_handler(IndicatorSetNotFoundError, _error(404))
    app.add_exception_handler(StrategyConfigError, _error(422))
    app.add_exception_handler(RequestValidationError, _validation_error)

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = CSP
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"  # 個人資料，不讓瀏覽器或代理伺服器快取
        return response

    @app.api_route("/api/{rest:path}", methods=["GET", "POST"], include_in_schema=False)
    def unknown_api(rest: str) -> JSONResponse:
        return JSONResponse({"detail": "沒有這個 API"}, status_code=404)

    if settings.static_dir is not None:
        _serve_frontend(app, settings.static_dir)
    return app


def _serve_frontend(app: FastAPI, dist: Path) -> None:
    """前端是單頁應用：/assets 是 build 出來、檔名含雜湊的檔案；其他路徑都回 index.html。"""
    dist = dist.resolve()
    app.mount("/assets", _ImmutableFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        file = (dist / path).resolve()
        if path and file.is_file() and file.is_relative_to(dist):
            return FileResponse(file)
        return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})


class _ImmutableFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response
