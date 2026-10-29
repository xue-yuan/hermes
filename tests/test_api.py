"""網頁 API：登入、只讀、回傳的數字和 services 算出來的完全相同。"""

import ast
import dataclasses
import hashlib
import math
import threading
from pathlib import Path

import jwt
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import make_prices
from hermes.api import SettingsError, WebSettings, create_app
from hermes.api import auth as auth_module
from hermes.api import routes
from hermes.api.sessions import RefreshError, RefreshStore
from hermes.backtest import CostModel
from hermes.data import Cache, DataLoader, Watchlist
from hermes.services import (
    chart_layout,
    chip_report,
    compare_strategies,
    indicator_snapshot,
    rule_backtest,
    screen,
    signal_history,
)
from hermes.strategies import parse_strategy
from test_rule_backtest import _write_stock

EMAIL = "viewer@example.com"
CROSS = """
name: CCI 回升
indicators:
  cci:  {type: cci, period: 20}
  ma20: {type: sma, period: 20}
buy:
  all:
    - {left: cci, op: crosses_above, right: -100}
    - {left: close, op: gt, right: ma20}
sell: {left: cci, op: crosses_below, right: 100}
forming:
  all:
    - {left: cci, op: lt, right: 0}
    - {left: close, op: lt, right: {ref: ma20, ago: 1}}
"""
FOREIGN = """
name: 外資連買
buy:  {left: {ref: foreign_net, ago: 1}, op: gt, right: 0, days: 2}
sell: {left: close, op: lt, right: 90}
"""
SET = """
name: 測試指標組
indicators:
  ma5: {type: sma, period: 5}
  kd:  {type: kd, period: 9, k_smooth: 3, d_smooth: 3}
  obv: {type: obv}
"""


def _write_chips(home, stock_id, days, seed=1):
    rng = np.random.default_rng(seed)
    cache = Cache(home)
    ds = [f"{d:%Y-%m-%d}" for d in days]
    rows = []
    for d in ds:
        rows += [{"date": d, "stock_id": stock_id, "name": n, "buy": int(rng.integers(0, 5e6)),
                  "sell": int(rng.integers(0, 5e6))}
                 for n in ("Foreign_Investor", "Foreign_Dealer_Self", "Investment_Trust", "Dealer_self",
                           "Dealer_Hedging")]
    cache.write("TaiwanStockInstitutionalInvestorsBuySell", stock_id, pd.DataFrame(rows))
    cache.write("TaiwanStockMarginPurchaseShortSale", stock_id, pd.DataFrame([
        {"date": d, "stock_id": stock_id, "MarginPurchaseTodayBalance": 30000 + i, "ShortSaleTodayBalance": 15 + i,
         "MarginPurchaseLimit": 1_000_000} for i, d in enumerate(ds)]))
    cache.write("TaiwanStockShareholding", stock_id, pd.DataFrame([
        {"date": d, "stock_id": stock_id, "ForeignInvestmentSharesRatio": 68 + i / 100, "NumberOfSharesIssued": 1e9}
        for i, d in enumerate(ds)]))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("web")
    home, sdir, idir = root / "home", root / "strategies", root / "indicators"
    base = make_prices(n=320, seed=7)
    days = pd.bdate_range("2024-01-02", periods=len(base))
    bars = [tuple(base[c].iloc[i] for c in ("open", "high", "low", "close")) for i in range(len(base))]
    closes = [b[3] for b in bars]
    k = 200  # 除息日：參考價比前一天收盤低 3%
    _write_stock(home, "1234", bars, days, dividends=[(days[k], closes[k - 1], closes[k - 1] * 0.97)])
    # 除權息：股數改變，本機沒有證交所的股數比例，除權息日以前的還原成交量算不出來
    Cache(home).write("TaiwanStockDividendResult", "1234", pd.DataFrame([
        {"date": f"{days[k]:%Y-%m-%d}", "stock_id": "1234", "before_price": closes[k - 1],
         "after_price": closes[k - 1] * 0.97, "stock_or_cache_dividend": "權息"}]))
    _write_chips(home, "1234", days)
    _write_stock(home, "5678", bars[:150], days[:150])
    sdir.mkdir()
    (sdir / "cross.yaml").write_text(CROSS, encoding="utf-8")
    (sdir / "foreign.yaml").write_text(FOREIGN, encoding="utf-8")
    (sdir / "broken.yaml").write_text("name: x\nbuy: {left: close, op: bigger, right: 1}\n", encoding="utf-8")
    idir.mkdir()
    (idir / "common.yaml").write_text(SET, encoding="utf-8")
    with Watchlist(home / "hermes.db") as wl:
        wl.add("1234", "測試")
        wl.add("5678", "測試")
    return home, sdir, idir


@pytest.fixture
def app(env, monkeypatch, tmp_path):
    home, sdir, idir = env
    monkeypatch.setenv("HERMES_HOME", str(home))
    settings = WebSettings("client-id.apps.googleusercontent.com", frozenset({EMAIL}), "s" * 40,
                           sessions_path=tmp_path / "sessions.db")
    claims = {"good": {"email": EMAIL, "email_verified": True},
              "upper": {"email": EMAIL.upper(), "email_verified": True},
              "other": {"email": "someone@example.com", "email_verified": True},
              "unverified": {"email": EMAIL, "email_verified": False}}

    def verify(credential):
        if credential not in claims:
            raise ValueError("bad token")
        return claims[credential]

    app = create_app(settings, verify)
    app.dependency_overrides[routes.get_loader] = lambda: DataLoader(Cache(home))
    app.dependency_overrides[routes.get_strategies_dir] = lambda: sdir
    app.dependency_overrides[routes.get_indicators_dir] = lambda: idir
    return app


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client(app):
    c = TestClient(app, base_url="https://testserver")
    r = c.post("/api/auth/google", json={"credential": "good"})
    assert r.status_code == 200
    c.headers.update(_bearer(r.json()["access_token"]))
    return c


@pytest.fixture
def loader(env):
    return DataLoader(Cache(env[0]))


# ---------------------------------------------------------------------------
# 登入：access token（JWT）加 refresh token


def test_login_flow(app):
    c = TestClient(app, base_url="https://testserver")
    assert c.get("/api/config").json() == {"google_client_id": "client-id.apps.googleusercontent.com"}
    r = c.get("/api/overview")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert c.post("/api/auth/google", json={"credential": "nope"}).status_code == 401
    assert c.post("/api/auth/google", json={"credential": "other"}).status_code == 403
    assert c.post("/api/auth/google", json={"credential": "unverified"}).status_code == 403
    assert c.post("/api/auth/refresh").status_code == 401  # 還沒登入，沒有 refresh token

    r = c.post("/api/auth/google", json={"credential": "upper"})
    body = r.json()
    assert r.status_code == 200 and body["email"] == EMAIL
    assert body["token_type"] == "Bearer" and body["expires_in"] == 900
    cookie = r.headers["set-cookie"].lower()
    for flag in ("hermes_refresh=", "httponly", "secure", "samesite=strict", "path=/api/auth", "max-age=604800"):
        assert flag in cookie
    assert c.get("/api/me", headers=_bearer(body["access_token"])).json() == {"email": EMAIL}
    # refresh token 只能拿來換發，不能直接呼叫 API
    assert c.get("/api/me").status_code == 401
    refresh = c.cookies[auth_module.REFRESH_COOKIE]
    assert c.get("/api/me", headers={"Cookie": f"hermes_refresh={refresh}"}).status_code == 401
    assert c.get("/api/me", headers=_bearer(refresh)).status_code == 401

    # access token 的內容：HS256、typ at+jwt、發給 hermes-web、15 分鐘
    header = jwt.get_unverified_header(body["access_token"])
    claims = jwt.decode(body["access_token"], options={"verify_signature": False})
    assert header["alg"] == "HS256" and header["typ"] == "at+jwt"
    assert claims["iss"] == "hermes" and claims["aud"] == "hermes-web" and claims["sub"] == EMAIL
    assert claims["exp"] - claims["iat"] == 900


def test_login_needs_json_body(app):
    c = TestClient(app, base_url="https://testserver")
    assert c.post("/api/auth/google", data={"credential": "good"}).status_code == 422  # 表單送不進來


def test_refresh_rotates_and_detects_reuse(app, monkeypatch):
    c = TestClient(app, base_url="https://testserver")
    c.post("/api/auth/google", json={"credential": "good"})
    start = auth_module._now()
    old = c.cookies[auth_module.REFRESH_COOKIE]
    r = c.post("/api/auth/refresh")
    assert r.status_code == 200
    new = c.cookies[auth_module.REFRESH_COOKIE]
    assert new != old
    assert c.get("/api/me", headers=_bearer(r.json()["access_token"])).status_code == 200
    assert c.post("/api/auth/refresh").status_code == 200  # 新的可以繼續換
    newest = c.cookies[auth_module.REFRESH_COOKIE]

    # 換過 30 秒後，舊的又被拿來用：可能被偷了，整串撤銷，連最新的也不能用（RFC 9700 §4.14.2）
    monkeypatch.setattr(auth_module, "_now", lambda: start + 31)
    other = TestClient(app, base_url="https://testserver")
    other.cookies.set(auth_module.REFRESH_COOKIE, old, path="/api/auth")
    r = other.post("/api/auth/refresh")
    assert r.status_code == 401 and "重複使用" in r.json()["detail"]
    assert "hermes_refresh=" in r.headers["set-cookie"] and "max-age=0" in r.headers["set-cookie"].lower()
    c.cookies.set(auth_module.REFRESH_COOKIE, newest, path="/api/auth")
    assert c.post("/api/auth/refresh").status_code == 401


def test_refresh_expires_when_idle(app, monkeypatch):
    c = TestClient(app, base_url="https://testserver")
    c.post("/api/auth/google", json={"credential": "good"})
    start = auth_module._now()
    # 6 天後用一次：期限從那時重新算
    monkeypatch.setattr(auth_module, "_now", lambda: start + 6 * 24 * 3600)
    assert c.post("/api/auth/refresh").status_code == 200
    monkeypatch.setattr(auth_module, "_now", lambda: start + 12 * 24 * 3600)
    assert c.post("/api/auth/refresh").status_code == 200
    # 之後 7 天都沒用：過期
    monkeypatch.setattr(auth_module, "_now", lambda: start + 19 * 24 * 3600 + 5)
    r = c.post("/api/auth/refresh")
    assert r.status_code == 401 and "太久沒有使用" in r.json()["detail"]


def test_logout_revokes_refresh_token(app):
    c = TestClient(app, base_url="https://testserver")
    access = c.post("/api/auth/google", json={"credential": "good"}).json()["access_token"]
    refresh = c.cookies[auth_module.REFRESH_COOKIE]
    r = c.post("/api/auth/logout")
    assert r.status_code == 200 and "max-age=0" in r.headers["set-cookie"].lower()
    c.cookies.set(auth_module.REFRESH_COOKIE, refresh, path="/api/auth")
    assert c.post("/api/auth/refresh").status_code == 401
    # access token 不存在伺服器上，前端登出時丟掉；被留下來的話最多再有效 15 分鐘
    assert c.get("/api/me", headers=_bearer(access)).status_code == 200


@pytest.mark.filterwarnings("ignore::jwt.warnings.InsecureKeyLengthWarning")  # 故意用 HS512 簽的假 token
def test_bad_access_tokens(app):
    c = TestClient(app, base_url="https://testserver")
    settings = app.state.settings
    now = auth_module._now()
    good = auth_module.issue_access(EMAIL, settings, now)
    assert c.get("/api/me", headers=_bearer(good)).status_code == 200

    expired = auth_module.issue_access(EMAIL, settings, now - 16 * 60)
    r = c.get("/api/me", headers=_bearer(expired))
    assert r.status_code == 401 and r.json()["detail"] == "登入已過期"

    head, payload, sig = good.split(".")
    tampered = f"{head}.{payload}.{sig[:-2]}{'AA' if not sig.endswith('AA') else 'BB'}"
    claims = jwt.decode(good, options={"verify_signature": False})
    bad = {
        "簽章被改": tampered,
        "別的密鑰": jwt.encode(claims, "x" * 40, algorithm="HS256", headers={"typ": "at+jwt"}),
        "alg none": jwt.encode(claims, None, algorithm="none", headers={"typ": "at+jwt"}),
        "別的演算法": jwt.encode(claims, settings.session_secret, algorithm="HS512", headers={"typ": "at+jwt"}),
        "沒有 typ": jwt.encode(claims, settings.session_secret, algorithm="HS256"),
        "發給別人": jwt.encode({**claims, "aud": "other"}, settings.session_secret, algorithm="HS256",
                           headers={"typ": "at+jwt"}),
        "別人發的": jwt.encode({**claims, "iss": "other"}, settings.session_secret, algorithm="HS256",
                           headers={"typ": "at+jwt"}),
        "沒有期限": jwt.encode({k: v for k, v in claims.items() if k != "exp"}, settings.session_secret,
                           algorithm="HS256", headers={"typ": "at+jwt"}),
        "不是 JWT": "abc",
    }
    for label, token in bad.items():
        assert c.get("/api/me", headers=_bearer(token)).status_code == 401, label
    assert c.get("/api/me", headers={"Authorization": good}).status_code == 401  # 少了 Bearer


def test_allowlist_and_secret_changes(app):
    c = TestClient(app, base_url="https://testserver")
    access = c.post("/api/auth/google", json={"credential": "good"}).json()["access_token"]
    settings = app.state.settings

    # 從允許名單拿掉：access token 立刻失效，refresh token 撤銷，加回名單也不能用
    app.state.settings = dataclasses.replace(settings, allowed_emails=frozenset({"other@example.com"}))
    assert c.get("/api/me", headers=_bearer(access)).status_code == 401
    assert c.post("/api/auth/refresh").status_code == 401
    app.state.settings = settings
    assert c.post("/api/auth/refresh").status_code == 401

    # 換了 session_secret：access token、refresh token 都失效
    access = c.post("/api/auth/google", json={"credential": "good"}).json()["access_token"]
    app.state.settings = dataclasses.replace(settings, session_secret="t" * 40)
    assert c.get("/api/me", headers=_bearer(access)).status_code == 401
    assert c.post("/api/auth/refresh").status_code == 401


def test_refresh_store_is_hashed(tmp_path):
    store = RefreshStore(tmp_path / "s.db", "k" * 40)
    issued = store.create(EMAIL, 1000.0, 3600)
    assert issued.token.encode() not in (tmp_path / "s.db").read_bytes()  # 只存 HMAC
    with pytest.raises(RefreshError):  # 換了密鑰就找不到
        RefreshStore(tmp_path / "s.db", "j" * 40).rotate(issued.token, 1001.0, 3600)


def test_rotation_grace_period(tmp_path):
    """換發的回應沒送到瀏覽器（例如換發途中重新整理）：30 秒內再用舊的不算被偷（Okta 的寬限期）。"""
    store = RefreshStore(tmp_path / "s.db", "k" * 40)
    first = store.create(EMAIL, 1000.0, 3600).token
    lost = store.rotate(first, 1010.0, 3600).token  # 這個回應沒送到
    again = store.rotate(first, 1025.0, 3600).token  # 15 秒後瀏覽器又用舊的：可以
    assert again != lost
    assert store.rotate(again, 1030.0, 3600)  # 新的照常可以用
    # 超過寬限期再用舊的：整串撤銷，寬限期內換到的也不能用
    with pytest.raises(RefreshError, match="重複使用"):
        store.rotate(first, 1041.0, 3600)
    with pytest.raises(RefreshError):
        store.rotate(lost, 1042.0, 3600)


def test_concurrent_rotation(tmp_path):
    """多個請求同時拿同一個 token 來換（例如兩個分頁）：都在寬限期內，全部成功，各拿到不同的新 token。"""
    store = RefreshStore(tmp_path / "s.db", "k" * 40)
    issued = store.create(EMAIL, 1000.0, 3600)
    results = []

    def worker():
        try:
            results.append(store.rotate(issued.token, 1001.0, 3600).token)
        except RefreshError as exc:
            results.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(isinstance(r, str) for r in results) and len(set(results)) == 8


def test_old_session_db_gets_grace_column(tmp_path):
    import sqlite3

    path = tmp_path / "s.db"
    with sqlite3.connect(path) as conn:  # 加上寬限期以前的格式
        conn.executescript("CREATE TABLE refresh_tokens (token_hash TEXT PRIMARY KEY, family TEXT NOT NULL, "
                           "email TEXT NOT NULL, issued_at REAL NOT NULL, expires_at REAL NOT NULL, "
                           "replaced INTEGER NOT NULL DEFAULT 0, revoked INTEGER NOT NULL DEFAULT 0);")
    conn.close()
    store = RefreshStore(path, "k" * 40)
    token = store.create(EMAIL, 1000.0, 3600).token
    assert store.rotate(token, 1001.0, 3600)


def test_settings_from_env(monkeypatch):
    for k in ("HERMES_GOOGLE_CLIENT_ID", "HERMES_ALLOWED_EMAILS", "HERMES_SESSION_SECRET", "HERMES_ACCESS_MINUTES",
              "HERMES_REFRESH_HOURS"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SettingsError, match="HERMES_GOOGLE_CLIENT_ID、HERMES_ALLOWED_EMAILS、HERMES_SESSION_SECRET"):
        WebSettings.from_env()
    monkeypatch.setenv("HERMES_GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("HERMES_ALLOWED_EMAILS", " A@x.com, b@y.com ,")
    monkeypatch.setenv("HERMES_SESSION_SECRET", "short")
    with pytest.raises(SettingsError, match="至少要 32"):
        WebSettings.from_env()
    monkeypatch.setenv("HERMES_SESSION_SECRET", "z" * 32)
    s = WebSettings.from_env()
    assert s.allowed_emails == {"a@x.com", "b@y.com"} and s.secure_cookie
    assert s.access_minutes == 15 and s.refresh_hours == 168
    monkeypatch.setenv("HERMES_ACCESS_MINUTES", "0")
    with pytest.raises(SettingsError, match="HERMES_ACCESS_MINUTES 要大於 0"):
        WebSettings.from_env()


def test_settings_from_dotenv(tmp_path, monkeypatch):
    from hermes.api.server import load_settings

    for k in ("HERMES_GOOGLE_CLIENT_ID", "HERMES_ALLOWED_EMAILS", "HERMES_SESSION_SECRET", "HERMES_INSECURE_COOKIE",
              "HERMES_ACCESS_MINUTES", "HERMES_REFRESH_HOURS"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / ".env").write_text(
        "# 註解\nHERMES_GOOGLE_CLIENT_ID=from-dotenv\nHERMES_ALLOWED_EMAILS=a@x.com\n"
        f"HERMES_SESSION_SECRET={'q' * 40}\nHERMES_INSECURE_COOKIE=1\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_ALLOWED_EMAILS", "b@y.com")  # 已經設定的環境變數優先
    s = load_settings(tmp_path)
    assert s.google_client_id == "from-dotenv" and s.allowed_emails == {"b@y.com"} and not s.secure_cookie

    (tmp_path / ".env").write_text("HERMES_GOOGLE_CLIENT_ID=\n", encoding="utf-8")
    for k in ("HERMES_GOOGLE_CLIENT_ID", "HERMES_SESSION_SECRET"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SettingsError, match="請在專案根目錄的 .env 填上：HERMES_GOOGLE_CLIENT_ID、HERMES_SESSION_SECRET"):
        load_settings(tmp_path)


# ---------------------------------------------------------------------------
# 只讀：網頁不能下載、不能寫入

API_DIR = Path(routes.__file__).parent
FORBIDDEN = {"MarketDataService", "MarketDataSync", "FinMindClient", "sync", "twse", "verify", "Watchlist"}


def test_api_only_reads():
    for router in (routes.router, auth_module.router):
        for route in router.routes:
            auth = ("/api/auth/google", "/api/auth/refresh", "/api/auth/logout")
            allowed = {"POST"} if route.path in auth else {"GET", "HEAD"}
            assert set(route.methods) <= allowed, route.path
    for path in API_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {a.name.split(".")[-1] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                 for a in n.names}
        modules = {p for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
                   for p in n.module.split(".")}
        assert not (names | modules) & FORBIDDEN, path.name


def _snapshot(home: Path) -> dict:
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(home.rglob("*")) if p.is_file()}


def test_endpoints_do_not_change_local_data(client, env):
    home = env[0]
    before = _snapshot(home)
    for url in ("/api/overview", "/api/screen", "/api/stocks/1234", "/api/stocks/1234/prices",
                "/api/stocks/1234/signals?strategy=cross",
                "/api/stocks/1234/tech", "/api/stocks/1234/chips?window=5",
                "/api/stocks/1234/backtest?strategy=cross",
                "/api/stocks/1234/compare?strategy=cross&strategy=foreign",
                "/api/screen?adjusted=true", "/api/stocks/1234/prices?adjusted=true",
                "/api/stocks/1234/tech?adjusted=true",
                "/api/stocks/1234/signals?strategy=cross&adjusted=true",
                "/api/stocks/1234/backtest?strategy=cross&adjusted=true",
                "/api/stocks/1234/compare?strategy=cross&strategy=foreign&adjusted=true"):
        assert client.get(url).status_code == 200, url
    assert _snapshot(home) == before


# ---------------------------------------------------------------------------
# 數字和 services 相同


def _json(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, float) and math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    return v


def _same(api_values, series):
    """API 的 null 對應空值，其他數字完全相同（不是近似）。"""
    assert list(api_values) == [_json(v) for v in series]


def test_signals_match_services(client, env, loader):
    _, sdir, _ = env
    body = client.get("/api/stocks/1234/signals?strategy=cross").json()
    hist = signal_history("1234", "cross", loader=loader, directory=sdir)
    full = hist.table(every_day=True)
    assert body["series"]["date"] == [f"{d:%Y-%m-%d}" for d in full.index]
    for col in ["open", "close", "cci", "ma20", "buy_streak", "sell_streak"]:
        _same(body["series"][col], full[col].tolist())
    # 只有一種價格：不附另一種價格的判斷
    assert not any(k.startswith(("other_", "adj_")) or k == "differs" for k in body["series"])
    _same(body["series"]["volume_lots"], (full["volume"] / 1000).tolist())

    last = full.index[-1]
    period = hist.table((last - pd.DateOffset(years=1)).date(), last.date(), every_day=True)
    shown = period[period["buy"] | period["sell"]]
    assert [r["date"] for r in body["rows"]] == [f"{d:%Y-%m-%d}" for d in shown.index[::-1]]
    assert body["summary"] == {
        "buy_signals": int((period["buy_streak"] == 1).sum()), "buy_days": int(period["buy"].sum()),
        "sell_signals": int((period["sell_streak"] == 1).sum()), "sell_days": int(period["sell"].sum())}
    layout = chart_layout(hist.strategy)
    assert body["layout"] == {"overlays": layout.overlays, "panels": layout.panels,
                              "thresholds": layout.thresholds, "reference_lines": {}}
    assert body["layout"]["thresholds"] == {"cci": [-100.0, 100.0]} and body["layout"]["overlays"] == ["ma20"]
    # 網頁上的條件和欄位都用白話名稱，不顯示設定檔裡的名稱
    assert body["strategy"]["buy"] == "CCI 順勢指標（20 日） 由下往上穿越 -100 且 收盤價 > 簡單移動平均（20 日）"
    assert body["column_labels"] == {"cci": "CCI 順勢指標（20 日）", "ma20": "簡單移動平均（20 日）",
                                     "ma20（前1天）": "前 1 天的 簡單移動平均（20 日）"}


def test_reference_lines_in_layout(client, env):
    """設定檔寫的參考線：子圖指標放在子圖名稱下，畫在價格圖上的指標放在 price。"""
    _, sdir, _ = env
    (sdir / "ref.yaml").write_text(CROSS + "reference_lines:\n  cci: [100, -100, 0]\n  ma20: [95]\n",
                                   encoding="utf-8")
    try:
        body = client.get("/api/stocks/1234/signals?strategy=ref").json()
    finally:
        (sdir / "ref.yaml").unlink()
    assert body["layout"]["reference_lines"] == {"cci": [-100.0, 0.0, 100.0], "price": [95.0]}
    assert body["layout"]["thresholds"] == {"cci": [-100.0, 100.0]}


def test_chart_layout_draws_only_used_lines():
    """多條線的指標只畫條件裡用到的線；用到的線都要畫。"""
    s = parse_strategy({"name": "t", "indicators": {"dc": {"type": "donchian", "period": 20},
                                                   "kd": {"type": "kd", "period": 9, "k_smooth": 3, "d_smooth": 3}},
                        "buy": {"left": "close", "op": "crosses_above", "right": {"ref": "dc.upper", "ago": 1}},
                        "sell": {"left": "kd.k", "op": "crosses_below", "right": "kd.d"}})
    layout = chart_layout(s)
    assert layout.overlays == ["dc.upper"]
    assert layout.panels == {"kd": ["kd.k", "kd.d"]}
    # 形成中的條件不影響買賣，圖上不多畫它用到的線、籌碼和門檻
    forming = {"all": [{"left": "close", "op": "gt", "right": "dc.lower"},
                       {"left": "foreign_net", "op": "gt", "right": 0},
                       {"left": "close", "op": "gt", "right": 50}]}
    assert chart_layout(parse_strategy({**s.model_dump(exclude_none=True), "forming": forming})) == layout


def test_threshold_follows_the_line_not_the_indicator():
    """同一個指標的線分在價格圖和子圖（布林通道的下緣在價格圖、%b 在子圖）：門檻畫在左邊那條線所在的圖。"""
    s = parse_strategy({"name": "t", "indicators": {"bb": {"type": "bollinger", "period": 20, "k": 2}},
                        "buy": {"all": [{"left": "close", "op": "lt", "right": "bb.lower"},
                                        {"left": "bb.pct_b", "op": "lt", "right": 0.05}]},
                        "sell": {"left": "bb.lower", "op": "gt", "right": 120}})
    layout = chart_layout(s)
    assert layout.overlays == ["bb.lower"]
    assert layout.panels == {"bb": ["bb.pct_b"]}
    assert layout.thresholds == {"bb": [0.05], "price": [120.0]}


def test_volume_threshold_is_in_lots(client, env):
    _, sdir, _ = env
    (sdir / "vol.yaml").write_text("name: 爆量\nbuy: {left: volume, op: gt, right: 3000000}\n"
                                   "sell: {left: close, op: lt, right: 1}\n", encoding="utf-8")
    try:
        body = client.get("/api/stocks/1234/signals?strategy=vol").json()
    finally:
        (sdir / "vol.yaml").unlink()
    assert body["layout"]["thresholds"] == {"volume": [3000.0], "price": [1.0]}  # 300 萬股 = 3000 張


def test_signals_range(client):
    body = client.get("/api/stocks/1234/signals?strategy=cross&start=2024-03-01&end=2024-06-28").json()
    assert body["range"] == {"start": "2024-03-01", "end": "2024-06-28"}
    assert client.get("/api/stocks/1234/signals?strategy=cross&start=2025-01-01&end=2024-01-01").status_code == 422
    # 只給結束日：結束日往前一年（資料從 2024-01-02 開始），不是用最新一天往前一年當起始日而報錯
    body = client.get("/api/stocks/1234/signals?strategy=cross&end=2024-02-29").json()
    assert body["range"] == {"start": "2024-01-02", "end": "2024-02-29"}


def test_prices_match_loader_and_signals(client, loader):
    """沒選策略也畫得出股價圖：日 K 和本機資料相同，日期、價格、區間也和買賣點的資料相同，圖才對得齊。"""
    body = client.get("/api/stocks/1234/prices").json()
    raw = loader.prices("1234", adjust=False)
    assert body["series"]["date"] == [f"{d:%Y-%m-%d}" for d in raw.index]
    for col in ("open", "high", "low", "close"):
        _same(body["series"][col], raw[col].tolist())
    _same(body["series"]["volume_lots"], (raw["volume"] / 1000).tolist())

    for q in ("", "&start=2024-03-01&end=2024-06-28", "&start=2024-03-02", "&end=2024-02-29"):
        sig = client.get(f"/api/stocks/1234/signals?strategy=cross{q}").json()
        px = client.get(f"/api/stocks/1234/prices?{q[1:]}").json()
        assert px["range"] == sig["range"], q
        for col in ("date", "open", "high", "low", "close", "volume_lots"):
            assert px["series"][col] == sig["series"][col], (q, col)
    assert client.get("/api/stocks/1234/prices?start=2025-01-01&end=2024-01-01").status_code == 422
    assert client.get("/api/stocks/1234/prices?start=2030-01-01").status_code == 422


def test_backtest_matches_services(client, env, loader):
    _, sdir, _ = env
    url = "/api/stocks/1234/backtest?strategy=cross&fee_discount=0.5&slippage=0.001"
    body = client.get(url).json()
    costs = CostModel(discount=0.5, slippage=0.001)
    rb = rule_backtest(signal_history("1234", "cross", loader=loader, directory=sdir), None, None, costs,
                       "open", loader)
    m = body["metrics"]
    assert [x["id"] for x in m] == ["benchmark", "strategy"]
    assert m[1]["metrics"] == {k: _json(v) for k, v in rb.result.metrics.items()}
    assert m[0]["metrics"]["total_return"] == rb.result.benchmark_metrics["total_return"]
    trades = rb.trades()
    assert len(body["trades"]) == len(trades)
    assert [t["entry_price"] for t in body["trades"]] == trades["entry_price"].tolist()[::-1]
    _same(body["curves"]["series"][0]["equity"], rb.result.equity.tolist())
    last, holding, action = rb.latest()
    assert body["latest"] == {"date": f"{last:%Y-%m-%d}", "holding": holding, "action": action}
    assert body["start"] == f"{rb.start:%Y-%m-%d}" and body["costs"]["discount"] == 0.5


def test_compare_matches_services(client, env, loader):
    _, sdir, _ = env
    body = client.get("/api/stocks/1234/compare?strategy=cross&strategy=foreign&sort=cagr").json()
    cmp = compare_strategies("1234", ["cross", "foreign"], sort="cagr", loader=loader, directory=sdir)
    assert [r["id"] for r in body["rows"]] == cmp.ids
    assert [r["metrics"]["cagr"] for r in body["rows"]] == [r.metrics["cagr"] for r in cmp.results]
    assert body["benchmark"]["metrics"]["cagr"] == cmp.benchmark.metrics["cagr"]
    assert [s["id"] for s in body["curves"]["series"]] == cmp.ids


def test_adjusted_mode_matches_services(client, env, loader):
    """還原模式：每個端點用還原價格判斷，數字和 services 的還原模式完全相同。"""
    _, sdir, _ = env
    body = client.get("/api/stocks/1234/signals?strategy=cross&adjusted=true").json()
    hist = signal_history("1234", "cross", adjust=True, loader=loader, directory=sdir)
    full = hist.table(every_day=True)
    assert body["adjusted"] is True
    for col in ["open", "close", "cci", "ma20", "buy_streak", "sell_streak"]:
        _same(body["series"][col], full[col].tolist())
    # K 線是還原價格，和沒選策略時的股價圖相同
    adj = loader.prices("1234", adjust=True)
    _same(body["series"]["close"], adj["close"].tolist())
    assert body["series"]["close"] != client.get("/api/stocks/1234/signals?strategy=cross").json()["series"]["close"]
    px = client.get("/api/stocks/1234/prices?adjusted=true").json()
    assert px["adjusted"] is True
    for col in ("date", "open", "high", "low", "close", "volume_lots"):
        assert px["series"][col] == body["series"][col], col
    assert client.get("/api/stocks/1234/signals?strategy=cross").json()["adjusted"] is False
    # 最近一次訊號也跟著模式：200 日均線跨過除息日，兩種價格的訊號日不同
    (sdir / "long.yaml").write_text("name: 站上兩百日線\nindicators:\n  ma200: {type: sma, period: 200}\n"
                                    "buy: {left: close, op: crosses_above, right: ma200}\n"
                                    "sell: {left: close, op: crosses_below, right: ma200}\n", encoding="utf-8")
    try:
        latest = {a: client.get(f"/api/stocks/1234/signals?strategy=long&adjusted={a}").json()["latest"]
                  for a in ("false", "true")}
        long = {a: signal_history("1234", "long", adjust=a == "true", loader=loader, directory=sdir)
                for a in ("false", "true")}
    finally:
        (sdir / "long.yaml").unlink()
    for a, h in long.items():
        for side in ("buy", "sell"):
            last = h.last_trigger(side)
            assert latest[a][f"last_{side}"] == (None if last is None else f"{last:%Y-%m-%d}")
    dates = {a: (latest[a]["last_buy"], latest[a]["last_sell"]) for a in latest}
    assert dates["false"] != dates["true"]  # 確認這個對照有意義

    bt = client.get("/api/stocks/1234/backtest?strategy=cross&adjusted=true").json()
    rb = rule_backtest(hist, None, None, CostModel(), "open", loader)
    assert bt["adjusted"] is True
    assert bt["metrics"][1]["metrics"] == {k: _json(v) for k, v in rb.result.metrics.items()}
    assert [t["entry_price"] for t in bt["trades"]] == rb.trades()["entry_price"].tolist()[::-1]

    sc = client.get("/api/screen?adjusted=true").json()
    report = screen(["1234", "5678"], loader=loader, directory=sdir, adjusted=True)
    assert sc["adjusted"] is True
    assert [(t["stock_id"], t["strategy_id"], t["side"]) for t in sc["triggered"]] == [
        (r.stock_id, r.strategy_id, side) for r, side in report.triggered_today()]
    assert [(t["stock_id"], t["strategy_id"], t["days"]) for t in sc["ongoing"]] == [
        (r.stock_id, r.strategy_id, days) for r, _, days in sorted(report.ongoing(), key=lambda x: x[2])]

    co = client.get("/api/stocks/1234/compare?strategy=cross&strategy=foreign&adjusted=true").json()
    cs = compare_strategies("1234", ["cross", "foreign"], loader=loader, directory=sdir, adjusted=True)
    assert co["adjusted"] is True
    assert [r["metrics"]["sharpe"] for r in co["rows"]] == [_json(r.metrics["sharpe"]) for r in cs.results]

    tech = client.get("/api/stocks/1234/tech?date=2024-03-01&adjusted=true").json()
    snap = indicator_snapshot("1234", "common", "2024-03-01", loader, env[2],
                              adjusted=True)
    assert tech["adjusted"] is True and tech["close"] == snap.close
    for r in tech["rows"]:
        _same([r["value"]], [snap.values[r["column"]]])


def test_tech_matches_services(client, env, loader):
    _, _, idir = env
    body = client.get("/api/stocks/1234/tech?date=2024-08-01").json()
    snap = indicator_snapshot("1234", "common", "2024-08-01", loader, idir)
    assert body["date"] == "2024-08-01"
    assert [r["column"] for r in body["rows"]] == list(snap.values.index)
    assert body["adjusted"] is False and body["close"] == snap.close
    for r in body["rows"]:
        _same([r["value"]], [snap.values[r["column"]]])
    obv = next(r for r in body["rows"] if r["column"] == "obv")
    assert not obv["unavailable"] and obv["value"] is not None and body["warnings"] == []
    # 還原模式：除權沒有股數比例，還原價格的 OBV 在除權日以前算不出來
    adj = client.get("/api/stocks/1234/tech?date=2024-08-01&adjusted=true").json()
    obv = next(r for r in adj["rows"] if r["column"] == "obv")
    assert obv["unavailable"] and obv["value"] is None
    assert any("沒有證交所的股數比例" in w for w in adj["warnings"])
    assert client.get("/api/stocks/1234/tech?set=nope").status_code == 404
    # 每一列都有白話名稱，多條線的指標加上線的名稱
    labels = {r["column"]: r["label"] for r in body["rows"]}
    assert labels["ma5"] == "簡單移動平均（5 日）"
    assert labels["kd.k"] == "KD 隨機指標（RSV 9 日、K 平滑 3、D 平滑 3）：K 值"
    kd = next(r for r in body["rows"] if r["column"] == "kd.d")
    assert (kd["indicator"], kd["line"]) == ("KD 隨機指標（RSV 9 日、K 平滑 3、D 平滑 3）", "D 值")
    assert next(r for r in body["rows"] if r["column"] == "ma5")["line"] is None


def test_chips_matches_services(client, loader):
    body = client.get("/api/stocks/1234/chips?window=5&window=20&days=3").json()
    rep = chip_report("1234", [5, 20], 3, loader)
    assert body["flow_columns"] == ["5 日", "20 日"]
    assert [f["values"] for f in body["flows"]] == rep.flows.to_numpy().tolist()
    assert [r["date"] for r in body["recent"]] == [f"{d:%Y-%m-%d}" for d in rep.recent.index[::-1]]
    assert body["foreign_ratio"]["changes"] == {"5": rep.foreign_ratio["chg_5"], "20": rep.foreign_ratio["chg_20"]}
    assert body["margin"]["margin_balance"] == rep.margin["margin_balance"]
    assert client.get("/api/stocks/1234/chips?window=0").status_code == 422


def test_screen_matches_services(client, env, loader):
    _, sdir, _ = env
    body = client.get("/api/screen").json()
    report = screen(["1234", "5678"], loader=loader, directory=sdir)
    assert body["latest_date"] == f"{report.latest_date:%Y-%m-%d}"
    assert [(t["stock_id"], t["strategy_id"], t["side"]) for t in body["triggered"]] == [
        (r.stock_id, r.strategy_id, s) for r, s in report.triggered_today()]
    assert [e["who"] for e in body["errors"]] == [w for w, _ in report.errors] == [
        "策略 broken", "股票 5678（策略 foreign）"]  # 5678 沒有籌碼資料
    assert body["stale"] == [{"stock_id": "5678", "date": "2024-07-29"}]
    only = client.get("/api/screen?strategy=cross").json()
    assert only["strategies"] == 1 and only["errors"] == []


def test_screen_forming_matches_services(client, env, loader):
    """形成中：和 services 相同；數值用白話名稱，前 N 天的值寫成「前 N 天的…」。"""
    _, sdir, _ = env
    for adjusted in (False, True):
        body = client.get(f"/api/screen{'?adjusted=true' if adjusted else ''}").json()
        report = screen(["1234", "5678"], loader=loader, directory=sdir, adjusted=adjusted)
        assert body["has_forming"] is True
        assert [(t["stock_id"], t["strategy_id"], t["days"]) for t in body["forming"]] == [
            (r.stock_id, r.strategy_id, days) for r, days in report.forming()]
        for t, (r, _) in zip(body["forming"], report.forming()):
            assert [v["value"] for v in t["values"]] == [
                r.values[c] for c in ("cci", "close", "ma20（前1天）")]
    [item] = client.get("/api/screen").json()["forming"]  # 只有 1234；5678 最後一天不成立
    assert item["stock_id"] == "1234" and item["days"] == 4
    assert [v["label"] for v in item["values"]] == [
        "CCI 順勢指標（20 日）", "收盤價", "前 1 天的 簡單移動平均（20 日）"]
    # 只選沒有寫形成中條件的策略
    assert client.get("/api/screen?strategy=foreign").json()["has_forming"] is False


def test_overview(client):
    body = client.get("/api/overview").json()
    assert [w["stock_id"] for w in body["watchlist"]] == ["1234", "5678"]
    by = {s["id"]: s for s in body["strategies"]}
    assert set(by) == {"broken", "cross", "foreign"} and by["broken"]["error"]
    assert by["cross"]["forming"] == "CCI 順勢指標（20 日） < 0 且 收盤價 < 前 1 天的 簡單移動平均（20 日）"
    assert by["foreign"]["forming"] is None
    assert by["cross"]["indicators"] == [{"name": "cci", "label": "CCI 順勢指標（20 日）"},
                                         {"name": "ma20", "label": "簡單移動平均（20 日）"}]
    assert body["indicator_sets"][0]["id"] == "common"


# ---------------------------------------------------------------------------
# 輸入檢查


@pytest.mark.parametrize(("url", "status"), [
    ("/api/stocks/9999", 404),  # 本機沒有資料
    ("/api/stocks/9999/prices", 404),
    ("/api/stocks/12%2F34", 404),
    ("/api/stocks/1234567", 422),
    ("/api/stocks/1234/signals?strategy=nope", 404),
    ("/api/stocks/1234/signals?strategy=../strategies/cross", 404),  # 只接受檔名，不接受路徑
    ("/api/stocks/1234/signals?strategy=broken", 422),  # 設定檔有誤
    ("/api/stocks/1234/backtest?strategy=cross&execution=noon", 422),
    ("/api/stocks/1234/backtest?strategy=cross&fee_discount=0", 422),
    ("/api/stocks/1234/compare?strategy=cross&sort=luck", 422),
    ("/api/nope", 404),
])
def test_bad_input(client, url, status):
    r = client.get(url)
    assert r.status_code == status, (url, r.text)
    assert "detail" in r.json()


@pytest.mark.parametrize(("url", "message"), [
    ("/api/stocks/1234/backtest?strategy=cross&fee_discount=0", "手續費折扣要大於 0"),
    ("/api/stocks/1234/backtest?strategy=cross&tax=abc", "證交稅要是數字"),
    ("/api/stocks/1234/backtest?strategy=cross&slippage=0.5", "單邊滑價要小於 0.1"),
    ("/api/stocks/1234/backtest?strategy=cross&start=2024-13-01", "起始日要是正確的日期（YYYY-MM-DD）"),
    ("/api/stocks/1234/chips?days=0", "明細天數要大於 0"),
    ("/api/stocks/1234/backtest", "沒有指定策略"),
    ("/api/stocks/1234567", "股票代號的格式不對"),
])
def test_bad_input_message_is_plain_chinese(client, url, message):
    """網址參數格式不對：網頁直接顯示這段訊息，要是中文、用畫面上的名稱，不是程式的參數名稱。"""
    r = client.get(url)
    assert r.status_code == 422
    assert r.json() == {"detail": message}


def test_missing_stock_message(client):
    r = client.get("/api/stocks/9999/signals?strategy=cross")
    assert r.status_code == 404 and "9999" in r.json()["detail"]


def test_api_responses_are_not_cached(client):
    r = client.get("/api/overview")
    assert r.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


# ---------------------------------------------------------------------------
# 前端檔案


def test_serves_frontend(env, tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>hermes</title>", encoding="utf-8")
    (dist / "assets" / "app-123.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    app = create_app(WebSettings("cid", frozenset({EMAIL}), "s" * 40, static_dir=dist), lambda c: {})
    c = TestClient(app, base_url="https://testserver")
    assert "<title>hermes</title>" in c.get("/").text
    assert "<title>hermes</title>" in c.get("/stocks/2330/signals").text  # 前端路由
    r = c.get("/assets/app-123.js")
    assert r.text == "console.log(1)" and "immutable" in r.headers["cache-control"]
    assert c.get("/favicon.svg").text == "<svg/>"
    assert "secret" not in c.get("/..%2Fsecret.txt").text
    assert c.get("/api/overview").status_code == 401  # API 不會被前端路由蓋掉
