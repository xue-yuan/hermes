// 後端 API 的型別和呼叫方式。所有數字都是後端算好的，前端只負責顯示。

export type Num = number | "Infinity" | "-Infinity" | null

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

// ---------------------------------------------------------------------------
// 登入：access token 只放在記憶體（重新整理頁面就沒了），每次呼叫 API 放在 Authorization；
// refresh token 在 HttpOnly cookie，JavaScript 讀不到，只用來換新的 access token。

let accessToken: string | null = null
let refreshing: Promise<boolean> | null = null

export interface TokenResponse {
  email: string
  access_token: string
  token_type: "Bearer"
  expires_in: number
}

export function setAccessToken(token: string | null) {
  accessToken = token
}

export function hasAccessToken(): boolean {
  return accessToken !== null
}

/**
 * 用 refresh token 換新的 access token。同一個分頁同時只換一次；多個分頁用 Web Locks 排隊，
 * 不然兩個分頁拿同一個 refresh token 去換，第二個會被當成重複使用，整串都被撤銷。
 */
export function refreshAccessToken(): Promise<boolean> {
  refreshing ??= (async () => {
    const run = async () => {
      const res = await fetch("/api/auth/refresh", { method: "POST", credentials: "same-origin" })
      if (!res.ok) {
        accessToken = null
        return false
      }
      accessToken = ((await res.json()) as TokenResponse).access_token
      return true
    }
    return navigator.locks ? navigator.locks.request("hermes-refresh", run) : run()
  })().finally(() => {
    refreshing = null
  })
  return refreshing
}

export async function api<T>(path: string, init?: RequestInit, retry = true): Promise<T> {
  const res = await fetch(path, {
    credentials: "same-origin",
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
    },
  })
  // access token 過期：換一個新的再試一次
  if (res.status === 401 && retry && !path.startsWith("/api/auth/") && (await refreshAccessToken())) {
    return api<T>(path, init, false)
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (typeof body.detail === "string") detail = body.detail
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("；")
    } catch {
      // 不是 JSON，用狀態碼
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

export function query(params: Record<string, string | number | boolean | null | undefined | (string | number)[]>) {
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v === null || v === undefined || v === "") continue
    if (Array.isArray(v)) v.forEach((x) => q.append(k, String(x)))
    else q.set(k, String(v))
  }
  const s = q.toString()
  return s ? `?${s}` : ""
}

// ---------------------------------------------------------------------------
export interface StrategyInfo {
  id: string
  name: string
  description: string
  buy: string
  sell: string
  forming: string | null
  indicators: { name: string; label: string }[]
}

export interface Overview {
  watchlist: { stock_id: string; name: string; added_at: string; last_price_date: string | null }[]
  strategies: ((StrategyInfo & { error: null }) | { id: string; error: string })[]
  indicator_sets: ({ id: string; name: string; description: string; error: null } | { id: string; error: string })[]
  sort_keys: string[]
}

/** 照策略的規則算的部位（不是使用者實際有沒有買）：bought 是買進後還沒賣出；action 是這天收盤後的訊號，null 是不用動。 */
export interface StrategyState {
  date: string
  bought: boolean
  action: "buy" | "sell" | null
}

export interface ScreenItem {
  stock_id: string
  name: string
  strategy_id: string
  strategy: string
  state: StrategyState | null
}

export type ScreenSignal = ScreenItem & { side: "buy" | "sell" }

export interface ScreenReport {
  adjusted: boolean
  latest_date: string | null
  stocks: number
  strategies: number
  stale: { stock_id: string; date: string }[]
  triggered: (ScreenSignal & { close: number })[]
  ongoing: (ScreenSignal & { days: number })[]
  has_forming: boolean
  forming: (ScreenItem & { days: number; values: { label: string; value: Num }[] })[]
  warnings: { who: string; message: string }[]
  errors: { who: string; message: string }[]
}

export interface StockInfo {
  stock_id: string
  name: string
  first_date: string
  last_date: string
  close: number
  in_watchlist: boolean
}

/** 買進、賣出條件連續成立的天數。 */
export interface Streaks {
  buy_streak: number
  sell_streak: number
}

/** 股價圖用的日 K（實際成交價；還原模式是還原價格），不用選策略。 */
export interface Prices {
  stock_id: string
  adjusted: boolean
  range: { start: string; end: string }
  series: {
    date: string[]
    open: Num[]
    high: Num[]
    low: Num[]
    close: Num[]
    volume_lots: Num[]
  }
}

export interface Signals {
  stock_id: string
  name: string
  strategy: StrategyInfo
  adjusted: boolean
  warnings: string[]
  range: { start: string; end: string }
  layout: {
    overlays: string[]
    panels: Record<string, string[]>
    thresholds: Record<string, number[]>
    reference_lines: Record<string, number[]>
  }
  indicator_columns: string[]
  column_labels: Record<string, string>
  series: Record<string, (number | boolean | null)[]> & { date: string[] }
  rows: (Streaks & { date: string; close: number; values: Record<string, Num> })[]
  summary: { buy_signals: number; buy_days: number; sell_signals: number; sell_days: number }
  latest: Streaks & { date: string; last_buy: string | null; last_sell: string | null }
}

export interface Tech {
  stock_id: string
  name: string
  set: { id: string; name: string; description: string }
  date: string
  adjusted: boolean
  close: number
  rows: {
    column: string
    label: string
    indicator: string
    line: string | null
    value: Num
    unavailable: boolean
  }[]
  warnings: string[]
}

export interface Chips {
  stock_id: string
  name: string
  as_of: string | null
  windows: number[]
  flow_columns: string[]
  flows: { who: string; values: Num[] }[]
  streaks: Record<string, number>
  recent_columns: string[]
  recent: { date: string; values: Num[] }[]
  foreign_ratio: { date: string; now: number; changes: Record<string, Num> } | null
  margin: {
    date: string
    margin_balance: number
    margin_usage: Num
    short_balance: number
    short_margin_ratio: Num
    changes: Record<string, Num>
  } | null
}

export type Metrics = Record<string, Num>

export interface Curves {
  date: string[]
  benchmark: { equity: Num[]; drawdown: Num[] }
  series: { id: string; name: string; equity: Num[]; drawdown: Num[]; position: Num[] }[]
}

export interface Trade {
  buy_signal: string
  entry: string
  entry_price: number
  sell_signal: string | null
  exit: string | null
  exit_price: Num
  return: Num
  bars: number
  buy_at_limit: boolean
  sell_at_limit: boolean
}

export interface Costs {
  commission: number
  discount: number
  tax: number
  slippage: number
}

export interface Backtest {
  stock_id: string
  name: string
  strategy: StrategyInfo
  start: string
  end: string
  warmup_end: string
  start_given: boolean
  adjusted: boolean
  execution: "open" | "close"
  costs: Costs
  metrics: { id: string; name: string; metrics: Metrics }[]
  curves: Curves
  trades: Trade[]
  latest: StrategyState | null
  warnings: string[]
}

export interface Compare {
  stock_id: string
  name: string
  sort: string
  adjusted: boolean
  start: string
  end: string
  execution: "open" | "close"
  costs: Costs
  rows: { id: string; name: string; metrics: Metrics }[]
  benchmark: { name: string; metrics: Metrics }
  curves: Curves
  warnings: string[]
}
