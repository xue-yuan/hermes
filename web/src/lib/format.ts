// 顯示用的格式；數字本身都由後端算好，這裡只決定顯示幾位小數。
import type { Num } from "@/lib/api"

const DASH = "—"

function finite(x: Num | undefined): x is number {
  return typeof x === "number" && Number.isFinite(x)
}

export function num(x: Num | undefined, digits = 2): string {
  if (x === "Infinity") return "∞"
  if (x === "-Infinity") return "−∞"
  if (!finite(x)) return DASH
  return x.toLocaleString("zh-TW", { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

export function pct(x: Num | undefined, digits = 2, sign = false): string {
  if (!finite(x)) return num(x)
  const s = (x * 100).toFixed(digits)
  return `${sign && x > 0 ? "+" : ""}${s}%`
}

export function signed(x: Num | undefined, digits = 0): string {
  if (!finite(x)) return num(x)
  return `${x > 0 ? "+" : ""}${num(x, digits)}`
}

/** 價格：最多兩位小數，去掉多餘的 0（和 CLI 相同）。 */
export function price(x: Num | undefined): string {
  if (!finite(x)) return num(x)
  return x.toLocaleString("zh-TW", { minimumFractionDigits: 0, maximumFractionDigits: 2 })
}

/** 指標數值：絕對值小於 1 時多顯示兩位（和 CLI 的 hermes tech 相同）。 */
export function indicator(x: Num | undefined): string {
  if (!finite(x) || x === 0) return num(x, 2)
  // 小於 1 的數字多留兩位小數；一百萬以上（例如 OBV）不用小數
  const a = Math.abs(x)
  return num(x, a < 1 ? 4 : a >= 1_000_000 ? 0 : 2)
}

/** 台股慣例：正數紅、負數綠。 */
export function tone(x: Num | undefined): string {
  if (!finite(x) || x === 0) return ""
  return x > 0 ? "text-up" : "text-down"
}

export function conditionState(streak: number): string {
  if (streak === 1) return "成立（今天新觸發）"
  return streak > 1 ? `成立（已持續 ${streak} 天）` : "不成立"
}

// 績效指標：名稱和格式和 CLI 的回測表相同
export const METRIC_ROWS: [string, string, (v: Num) => string][] = [
  ["total_return", "總報酬", (v) => pct(v)],
  ["cagr", "年化報酬 CAGR", (v) => pct(v)],
  ["ann_vol", "年化波動", (v) => pct(v)],
  ["sharpe", "Sharpe", (v) => num(v)],
  ["sortino", "Sortino", (v) => num(v)],
  ["max_drawdown", "最大回撤", (v) => pct(v)],
  ["calmar", "Calmar", (v) => num(v)],
  ["mdd_duration", "最長水下天數", (v) => num(v, 0)],
  ["psr", "PSR(SR>0)", (v) => pct(v, 1)],
  ["exposure", "持倉時間比例", (v) => pct(v, 1)],
  ["turnover", "年換手（倍）", (v) => num(v, 1)],
  ["trades", "交易次數", (v) => num(v, 0)],
  ["win_rate", "勝率", (v) => pct(v, 1)],
  ["avg_trade", "平均每筆報酬", (v) => pct(v)],
  ["profit_factor", "獲利因子", (v) => num(v)],
  ["avg_hold", "平均持有天數", (v) => num(v, 1)],
  ["alpha", "Alpha（年化）", (v) => pct(v)],
  ["beta", "Beta", (v) => num(v)],
  ["info_ratio", "資訊比率", (v) => num(v)],
]

export const SORT_LABELS: Record<string, string> = {
  sharpe: "Sharpe",
  cagr: "年化報酬 CAGR",
  calmar: "Calmar",
  max_drawdown: "最大回撤（越小越前面）",
}
