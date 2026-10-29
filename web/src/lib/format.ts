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
  if (streak === 1) return "成立（新觸發）"
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

/** 績效指標的白話說明：網頁上每個指標名稱都要看得到它是什麼意思。 */
export const METRIC_HELP: Record<string, string> = {
  total_return: "整段期間賺或賠了多少，包含股利和交易成本。",
  cagr: "把總報酬換算成平均每年的複利報酬率（用經過的日曆時間換算）。",
  ann_vol: "每天報酬的起伏（標準差）換算成一年；越大代表淨值上下震盪越劇烈。",
  sharpe: "平均每天的報酬 ÷ 每天報酬的起伏，再換算成一年：承受同樣的起伏，換到多少報酬。越高越好。",
  sortino: "和 Sharpe 類似，但分母只看下跌的起伏，上漲的起伏不算風險。越高越好。",
  max_drawdown: "淨值從之前的最高點（包含一開始的本金）往下跌，最多跌掉幾 %。",
  calmar: "年化報酬 ÷ 最大回撤：每承受 1% 的回撤，每年賺幾 %。越高越好。",
  mdd_duration: "淨值低於之前的最高點，連續最久有幾個交易日。",
  psr: "考慮報酬分布的偏斜、極端值和資料天數以後，真正的 Sharpe 大於 0 的機率；越接近 100% 越不像運氣。",
  exposure: "策略買進後還沒賣出的交易日，佔全部交易日的比例。",
  turnover: "平均每年買進加賣出的金額是資金的幾倍；買進持有只在一開始買一次。",
  trades: "一次買進到賣出算一筆，還沒賣出的也算。",
  win_rate: "報酬大於 0 的交易佔幾 %。",
  avg_trade: "每筆交易報酬（包含成本）的平均。",
  profit_factor: "賺錢交易的報酬加總 ÷ 賠錢交易的虧損加總；大於 1 代表賺的比賠的多，沒有賠錢的交易時是 ∞。",
  avg_hold: "每筆交易平均持有幾個交易日。",
  alpha: "扣掉跟著買進持有同一檔股票漲跌的部分（見 Beta）以後，平均每年多賺的報酬。",
  beta: "買進持有漲跌 1% 的日子，策略平均跟著漲跌幾 %；空手的日子越多，Beta 越小。",
  info_ratio: "策略每天比買進持有多賺的報酬，平均值 ÷ 起伏，再換算成一年；越高代表贏得越穩定。",
}

export const SORT_LABELS: Record<string, string> = {
  sharpe: "Sharpe",
  cagr: "年化報酬 CAGR",
  calmar: "Calmar",
  max_drawdown: "最大回撤（越小越前面）",
}
