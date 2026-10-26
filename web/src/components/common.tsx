import type { ReactNode } from "react"
import { AlertTriangle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Skeleton } from "@/components/ui/skeleton"
import { ApiError } from "@/lib/api"
import { cn } from "@/lib/utils"

export const DISCLAIMER = "※ 本工具僅供研究與教育用途，回測績效不代表未來報酬，不構成投資建議。"
/** 還原價格的白話說明：還原模式時，頁面上第一次出現「還原價格」的地方要附上。 */
export const ADJUSTED_HELP =
  "還原價格：把除權息、減資、分割以前的股價按比例調整，扣掉這些事件造成的跳空（也叫「還原權值」），成交量也按股數比例調整。"

/** 判斷用的價格：還原模式用還原價格，否則用實際成交價。每個模式只顯示自己那種價格。 */
export function priceName(adjusted: boolean): string {
  return adjusted ? "還原價格" : "實際成交價"
}

/** 資料問題的提醒：缺資料不補值、沒有官方紀錄的跳空、還原價格不比對。 */
export function Warnings({ items, title = "注意" }: { items: string[]; title?: string }) {
  if (!items.length) return null
  return (
    <Alert className="border-warn/50 bg-warn-bg">
      <AlertTriangle className="text-warn" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>
        <ul className="list-disc space-y-1 pl-4">
          {items.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      </AlertDescription>
    </Alert>
  )
}

export function ErrorBox({ error }: { error: unknown }) {
  const msg = error instanceof ApiError || error instanceof Error ? error.message : String(error)
  return (
    <Alert variant="destructive">
      <AlertTriangle />
      <AlertTitle>錯誤</AlertTitle>
      <AlertDescription className="whitespace-pre-line">{msg}</AlertDescription>
    </Alert>
  )
}

export function Loading() {
  return (
    <div className="space-y-3">
      <Skeleton className="h-8 w-1/3" />
      <Skeleton className="h-64 w-full" />
    </div>
  )
}

export function Side({ side, children }: { side: "buy" | "sell"; children?: ReactNode }) {
  return (
    <span className={cn("font-semibold", side === "buy" ? "text-up" : "text-down")}>
      {children ?? (side === "buy" ? "買進" : "賣出")}
    </span>
  )
}

/** 欄位名稱加上輸入框；group 用在一組按鈕、勾選框（用 label 包住會把名稱套到第一個按鈕上）。 */
export function Field({
  label,
  children,
  className,
  group = false,
}: {
  label: string
  children: ReactNode
  className?: string
  group?: boolean
}) {
  const style = cn("flex flex-col gap-1.5 text-sm", className)
  const title = <span className="text-muted-foreground">{label}</span>
  if (group) {
    return (
      <div role="group" aria-label={label} className={style}>
        {title}
        {children}
      </div>
    )
  }
  return (
    <label className={style}>
      {title}
      {children}
    </label>
  )
}

export function Disclaimer() {
  return <p className="text-xs text-muted-foreground">{DISCLAIMER}</p>
}
