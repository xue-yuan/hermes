import { useSearchParams } from "react-router"
import { Checkbox } from "@/components/ui/checkbox"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useOverview } from "@/components/Layout"
import type { StrategyInfo } from "@/lib/api"

/** 網址上的查詢參數：選了什麼都留在網址，可以直接分享或重新整理。 */
export function useParam(name: string): [string, (v: string | null) => void] {
  const [params, setParams] = useSearchParams()
  const set = (v: string | null) =>
    setParams(
      (p) => {
        if (v === null || v === "") p.delete(name)
        else p.set(name, v)
        return p
      },
      { replace: true },
    )
  return [params.get(name) ?? "", set]
}

/** 可以有多個值的查詢參數，例如 ?strategy=a&strategy=b。 */
export function useParamList(name: string): [string[], (v: string[]) => void] {
  const [params, setParams] = useSearchParams()
  const set = (v: string[]) =>
    setParams(
      (p) => {
        p.delete(name)
        v.forEach((x) => p.append(name, x))
        return p
      },
      { replace: true },
    )
  return [params.getAll(name), set]
}

export function useStrategies(): StrategyInfo[] {
  const overview = useOverview()
  return (overview.data?.strategies ?? []).filter((s): s is StrategyInfo & { error: null } => s.error === null)
}

export function StrategySelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const strategies = useStrategies()
  return (
    <Select value={value || undefined} onValueChange={onChange}>
      <SelectTrigger className="w-full sm:w-72" aria-label="策略">
        <SelectValue placeholder="選擇策略" />
      </SelectTrigger>
      <SelectContent>
        {strategies.map((s) => (
          <SelectItem key={s.id} value={s.id}>
            {s.name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

/** 勾選多個策略；新勾的排在最後。 */
export function StrategyChecks({ value, onChange }: { value: string[]; onChange: (v: string[]) => void }) {
  const strategies = useStrategies()
  const toggle = (id: string, on: boolean) => onChange(on ? [...value, id] : value.filter((x) => x !== id))
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-2">
      {strategies.map((s) => (
        <label key={s.id} className="flex items-center gap-2 text-sm">
          <Checkbox checked={value.includes(s.id)} onCheckedChange={(v) => toggle(s.id, v === true)} />
          {s.name}
        </label>
      ))}
    </div>
  )
}

export function StrategyRules({ s }: { s: StrategyInfo }) {
  return (
    <div className="space-y-0.5 text-sm">
      <p>
        <span className="text-up">買進</span>：{s.buy}
      </p>
      <p>
        <span className="text-down">賣出</span>：{s.sell}
      </p>
      {s.forming !== null ? (
        <p>
          <span className="text-muted-foreground">形成中</span>：{s.forming}
        </p>
      ) : null}
    </div>
  )
}

/** YYYY-MM-DD 往前 years 年（只用來決定顯示的區間）。 */
export function yearsBefore(day: string, years: number): string {
  const [y, m, d] = day.split("-").map(Number)
  const t = new Date(Date.UTC(y - years, m - 1, d))
  return t.toISOString().slice(0, 10)
}
