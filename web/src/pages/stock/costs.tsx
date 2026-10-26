import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Field } from "@/components/common"
import type { Costs } from "@/lib/api"
import { pct } from "@/lib/format"

// 和 CLI 的 hermes backtest 相同：手續費不打折、滑價 0、股票證交稅 0.3%
export const COST_FIELDS = { execution: "open", fee_discount: "1", slippage: "0", tax: "0.003" }
export type CostForm = typeof COST_FIELDS & { start: string; end: string }

export function CostInputs({
  form,
  onChange,
  first,
  last,
}: {
  form: CostForm
  onChange: (f: CostForm) => void
  first: string
  last: string
}) {
  const set = (k: keyof CostForm) => (e: { target: { value: string } }) => onChange({ ...form, [k]: e.target.value })
  return (
    <>
      <Field label="起始日（預設指標都算得出來的第一天）">
        <Input type="date" min={first} max={last} value={form.start} onChange={set("start")} />
      </Field>
      <Field label="結束日">
        <Input type="date" min={first} max={last} value={form.end} onChange={set("end")} />
      </Field>
      <Field label="成交方式">
        <Select value={form.execution} onValueChange={(v) => onChange({ ...form, execution: v })}>
          <SelectTrigger className="w-40" aria-label="成交方式">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="open">隔天開盤</SelectItem>
            <SelectItem value="close">當天收盤</SelectItem>
          </SelectContent>
        </Select>
      </Field>
      <Field label="手續費折扣（0.28 = 2.8 折）">
        <Input className="w-28" inputMode="decimal" value={form.fee_discount} onChange={set("fee_discount")} />
      </Field>
      <Field label="單邊滑價">
        <Input className="w-24" inputMode="decimal" value={form.slippage} onChange={set("slippage")} />
      </Field>
      <Field label="證交稅（ETF 0.001）">
        <Input className="w-24" inputMode="decimal" value={form.tax} onChange={set("tax")} />
      </Field>
    </>
  )
}

export function costText(c: Costs): string {
  return `手續費 ${pct(c.commission, 4)}（折扣 ${c.discount}）、證交稅 ${pct(c.tax, 1)}、單邊滑價 ${pct(c.slippage)}`
}
