import { useState, type FormEvent } from "react"
import { useQuery } from "@tanstack/react-query"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { EquityChart } from "@/components/charts/EquityChart"
import { ADJUSTED_HELP, ErrorBox, Field, Loading, Warnings, priceName } from "@/components/common"
import { useAdjusted } from "@/hooks/useAdjusted"
import { api, query, type Compare, type Metrics, type StockInfo } from "@/lib/api"
import { METRIC_HELP, SORT_LABELS, num, pct, tone } from "@/lib/format"
import { cn } from "@/lib/utils"
import { COST_FIELDS, CostInputs, costText } from "@/pages/stock/costs"
import { EQUITY_CAPTION, MetricHelp } from "@/pages/stock/BacktestTab"
import { StrategyChecks } from "@/pages/stock/shared"
import { useSearchParams } from "react-router"

const RANK: [string, string, (m: Metrics) => string, string?][] = [
  ["cagr", "年化報酬", (m) => pct(m.cagr), "cagr"],
  ["sharpe", "Sharpe", (m) => num(m.sharpe)],
  ["max_drawdown", "最大回撤", (m) => pct(m.max_drawdown)],
  ["calmar", "Calmar", (m) => num(m.calmar)],
  ["win_rate", "勝率", (m) => pct(m.win_rate, 1)],
  ["trades", "交易次數", (m) => num(m.trades ?? 0, 0)],
  ["exposure", "持倉比例", (m) => pct(m.exposure, 1)],
  ["psr", "PSR", (m) => pct(m.psr, 1)],
]

/** 並排比較設定檔策略，附上買進持有。機器學習策略要長時間訓練，網頁不提供。 */
export function CompareTab({ stock, dark }: { stock: StockInfo; dark: boolean }) {
  const adjusted = useAdjusted()
  const [params, setParams] = useSearchParams()
  // 和其他分頁同一個參數名稱，換分頁時帶著勾好的策略
  const chosen = params.getAll("strategy")
  const sort = params.get("sort") ?? "sharpe"
  const form = {
    start: params.get("start") ?? "",
    end: params.get("end") ?? "",
    ...Object.fromEntries(Object.entries(COST_FIELDS).map(([k, v]) => [k, params.get(k) ?? v])),
  } as typeof COST_FIELDS & { start: string; end: string }
  const [draft, setDraft] = useState({ ...form, s: chosen, sort })
  const q = useQuery({
    queryKey: ["compare", stock.stock_id, params.toString(), adjusted],
    queryFn: () =>
      api<Compare>(
        `/api/stocks/${stock.stock_id}/compare${query({ strategy: chosen, sort, ...form, adjusted: adjusted || null })}`,
      ),
    enabled: chosen.length > 0,
  })
  const submit = (e: FormEvent) => {
    e.preventDefault()
    setParams(
      (p) => {
        p.delete("strategy")
        draft.s.forEach((s) => p.append("strategy", s))
        for (const [k, v] of Object.entries({ ...draft, s: undefined })) {
          if (k === "s") continue
          const val = String(v ?? "").trim()
          if (val && val !== (COST_FIELDS as Record<string, string>)[k] && !(k === "sort" && val === "sharpe"))
            p.set(k, val)
          else p.delete(k)
        }
        return p
      },
      { replace: true },
    )
  }
  return (
    <div className="space-y-4">
      <form onSubmit={submit} className="space-y-3">
        <Field label="要比較的策略" group>
          <StrategyChecks value={draft.s} onChange={(s) => setDraft({ ...draft, s })} />
        </Field>
        <div className="flex flex-wrap items-end gap-3">
          <CostInputs
            form={draft}
            onChange={(f) => setDraft({ ...draft, ...f })}
            first={stock.first_date}
            last={stock.last_date}
          />
          <Field label="排序依據">
            <Select value={draft.sort} onValueChange={(v) => setDraft({ ...draft, sort: v })}>
              <SelectTrigger className="w-56" aria-label="排序依據">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(SORT_LABELS).map(([k, v]) => (
                  <SelectItem key={k} value={k}>
                    {v}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          <Button type="submit" disabled={!draft.s.length}>
            比較
          </Button>
        </div>
      </form>
      {!chosen.length ? <p className="text-sm text-muted-foreground">勾選要比較的策略，按「比較」。</p> : null}
      {q.isFetching && !q.data ? <Loading /> : null}
      {q.error ? <ErrorBox error={q.error} /> : null}
      {q.data ? <CompareView c={q.data} dark={dark} /> : null}
    </div>
  )
}

function CompareView({ c, dark }: { c: Compare; dark: boolean }) {
  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>
            策略比較（{c.start} ～ {c.end}，依 {SORT_LABELS[c.sort] ?? c.sort} 排序）
          </CardTitle>
          <CardDescription>
            訊號用{priceName(c.adjusted)}判斷，報酬包含股利。{c.adjusted ? ADJUSTED_HELP : null}成本：
            {costText(c.costs)}；{c.execution === "open" ? "隔天開盤" : "當天收盤"}成交。
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>策略</TableHead>
                {RANK.map(([k, label]) => (
                  <TableHead key={k} className="text-right" title={METRIC_HELP[k]}>
                    {label}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {[
                ...c.rows.map((r) => ({ ...r, bench: false })),
                { id: "benchmark", name: c.benchmark.name, metrics: c.benchmark.metrics, bench: true },
              ].map((r) => (
                <TableRow key={r.id} className={cn(r.bench && "font-semibold")}>
                  <TableCell>{r.name}</TableCell>
                  {RANK.map(([k, , fmt]) => (
                    <TableCell key={k} className={cn("text-right tabular-nums", k === "cagr" && tone(r.metrics.cagr))}>
                      {fmt(r.metrics)}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <MetricHelp items={RANK.map(([k, label]) => [k, label])} />
        </CardContent>
      </Card>
      <Warnings items={c.warnings} />
      <Card className="py-2">
        <CardContent className="px-2">
          <EquityChart curves={c.curves} dark={dark} />
          <p className="px-2 text-xs text-muted-foreground">{EQUITY_CAPTION}</p>
        </CardContent>
      </Card>
      <p className="text-xs text-muted-foreground">
        比較多個策略後挑最好的，本身就是一種多重檢定，最好的那個可能只是運氣好。
      </p>
    </>
  )
}
