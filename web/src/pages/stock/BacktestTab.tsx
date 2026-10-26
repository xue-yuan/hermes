import { useState, type FormEvent } from "react"
import { useSearchParams } from "react-router"
import { useQuery } from "@tanstack/react-query"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { EquityChart } from "@/components/charts/EquityChart"
import { ADJUSTED_HELP, ErrorBox, Field, Loading, Warnings, priceName } from "@/components/common"
import { useAdjusted } from "@/hooks/useAdjusted"
import { api, query, type Backtest, type Metrics, type StockInfo, type Trade } from "@/lib/api"
import { METRIC_ROWS, pct, price, tone } from "@/lib/format"
import { cn } from "@/lib/utils"
import { COST_FIELDS, CostInputs, costText, type CostForm } from "@/pages/stock/costs"
import { StrategyRules, StrategySelect } from "@/pages/stock/shared"

const KEYS = ["strategy", "start", "end", ...Object.keys(COST_FIELDS)] as const

export function useForm(): [CostForm & { strategy: string }, (f: Record<string, string>) => void, URLSearchParams] {
  const [params, setParams] = useSearchParams()
  const form = {
    strategy: params.get("strategy") ?? "",
    start: params.get("start") ?? "",
    end: params.get("end") ?? "",
    ...Object.fromEntries(Object.entries(COST_FIELDS).map(([k, v]) => [k, params.get(k) ?? v])),
  } as CostForm & { strategy: string }
  const apply = (f: Record<string, string>) =>
    setParams(
      (p) => {
        for (const k of KEYS) {
          const v = f[k]?.trim()
          if (v && v !== (COST_FIELDS as Record<string, string>)[k]) p.set(k, v)
          else p.delete(k)
        }
        return p
      },
      { replace: true },
    )
  return [form, apply, params]
}

export function MetricsTable({ cols }: { cols: { name: string; metrics: Metrics }[] }) {
  const rows = METRIC_ROWS.filter(([k]) => cols.some((c) => k in c.metrics))
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>指標</TableHead>
          {cols.map((c) => (
            <TableHead key={c.name} className="text-right">
              {c.name}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map(([k, label, fmt]) => (
          <TableRow key={k}>
            <TableCell className="font-medium">{label}</TableCell>
            {cols.map((c) => (
              <TableCell key={c.name} className="text-right tabular-nums">
                {k in c.metrics ? fmt(c.metrics[k]) : "—"}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function Trades({ trades }: { trades: Trade[] }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          {["買進訊號", "買進日", "買進價", "賣出訊號", "賣出日", "賣出價", "報酬", "持有天數"].map((c) => (
            <TableHead key={c} className="text-right">
              {c}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {trades.map((t) => (
          <TableRow key={t.entry} className="text-right tabular-nums">
            <TableCell>{t.buy_signal}</TableCell>
            <TableCell>{t.entry}</TableCell>
            <TableCell>
              {price(t.entry_price)}
              {t.buy_at_limit ? <span className="ml-1 text-warn">⚠ 漲停</span> : null}
            </TableCell>
            {t.exit === null ? (
              <>
                <TableCell className="text-muted-foreground">持有中</TableCell>
                <TableCell>—</TableCell>
                <TableCell>—</TableCell>
              </>
            ) : (
              <>
                <TableCell>{t.sell_signal}</TableCell>
                <TableCell>{t.exit}</TableCell>
                <TableCell>
                  {price(t.exit_price)}
                  {t.sell_at_limit ? <span className="ml-1 text-warn">⚠ 跌停</span> : null}
                </TableCell>
              </>
            )}
            <TableCell className={cn(tone(t.return))}>{pct(t.return, 2, true)}</TableCell>
            <TableCell>{t.bars}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

/** 設定檔策略的回測：訊號用實際成交價判斷（還原模式用還原價格），報酬包含股利。 */
export function BacktestTab({ stock, dark }: { stock: StockInfo; dark: boolean }) {
  const adjusted = useAdjusted()
  const [form, apply, params] = useForm()
  const [draft, setDraft] = useState(form)
  const q = useQuery({
    queryKey: ["backtest", stock.stock_id, params.toString(), adjusted],
    queryFn: () =>
      api<Backtest>(
        `/api/stocks/${stock.stock_id}/backtest${query({
          strategy: form.strategy,
          start: form.start,
          end: form.end,
          execution: form.execution,
          fee_discount: form.fee_discount,
          slippage: form.slippage,
          tax: form.tax,
          adjusted: adjusted || null,
        })}`,
      ),
    enabled: !!form.strategy,
  })
  const submit = (e: FormEvent) => {
    e.preventDefault()
    apply(draft)
  }
  return (
    <div className="space-y-4">
      <form onSubmit={submit} className="flex flex-wrap items-end gap-3">
        <Field label="策略">
          <StrategySelect value={draft.strategy} onChange={(v) => setDraft({ ...draft, strategy: v })} />
        </Field>
        <CostInputs
          form={draft}
          onChange={(f) => setDraft({ ...draft, ...f })}
          first={stock.first_date}
          last={stock.last_date}
        />
        <Button type="submit" disabled={!draft.strategy}>
          回測
        </Button>
      </form>
      {!form.strategy ? <p className="text-sm text-muted-foreground">選一個策略，按「回測」。</p> : null}
      {q.isFetching && !q.data ? <Loading /> : null}
      {q.error ? <ErrorBox error={q.error} /> : null}
      {q.data ? <BacktestView b={q.data} dark={dark} /> : null}
    </div>
  )
}

function BacktestView({ b, dark }: { b: Backtest; dark: boolean }) {
  const when = b.execution === "open" ? "隔天開盤" : "當天收盤"
  const beginNote = !b.start_given
    ? "（指標都算得出來的第一天）"
    : b.warmup_end > b.start
      ? `（指標到 ${b.warmup_end} 才都算得出來，在這之前不會有訊號）`
      : ""
  const limit = b.trades.some((t) => t.buy_at_limit || t.sell_at_limit)
  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>{b.strategy.name}</CardTitle>
          <CardDescription>回測</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <StrategyRules s={b.strategy} />
          <p>
            期間：{b.start}
            {beginNote} ～ {b.end}，從空手開始
          </p>
          <p>
            做法：買進條件成立後{when}買進，持有到賣出條件成立後{when}賣出；不加碼、不放空
          </p>
          <p>價格：訊號用{priceName(b.adjusted)}判斷；報酬包含股利</p>
          {b.adjusted ? <p className="text-muted-foreground">{ADJUSTED_HELP}</p> : null}
          <p>成本：{costText(b.costs)}</p>
          {b.latest ? (
            <p className="rounded-md bg-muted/50 p-2">
              最新（{b.latest.date} 收盤後）：照策略應該{b.latest.holding ? "持有" : "空手"}
              {b.latest.action ? (
                <b className={b.latest.action === "buy" ? "text-up" : "text-down"}>
                  ，{b.execution === "open" ? "下一個交易日開盤" : "今天收盤"}
                  {b.latest.action === "buy" ? "買進" : "賣出"}
                </b>
              ) : null}
            </p>
          ) : null}
        </CardContent>
      </Card>
      <Warnings items={b.warnings} />
      <Card>
        <CardHeader>
          <CardTitle>績效</CardTitle>
        </CardHeader>
        <CardContent>
          <MetricsTable cols={b.metrics} />
        </CardContent>
      </Card>
      <Card className="py-2">
        <CardContent className="px-2">
          <EquityChart curves={b.curves} dark={dark} />
          <p className="px-2 text-xs text-muted-foreground">上：淨值（對數刻度）；中：回撤 %；下：部位。</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>交易明細（共 {b.trades.length} 筆，新的在前）</CardTitle>
          <CardDescription>價格是當天的成交價，報酬包含股利和成本。</CardDescription>
        </CardHeader>
        <CardContent>
          {b.trades.length ? (
            <Trades trades={b.trades} />
          ) : (
            <p className="text-sm text-muted-foreground">這段期間沒有任何交易。</p>
          )}
          {limit ? (
            <p className="mt-2 text-sm text-warn">
              ⚠ 漲停：用漲停價買進；跌停：用跌停價賣出。實際上可能排不到，回測結果會偏樂觀
            </p>
          ) : null}
        </CardContent>
      </Card>
    </>
  )
}
