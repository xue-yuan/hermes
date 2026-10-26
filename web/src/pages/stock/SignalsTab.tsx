import { useQueries, useQuery, type UseQueryResult } from "@tanstack/react-query"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { SignalsChart } from "@/components/charts/SignalsChart"
import { ADJUSTED_HELP, ErrorBox, Field, Loading, Warnings, priceName } from "@/components/common"
import { useAdjusted } from "@/hooks/useAdjusted"
import { api, query, type Backtest, type Prices, type Signals, type StockInfo, type Streaks } from "@/lib/api"
import { conditionState, indicator, price } from "@/lib/format"
import { cn } from "@/lib/utils"
import { StrategyChecks, StrategyRules, useParam, useParamList, yearsBefore } from "@/pages/stock/shared"

type SignalRow = Signals["rows"][number]

// 欄位太多時表格可以左右捲動，日期和收盤固定在左邊；要有底色，捲動時才不會被後面的欄位蓋過去
const STICKY_DATE = "sticky left-0 z-10 w-28 min-w-28 bg-card"
const STICKY_CLOSE = "sticky left-28 z-10 w-20 min-w-20 bg-card shadow-[inset_-1px_0_0_var(--color-border)]"

function SignalLabel({ r }: { r: Streaks }) {
  const parts: [number, string, string][] = [
    [r.buy_streak, "買", "text-up"],
    [r.sell_streak, "賣", "text-down"],
  ]
  const shown = parts.filter(([n]) => n > 0)
  if (!shown.length) return <span>無</span>
  return (
    <>
      {shown.map(([n, label, color], i) => (
        <span key={label} className={color}>
          {i > 0 ? "、" : ""}
          {n === 1 ? <b>{label}</b> : `${label}（第 ${n} 天）`}
        </span>
      ))}
    </>
  )
}

// 放在元件外面，查詢結果沒變時 list 也不會變，圖才不會一直重畫
const combineSignals = (results: UseQueryResult<Signals>[]) => ({
  results,
  list: results.flatMap((r) => (r.data ? [r.data] : [])),
  pending: results.some((r) => r.isPending),
})

const combineBacktests = (results: UseQueryResult<Backtest>[]) => ({
  results,
  list: results.flatMap((r) => (r.data ? [r.data] : [])),
  pending: results.some((r) => r.isPending),
})

/**
 * 歷史買賣點：上面是一張圖，股價下面接著每個策略用到的指標（買賣點標在每個策略的第一格），一次看完；
 * 沒有勾策略時只畫股價。圖下面是說明和每個策略的規則、最新狀態；最後是表格（多個策略時用分頁切換，第一頁是總表）。
 * 表格和統計看選的區間（預設最近一年，和 CLI 相同）。
 */
export function SignalsTab({ stock, dark }: { stock: StockInfo; dark: boolean }) {
  const adjusted = useAdjusted()
  const name = priceName(adjusted)
  const [strategies, setStrategies] = useParamList("strategy")
  const [preset, setPreset] = useParam("range")
  const [start, setStart] = useParam("start")
  const [end, setEnd] = useParam("end")
  const range = preset || "1y"
  const from =
    range === "custom"
      ? start
      : range === "all"
        ? stock.first_date
        : range === "3y"
          ? yearsBefore(stock.last_date, 3)
          : ""
  const to = range === "custom" ? end : ""
  // 股價另外拿，沒有勾策略也畫得出來；圖一開始顯示的區間和表格相同
  const mode = adjusted || null
  const px = useQuery({
    queryKey: ["prices", stock.stock_id, from, to, adjusted],
    queryFn: () =>
      api<Prices>(`/api/stocks/${stock.stock_id}/prices${query({ start: from, end: to, adjusted: mode })}`),
  })
  const sig = useQueries({
    queries: strategies.map((strategy) => ({
      queryKey: ["signals", stock.stock_id, strategy, from, to, adjusted],
      queryFn: () =>
        api<Signals>(
          `/api/stocks/${stock.stock_id}/signals${query({ strategy, start: from, end: to, adjusted: mode })}`,
        ),
    })),
    combine: combineSignals,
  })
  // 圖上只標實際交易的訊號日，用回測的交易明細，條件和「回測」分頁的預設相同；圖用全部歷史，所以不跟著表格的區間變
  const bt = useQueries({
    queries: strategies.map((strategy) => ({
      queryKey: ["signals-backtest", stock.stock_id, strategy, adjusted],
      queryFn: () => api<Backtest>(`/api/stocks/${stock.stock_id}/backtest${query({ strategy, adjusted: mode })}`),
    })),
    combine: combineBacktests,
  })
  const failed = strategies.filter((_, i) => sig.results[i]?.error)
  const first = sig.list[0]

  return (
    <div className="space-y-4">
      <Field label="策略（可以選多個）" group>
        <StrategyChecks value={strategies} onChange={setStrategies} />
      </Field>
      <div className="flex flex-wrap items-end gap-3">
        <Field label="表格和統計的區間" group>
          <ToggleGroup type="single" variant="outline" value={range} onValueChange={(v) => v && setPreset(v)}>
            <ToggleGroupItem value="1y">近 1 年</ToggleGroupItem>
            <ToggleGroupItem value="3y">近 3 年</ToggleGroupItem>
            <ToggleGroupItem value="all">全部</ToggleGroupItem>
            <ToggleGroupItem value="custom">自訂</ToggleGroupItem>
          </ToggleGroup>
        </Field>
        {range === "custom" ? (
          <>
            <Field label="起始日">
              <Input
                type="date"
                value={start}
                min={stock.first_date}
                max={stock.last_date}
                onChange={(e) => setStart(e.target.value)}
              />
            </Field>
            <Field label="結束日">
              <Input
                type="date"
                value={end}
                min={stock.first_date}
                max={stock.last_date}
                onChange={(e) => setEnd(e.target.value)}
              />
            </Field>
          </>
        ) : null}
      </div>
      {!strategies.length ? (
        <p className="text-sm text-muted-foreground">
          圖是股價（{name}）。勾選策略，看歷史買賣點；可以選多個：股價下面接著每個策略用到的指標，買賣點標在上面。
        </p>
      ) : null}
      {px.error ? <ErrorBox error={px.error} /> : null}
      {sig.results.map((q, i) => (q.error ? <ErrorBox key={strategies[i]} error={q.error} /> : null))}
      {px.isPending || sig.pending ? <Loading /> : null}
      {px.data && !sig.pending ? (
        <Card className="py-2">
          <CardContent className="px-2">
            <SignalsChart
              price={px.data.series}
              strategies={sig.list}
              backtests={bt.list}
              dark={dark}
              from={px.data.range.start}
              to={px.data.range.end}
            />
          </CardContent>
        </Card>
      ) : null}
      {!sig.pending && first ? (
        <>
          {bt.pending ? <p className="text-sm text-muted-foreground">買賣點計算中…</p> : null}
          {bt.results.map((q, i) => (q.error ? <ErrorBox key={strategies[i]} error={q.error} /> : null))}
          <div className="space-y-1 text-sm text-muted-foreground">
            <p>
              最上面是股價（{name}
              ）；下面每個策略一段，畫策略用到的指標，每一格的左上角寫策略名稱和線的名稱。跟股價同單位的指標（均線、SAR
              這類）和收盤價（灰線）畫在同一格，其他指標各自一格。黃色實線是買賣條件裡的門檻，灰色虛線是策略另外寫的參考線（只畫線，不影響買賣）。
            </p>
            <p>
              ▲ 買進、▼
              賣出標在每個策略的第一格，是實際交易的訊號日，下一個交易日開盤成交；持有中又出現的買進訊號、空手時的賣出訊號不會交易，不標。
            </p>
            <p>
              買賣判斷用{name}。{adjusted ? ADJUSTED_HELP : null}
            </p>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            {sig.list.map((data) => (
              <StrategyCard
                key={data.strategy.id}
                data={data}
                b={bt.list.find((b) => b.strategy.id === data.strategy.id)}
              />
            ))}
          </div>
          <SignalTables list={sig.list} ids={strategies} failed={failed} />
        </>
      ) : null}
    </div>
  )
}

/** 一個策略的規則和最新狀態（放在圖的下面）。 */
function StrategyCard({ data, b }: { data: Signals; b: Backtest | undefined }) {
  const latest = data.latest
  return (
    <div className="space-y-2">
      <Card>
        <CardHeader>
          <CardTitle>{data.strategy.name}</CardTitle>
          {b ? <CardDescription>買賣點從 {b.start} 開始（指標都算得出來的第一天）</CardDescription> : null}
        </CardHeader>
        <CardContent className="space-y-3">
          <StrategyRules s={data.strategy} />
          <div className="rounded-md bg-muted/50 p-3 text-sm">
            <p>
              最新一天（{latest.date}）：買進條件{conditionState(latest.buy_streak)}；賣出條件
              {conditionState(latest.sell_streak)}
            </p>
            <p>
              最近一次買進訊號：{latest.last_buy ?? "—"}；最近一次賣出訊號：{latest.last_sell ?? "—"}
            </p>
          </div>
        </CardContent>
      </Card>
      <Warnings items={data.warnings} />
    </div>
  )
}

/** 表格：一個策略時直接列出；多個策略時用分頁切換，第一頁是總表。 */
function SignalTables({ list, ids, failed }: { list: Signals[]; ids: string[]; failed: string[] }) {
  const [tab, setTab] = useParam("table")
  if (ids.length === 1) return <SignalTable data={list[0]} />
  const current = list.some((d) => d.strategy.id === tab) ? tab : "all"
  return (
    <Tabs value={current} onValueChange={(v) => setTab(v === "all" ? null : v)}>
      <TabsList className="flex h-auto flex-wrap">
        <TabsTrigger value="all">總表</TabsTrigger>
        {list.map((d) => (
          <TabsTrigger key={d.strategy.id} value={d.strategy.id}>
            {d.strategy.name}
          </TabsTrigger>
        ))}
      </TabsList>
      <TabsContent value="all">
        <SummaryTable list={list} failed={failed} />
      </TabsContent>
      {list.map((d) => (
        <TabsContent key={d.strategy.id} value={d.strategy.id}>
          <SignalTable data={d} />
        </TabsContent>
      ))}
    </Tabs>
  )
}

/** 總表：任一策略有訊號的日子，每個策略一欄；那天沒有訊號的策略寫「—」。 */
function SummaryTable({ list, failed }: { list: Signals[]; failed: string[] }) {
  const byDate = new Map<string, { close: number; cells: (SignalRow | undefined)[] }>()
  list.forEach((d, k) => {
    for (const r of d.rows) {
      const row = byDate.get(r.date) ?? { close: r.close, cells: list.map(() => undefined) }
      row.cells[k] = r
      byDate.set(r.date, row)
    }
  })
  // 新的日子在上面，和各策略的表格相同
  const dates = [...byDate.keys()].sort().reverse()
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          總表：{list[0].range.start} ～ {list[0].range.end}
        </CardTitle>
        <CardDescription>
          列出任一策略的條件成立的日子；「—」是那個策略當天沒有訊號。指標數值看各策略的分頁。
          {failed.length ? `有 ${failed.length} 個策略出錯，沒有列入（見上面）。` : null}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {dates.length ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className={STICKY_DATE}>日期</TableHead>
                <TableHead className={cn(STICKY_CLOSE, "text-right")}>收盤</TableHead>
                {list.map((d) => (
                  <TableHead key={d.strategy.id}>{d.strategy.name}</TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {dates.map((date) => {
                const row = byDate.get(date)!
                return (
                  <TableRow key={date}>
                    <TableCell className={STICKY_DATE}>{date}</TableCell>
                    <TableCell className={cn(STICKY_CLOSE, "text-right tabular-nums")}>{price(row.close)}</TableCell>
                    {row.cells.map((r, k) => (
                      <TableCell key={list[k].strategy.id}>
                        {r ? <SignalLabel r={r} /> : <span className="text-muted-foreground">—</span>}
                      </TableCell>
                    ))}
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        ) : (
          <p className="text-sm text-muted-foreground">這段期間沒有任何訊號。</p>
        )}
      </CardContent>
    </Card>
  )
}

function SignalTable({ data }: { data: Signals }) {
  const sm = data.summary
  const cols = data.indicator_columns
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          {data.range.start} ～ {data.range.end}
        </CardTitle>
        <CardDescription>
          買進訊號 {sm.buy_signals} 次（條件成立 {sm.buy_days} 天）、賣出訊號 {sm.sell_signals} 次（條件成立{" "}
          {sm.sell_days} 天）
        </CardDescription>
      </CardHeader>
      <CardContent>
        {data.rows.length ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className={STICKY_DATE}>日期</TableHead>
                <TableHead className={cn(STICKY_CLOSE, "text-right")}>收盤</TableHead>
                {cols.map((c) => (
                  <TableHead key={c} className="text-right">
                    {data.column_labels[c] ?? c}
                  </TableHead>
                ))}
                <TableHead>訊號</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.rows.map((r) => (
                <TableRow key={r.date}>
                  <TableCell className={STICKY_DATE}>{r.date}</TableCell>
                  <TableCell className={cn(STICKY_CLOSE, "text-right tabular-nums")}>{price(r.close)}</TableCell>
                  {cols.map((c) => (
                    <TableCell key={c} className="text-right tabular-nums">
                      {indicator(r.values[c])}
                    </TableCell>
                  ))}
                  <TableCell>
                    <SignalLabel r={r} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <p className="text-sm text-muted-foreground">這段期間沒有任何訊號。</p>
        )}
      </CardContent>
    </Card>
  )
}
