import { Fragment } from "react"
import { useQuery } from "@tanstack/react-query"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ADJUSTED_HELP, ErrorBox, Field, Loading, Warnings, priceName } from "@/components/common"
import { useOverview } from "@/components/Layout"
import { useAdjusted } from "@/hooks/useAdjusted"
import { api, query, type StockInfo, type Tech } from "@/lib/api"
import { indicator, price } from "@/lib/format"
import { cn } from "@/lib/utils"
import { useParam } from "@/pages/stock/shared"

/** 指標組在某一天的數值：只列數值，不判斷買賣。用實際成交價算，還原模式用還原價格。 */
export function TechTab({ stock }: { stock: StockInfo }) {
  const adjusted = useAdjusted()
  const overview = useOverview()
  const [set, setSet] = useParam("set")
  const [day, setDay] = useParam("date")
  const sets = (overview.data?.indicator_sets ?? []).filter(
    (s): s is { id: string; name: string; description: string; error: null } => s.error === null,
  )
  const chosen = set || "common"
  const q = useQuery({
    queryKey: ["tech", stock.stock_id, chosen, day, adjusted],
    queryFn: () =>
      api<Tech>(`/api/stocks/${stock.stock_id}/tech${query({ set: chosen, date: day, adjusted: adjusted || null })}`),
  })
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <Field label="指標組">
          <Select value={chosen} onValueChange={setSet}>
            <SelectTrigger className="w-full sm:w-64" aria-label="指標組">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {sets.map((s) => (
                <SelectItem key={s.id} value={s.id}>
                  {s.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Field label="日期（預設最新一天）">
          <Input
            type="date"
            value={day}
            min={stock.first_date}
            max={stock.last_date}
            onChange={(e) => setDay(e.target.value)}
          />
        </Field>
      </div>
      {q.isPending ? <Loading /> : null}
      {q.error ? <ErrorBox error={q.error} /> : null}
      {q.data ? (
        <>
          <Warnings items={q.data.warnings} />
          <Card>
            <CardHeader>
              <CardTitle>
                {q.data.date}　收盤 {price(q.data.close)}
                <span className="ml-2 text-sm font-normal text-muted-foreground">（{priceName(q.data.adjusted)}）</span>
              </CardTitle>
              <CardDescription>
                {q.data.set.name}：只列出數值，不判斷買賣。
                {q.data.adjusted ? (
                  <>
                    <br />
                    {ADJUSTED_HELP}
                  </>
                ) : null}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>指標</TableHead>
                    <TableHead className="text-right">數值</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {q.data.rows.map((r, i) => {
                    // 多條線的指標：第一條線前面先列一次指標名稱，下面每列只寫線的名稱
                    const first = r.line !== null && q.data.rows[i - 1]?.indicator !== r.indicator
                    return (
                      <Fragment key={r.column}>
                        {first ? (
                          <TableRow className="hover:bg-transparent">
                            <TableCell colSpan={2} className="whitespace-normal pt-3 font-medium">
                              {r.indicator}
                            </TableCell>
                          </TableRow>
                        ) : null}
                        <TableRow>
                          <TableCell className={cn("whitespace-normal", r.line !== null && "pl-6")}>
                            <span className={cn(r.line !== null && "text-muted-foreground")}>
                              {r.line ?? r.indicator}
                            </span>
                          </TableCell>
                          <TableCell className="text-right tabular-nums">
                            {r.unavailable ? <span className="text-warn">無法計算</span> : indicator(r.value)}
                          </TableCell>
                        </TableRow>
                      </Fragment>
                    )
                  })}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      ) : null}
    </div>
  )
}
