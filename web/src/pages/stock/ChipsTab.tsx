import { useState, type FormEvent } from "react"
import { useQuery } from "@tanstack/react-query"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ErrorBox, Field, Loading } from "@/components/common"
import { api, query, type Chips, type Num, type StockInfo } from "@/lib/api"
import { num, pct, signed, tone } from "@/lib/format"
import { cn } from "@/lib/utils"
import { useParam } from "@/pages/stock/shared"

function Signed({ v, digits = 0 }: { v: Num; digits?: number }) {
  return <span className={cn("tabular-nums", tone(v))}>{signed(v, digits)}</span>
}

/** 籌碼面：三大法人、外資持股比例、融資融券。N 日由你指定，沒有預設值。 */
export function ChipsTab({ stock }: { stock: StockInfo }) {
  const [windows, setWindows] = useParam("w")
  const [days, setDays] = useParam("days")
  const [draft, setDraft] = useState({ windows, days: days || "10" })
  const ws = windows.split(/[,，\s]+/).filter(Boolean)
  const q = useQuery({
    queryKey: ["chips", stock.stock_id, windows, days],
    queryFn: () => api<Chips>(`/api/stocks/${stock.stock_id}/chips${query({ window: ws, days: days || 10 })}`),
  })
  const apply = (e: FormEvent) => {
    e.preventDefault()
    setWindows(draft.windows.trim())
    setDays(draft.days.trim())
  }
  const d = q.data
  const asOf = (date: string) => (d && date !== d.as_of ? `（${date}）` : "")
  const changes = (c: Record<string, Num>, digits: number) =>
    d && d.windows.length ? (
      <span className="text-muted-foreground">
        （和 N 個交易日前相比：
        {d.windows.map((w, i) => (
          <span key={w}>
            {i > 0 ? "、" : ""}
            {w} 日 <Signed v={c[String(w)]} digits={digits} />
          </span>
        ))}
        ）
      </span>
    ) : null
  return (
    <div className="space-y-4">
      <form onSubmit={apply} className="flex flex-wrap items-end gap-3">
        <Field label="近 N 個交易日（可多個，例如 5, 20）">
          <Input
            value={draft.windows}
            onChange={(e) => setDraft({ ...draft, windows: e.target.value })}
            placeholder="5, 20"
          />
        </Field>
        <Field label="明細顯示最近幾個交易日">
          <Input
            type="number"
            min={1}
            max={250}
            value={draft.days}
            onChange={(e) => setDraft({ ...draft, days: e.target.value })}
          />
        </Field>
        <Button type="submit">套用</Button>
      </form>
      {q.isPending ? <Loading /> : null}
      {q.error ? <ErrorBox error={q.error} /> : null}
      {d ? (
        <>
          <Card>
            <CardHeader>
              <CardTitle>籌碼面{d.as_of ? `　資料日期 ${d.as_of}` : ""}</CardTitle>
              {!d.windows.length ? (
                <CardDescription>要看近 N 個交易日的合計和變化，在上面填天數。</CardDescription>
              ) : null}
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              {Object.entries(d.streaks).some(([, v]) => v) ? (
                <p>
                  連續買賣超：
                  {Object.entries(d.streaks)
                    .filter(([, v]) => v)
                    .map(([k, v]) => `${k} 連${v > 0 ? "買" : "賣"} ${Math.abs(v)} 日`)
                    .join("、")}
                </p>
              ) : null}
              {d.foreign_ratio ? (
                <p>
                  外資持股比例{asOf(d.foreign_ratio.date)}：<b>{num(d.foreign_ratio.now)}%</b>
                  {changes(d.foreign_ratio.changes, 2)}
                </p>
              ) : null}
              {d.margin ? (
                <>
                  <p>
                    融資餘額{asOf(d.margin.date)}：{num(d.margin.margin_balance, 0)} 張{changes(d.margin.changes, 0)}
                    ；使用率 {pct(d.margin.margin_usage)}
                  </p>
                  <p>
                    融券餘額：{num(d.margin.short_balance, 0)} 張；券資比 {pct(d.margin.short_margin_ratio)}
                  </p>
                </>
              ) : null}
            </CardContent>
          </Card>
          {d.flows.length && d.flow_columns.length ? (
            <Card>
              <CardHeader>
                <CardTitle>三大法人買賣超合計（張）</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>法人</TableHead>
                      {d.flow_columns.map((c) => (
                        <TableHead key={c} className="text-right">
                          {c}
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {d.flows.map((f) => (
                      <TableRow key={f.who}>
                        <TableCell>{f.who}</TableCell>
                        {f.values.map((v, i) => (
                          <TableCell key={i} className="text-right">
                            <Signed v={v} />
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : null}
          {d.recent.length ? (
            <Card>
              <CardHeader>
                <CardTitle>最近 {d.recent.length} 個交易日明細（張）</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>日期</TableHead>
                      {d.recent_columns.map((c) => (
                        <TableHead key={c} className="text-right">
                          {c}
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {d.recent.map((r) => (
                      <TableRow key={r.date}>
                        <TableCell>{r.date}</TableCell>
                        {r.values.map((v, i) => (
                          <TableCell key={i} className="text-right">
                            <Signed v={v} />
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : null}
        </>
      ) : null}
    </div>
  )
}
