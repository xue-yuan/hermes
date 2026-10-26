import { Link } from "react-router"
import { useQuery } from "@tanstack/react-query"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ADJUSTED_HELP, ErrorBox, Loading, Side, Warnings, priceName } from "@/components/common"
import { useAdjusted } from "@/hooks/useAdjusted"
import { api, query, type ScreenItem, type ScreenReport } from "@/lib/api"
import { price } from "@/lib/format"

function StockCell({ r }: { r: ScreenItem }) {
  return (
    <>
      <TableCell>
        <Link className="font-medium underline-offset-4 hover:underline" to={`/stocks/${r.stock_id}`}>
          {r.stock_id}
        </Link>
      </TableCell>
      <TableCell>{r.name}</TableCell>
      <TableCell>{r.strategy}</TableCell>
    </>
  )
}

/** 每日選股報告：收盤後看，決定下一個交易日開盤要不要買賣。 */
export function ScreenPage() {
  const adjusted = useAdjusted()
  const q = useQuery({
    queryKey: ["screen", adjusted],
    queryFn: () => api<ScreenReport>(`/api/screen${query({ adjusted: adjusted || null })}`),
  })
  if (q.isPending) return <Loading />
  if (q.error) return <ErrorBox error={q.error} />
  const r = q.data
  if (!r.latest_date) {
    return (
      <>
        <Card>
          <CardHeader>
            <CardTitle>每日選股報告</CardTitle>
            <CardDescription>沒有可以計算的股票和策略。自選股、策略都用 CLI 管理。</CardDescription>
          </CardHeader>
        </Card>
        <Warnings title="錯誤" items={r.errors.map((e) => `${e.who}：${e.message}`)} />
      </>
    )
  }
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>每日選股報告</CardTitle>
          <CardDescription>
            資料日期 <b className="text-foreground">{r.latest_date}</b>　{r.stocks} 檔股票 × {r.strategies} 個策略
            <br />
            價格：用{priceName(r.adjusted)}判斷
            {r.adjusted ? (
              <>
                <br />
                {ADJUSTED_HELP}
              </>
            ) : null}
          </CardDescription>
        </CardHeader>
      </Card>
      <Warnings
        items={[
          ...r.stale.map((s) => `${s.stock_id} 的資料只到 ${s.date}，比其他股票舊（CLI 執行 hermes update 更新）`),
          ...r.warnings.map((w) => `${w.who}：${w.message}`),
        ]}
      />
      <Warnings title="錯誤" items={r.errors.map((e) => `${e.who}：${e.message}`)} />

      <Card>
        <CardHeader>
          <CardTitle>① 今天新觸發</CardTitle>
        </CardHeader>
        <CardContent>
          {r.triggered.length ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>代號</TableHead>
                  <TableHead>名稱</TableHead>
                  <TableHead>策略</TableHead>
                  <TableHead>訊號</TableHead>
                  <TableHead className="text-right">收盤</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {r.triggered.map((t) => (
                  <TableRow key={`${t.stock_id}-${t.strategy_id}-${t.side}`}>
                    <StockCell r={t} />
                    <TableCell>
                      <Side side={t.side} />
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{price(t.close)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-sm text-muted-foreground">今天沒有新觸發的訊號。</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>② 條件仍成立</CardTitle>
        </CardHeader>
        <CardContent>
          {r.ongoing.length ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>代號</TableHead>
                  <TableHead>名稱</TableHead>
                  <TableHead>策略</TableHead>
                  <TableHead>訊號</TableHead>
                  <TableHead className="text-right">已持續</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {r.ongoing.map((t) => (
                  <TableRow key={`${t.stock_id}-${t.strategy_id}-${t.side}`}>
                    <StockCell r={t} />
                    <TableCell>
                      <Side side={t.side} />
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{t.days} 天</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-sm text-muted-foreground">沒有持續成立中的條件。</p>
          )}
        </CardContent>
      </Card>

      {r.has_forming ? (
        <Card>
          <CardHeader>
            <CardTitle>③ 形成中</CardTitle>
            <CardDescription>
              策略寫的「形成中」條件今天成立、還沒觸發買進。例如 Darvas
              箱型：箱子已經成形，收盤還在箱頂和箱底之間，等突破箱頂。
            </CardDescription>
          </CardHeader>
          <CardContent>
            {r.forming.length ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>股票</TableHead>
                    <TableHead>策略／今天的數值</TableHead>
                    <TableHead className="text-right">已持續</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {r.forming.map((t) => (
                    <TableRow key={`${t.stock_id}-${t.strategy_id}`}>
                      {/* 手機上也不用左右滑：名稱放在代號下面，數值放在策略下面 */}
                      <TableCell className="align-top">
                        <Link className="font-medium underline-offset-4 hover:underline" to={`/stocks/${t.stock_id}`}>
                          {t.stock_id}
                        </Link>
                        <div className="text-muted-foreground">{t.name}</div>
                      </TableCell>
                      <TableCell className="whitespace-normal align-top">
                        <div>{t.strategy}</div>
                        {t.values.map((v) => (
                          <div key={v.label} className="text-xs">
                            <span className="text-muted-foreground">{v.label}</span>{" "}
                            <span className="tabular-nums">{price(v.value)}</span>
                          </div>
                        ))}
                      </TableCell>
                      <TableCell className="text-right align-top tabular-nums">{t.days} 天</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <p className="text-sm text-muted-foreground">沒有形成中的型態。</p>
            )}
          </CardContent>
        </Card>
      ) : null}
    </div>
  )
}
