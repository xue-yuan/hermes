import { Link } from "react-router"
import { useQuery } from "@tanstack/react-query"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { AdjustedHelp, ErrorBox, Loading, NextAction, Side, Warnings, boughtText, priceName } from "@/components/common"
import { useAdjusted } from "@/hooks/useAdjusted"
import { api, query, type ScreenItem, type ScreenReport } from "@/lib/api"
import { price } from "@/lib/format"

// 手機上名稱放在代號下面、策略名稱可以換行，最右邊的訊號才不會被擠到畫面外面；
// 畫面不夠寬時，「策略目前」「下一個交易日開盤」放在策略名稱下面
function StockCell({ r }: { r: ScreenItem }) {
  return (
    <>
      <TableCell className="align-top">
        <Link className="font-medium underline-offset-4 hover:underline" to={`/stocks/${r.stock_id}`}>
          {r.stock_id}
        </Link>
        <div className="text-muted-foreground sm:hidden">{r.name}</div>
      </TableCell>
      <TableCell className="hidden align-top sm:table-cell">{r.name}</TableCell>
      <TableCell className="whitespace-normal align-top">
        {r.strategy}
        <div className="mt-1 space-y-0.5 text-xs lg:hidden">
          <div>
            <span className="text-muted-foreground">策略目前：</span>
            {/* 值整段一起換行，不會拆成「不」「用動」 */}
            <span className="whitespace-nowrap">{boughtText(r.state)}</span>
          </div>
          <div>
            <span className="text-muted-foreground">下一個交易日開盤：</span>
            <span className="whitespace-nowrap">
              <NextAction state={r.state} />
            </span>
          </div>
        </div>
      </TableCell>
    </>
  )
}

function StockHeads() {
  return (
    <>
      <TableHead>代號</TableHead>
      <TableHead className="hidden sm:table-cell">名稱</TableHead>
      <TableHead>策略</TableHead>
    </>
  )
}

function StateHeads() {
  return (
    <>
      <TableHead className="hidden lg:table-cell">策略目前</TableHead>
      <TableHead className="hidden lg:table-cell">下一個交易日開盤</TableHead>
    </>
  )
}

function StateCells({ r }: { r: ScreenItem }) {
  return (
    <>
      <TableCell className="hidden align-top lg:table-cell">{boughtText(r.state)}</TableCell>
      <TableCell className="hidden align-top lg:table-cell">
        <NextAction state={r.state} />
      </TableCell>
    </>
  )
}

/** 兩欄的意思：策略的部位不是使用者實際有沒有買。 */
const STATE_NOTE =
  "「策略目前」「下一個交易日開盤」是照策略的規則、從指標都算得出來的第一天算的（和回測相同），不是你實際有沒有買。"

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
          </CardDescription>
          {r.adjusted ? <AdjustedHelp /> : null}
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
          <CardTitle>① 新觸發（{r.latest_date}）</CardTitle>
        </CardHeader>
        <CardContent>
          {r.triggered.length ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <StockHeads />
                  <TableHead>訊號</TableHead>
                  <StateHeads />
                  <TableHead className="text-right">收盤</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {r.triggered.map((t) => (
                  <TableRow key={`${t.stock_id}-${t.strategy_id}-${t.side}`}>
                    <StockCell r={t} />
                    <TableCell className="align-top">
                      <Side side={t.side} />
                    </TableCell>
                    <StateCells r={t} />
                    <TableCell className="text-right align-top tabular-nums">{price(t.close)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-sm text-muted-foreground">這天沒有新觸發的訊號。</p>
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
                  <StockHeads />
                  <TableHead>訊號</TableHead>
                  <TableHead className="text-right">已持續</TableHead>
                  <StateHeads />
                </TableRow>
              </TableHeader>
              <TableBody>
                {r.ongoing.map((t) => (
                  <TableRow key={`${t.stock_id}-${t.strategy_id}-${t.side}`}>
                    <StockCell r={t} />
                    <TableCell className="align-top">
                      <Side side={t.side} />
                    </TableCell>
                    <TableCell className="text-right align-top tabular-nums">{t.days} 天</TableCell>
                    <StateCells r={t} />
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-sm text-muted-foreground">沒有持續成立中的條件。</p>
          )}
        </CardContent>
      </Card>
      {r.triggered.length || r.ongoing.length ? <p className="text-xs text-muted-foreground">{STATE_NOTE}</p> : null}

      {r.has_forming ? (
        <Card>
          <CardHeader>
            <CardTitle>③ 形成中</CardTitle>
            <CardDescription>
              策略寫的「形成中」條件在這天成立、還沒觸發買進。例如 Darvas
              箱型：箱子已經成形，收盤還在箱頂和箱底之間，等突破箱頂。
            </CardDescription>
          </CardHeader>
          <CardContent>
            {r.forming.length ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>股票</TableHead>
                    <TableHead>策略／這天的數值</TableHead>
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
