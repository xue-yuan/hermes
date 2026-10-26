import { Link } from "react-router"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ErrorBox, Loading } from "@/components/common"
import { useOverview } from "@/components/Layout"

export function WatchlistPage() {
  const q = useOverview()
  if (q.isPending) return <Loading />
  if (q.error) return <ErrorBox error={q.error} />
  const items = q.data.watchlist
  return (
    <Card>
      <CardHeader>
        <CardTitle>自選股（{items.length} 檔）</CardTitle>
        <CardDescription>
          網頁只能看。加入、移除請用 CLI：hermes watchlist add／remove，再執行 hermes update 下載資料。
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>代號</TableHead>
              <TableHead>名稱</TableHead>
              <TableHead>加入日期</TableHead>
              <TableHead>本機股價最新日</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((w) => (
              <TableRow key={w.stock_id}>
                <TableCell>
                  <Link className="font-medium underline-offset-4 hover:underline" to={`/stocks/${w.stock_id}`}>
                    {w.stock_id}
                  </Link>
                </TableCell>
                <TableCell>{w.name}</TableCell>
                <TableCell>{w.added_at}</TableCell>
                <TableCell>{w.last_price_date ?? <span className="text-warn">尚未下載</span>}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
