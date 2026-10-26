import { Link, useNavigate, useParams, useSearchParams } from "react-router"
import { useQuery } from "@tanstack/react-query"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ErrorBox, Loading } from "@/components/common"
import { api, type StockInfo } from "@/lib/api"
import { price } from "@/lib/format"
import { BacktestTab } from "@/pages/stock/BacktestTab"
import { ChipsTab } from "@/pages/stock/ChipsTab"
import { CompareTab } from "@/pages/stock/CompareTab"
import { SignalsTab } from "@/pages/stock/SignalsTab"
import { TechTab } from "@/pages/stock/TechTab"

const TABS = [
  ["signals", "歷史買賣點"],
  ["tech", "技術指標"],
  ["chips", "籌碼"],
  ["backtest", "回測"],
  ["compare", "策略比較"],
] as const

export function StockPage({ dark }: { dark: boolean }) {
  const { id = "", tab = "signals" } = useParams()
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const q = useQuery({
    queryKey: ["stock", id],
    queryFn: () => api<StockInfo>(`/api/stocks/${encodeURIComponent(id)}`),
  })
  if (q.isPending) return <Loading />
  if (q.error) return <ErrorBox error={q.error} />
  const stock = q.data
  // 換分頁時只帶著選的策略，其他設定（區間、成本）各分頁分開；回測只用第一個
  const go = (t: string) => {
    const strategies = new URLSearchParams(params.getAll("strategy").map((s) => ["strategy", s])).toString()
    void navigate(`/stocks/${id}/${t}${strategies ? `?${strategies}` : ""}`)
  }
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h1 className="text-2xl font-semibold">
          {stock.stock_id} {stock.name}
        </h1>
        <span className="text-sm text-muted-foreground">
          {stock.last_date} 收盤 <b className="text-foreground tabular-nums">{price(stock.close)}</b>　本機資料{" "}
          {stock.first_date} ～ {stock.last_date}
        </span>
        {!stock.in_watchlist ? (
          <Badge variant="outline">
            不在
            <Link to="/watchlist" className="underline">
              自選股
            </Link>
          </Badge>
        ) : null}
      </div>
      <Tabs value={tab} onValueChange={go}>
        <TabsList className="flex h-auto flex-wrap">
          {TABS.map(([k, label]) => (
            <TabsTrigger key={k} value={k}>
              {label}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
      {tab === "signals" ? <SignalsTab stock={stock} dark={dark} /> : null}
      {tab === "tech" ? <TechTab stock={stock} /> : null}
      {tab === "chips" ? <ChipsTab stock={stock} /> : null}
      {tab === "backtest" ? <BacktestTab stock={stock} dark={dark} /> : null}
      {tab === "compare" ? <CompareTab stock={stock} dark={dark} /> : null}
    </div>
  )
}
