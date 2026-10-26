import { Suspense, lazy } from "react"
import { BrowserRouter, Navigate, Route, Routes } from "react-router"
import { useQuery } from "@tanstack/react-query"
import { ErrorBox, Loading } from "@/components/common"
import { Layout } from "@/components/Layout"
import { Login } from "@/components/Login"
import { useDark } from "@/hooks/useDark"
import { ApiError, api, hasAccessToken, refreshAccessToken } from "@/lib/api"
import { ScreenPage } from "@/pages/Screen"
import { StrategiesPage } from "@/pages/Strategies"
import { WatchlistPage } from "@/pages/Watchlist"

// 個股頁面有圖表套件，分開載入，登入頁和選股報告比較快
const StockPage = lazy(() => import("@/pages/Stock").then((m) => ({ default: m.StockPage })))

export function App() {
  const dark = useDark()
  // 打開頁面時記憶體裡沒有 access token：先用 refresh token（cookie）換，換不到就是還沒登入
  const me = useQuery({
    queryKey: ["me"],
    queryFn: async () => {
      if (!hasAccessToken() && !(await refreshAccessToken())) throw new ApiError(401, "請先登入")
      return api<{ email: string }>("/api/me")
    },
    retry: false,
  })
  if (me.isPending) {
    return (
      <div className="p-6">
        <Loading />
      </div>
    )
  }
  if (me.error) {
    if (me.error instanceof ApiError && me.error.status === 401) return <Login />
    return (
      <div className="p-6">
        <ErrorBox error={me.error} />
      </div>
    )
  }
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout email={me.data.email} />}>
          <Route index element={<ScreenPage />} />
          <Route path="watchlist" element={<WatchlistPage />} />
          <Route path="strategies" element={<StrategiesPage />} />
          <Route
            path="stocks/:id/:tab?"
            element={
              <Suspense fallback={<Loading />}>
                <StockPage dark={dark} />
              </Suspense>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
