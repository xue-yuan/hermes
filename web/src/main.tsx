import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { QueryCache, QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { TooltipProvider } from "@/components/ui/tooltip"
import { App } from "@/App"
import { ApiError } from "@/lib/api"
import "@/index.css"

const queryClient: QueryClient = new QueryClient({
  queryCache: new QueryCache({
    // 登入過期：重新確認登入狀態，回到登入頁（me 本身失敗時不用，不然會一直重試）
    onError: (error, q) => {
      if (error instanceof ApiError && error.status === 401 && q.queryKey[0] !== "me") {
        void queryClient.invalidateQueries({ queryKey: ["me"] })
      }
    },
  }),
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      refetchOnWindowFocus: false,
      retry: (count, error) => !(error instanceof ApiError && error.status < 500) && count < 2,
    },
  },
})

// Google 登入只認「已授權的 JavaScript 來源」裡的網址（http://localhost:8000）；用 127.0.0.1 打開時換成 localhost
if (window.location.hostname === "127.0.0.1") {
  window.location.replace(window.location.href.replace("//127.0.0.1", "//localhost"))
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <App />
      </TooltipProvider>
    </QueryClientProvider>
  </StrictMode>,
)
