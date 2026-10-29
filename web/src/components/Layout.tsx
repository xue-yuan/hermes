import { useEffect, useRef, useState, type FormEvent } from "react"
import { Link, NavLink, Outlet, useNavigate } from "react-router"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { LogOut } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { ADJUSTED_HELP, Disclaimer } from "@/components/common"
import { setAdjusted, useAdjusted } from "@/hooks/useAdjusted"
import { logout } from "@/components/Login"
import { api, type Overview } from "@/lib/api"
import { cn } from "@/lib/utils"

const NAV = [
  ["/", "每日選股"],
  ["/watchlist", "自選股"],
  ["/strategies", "策略與指標"],
] as const

export function useOverview() {
  return useQuery({ queryKey: ["overview"], queryFn: () => api<Overview>("/api/overview") })
}

function StockSearch() {
  const navigate = useNavigate()
  const overview = useOverview()
  const [value, setValue] = useState("")
  const submit = (e: FormEvent) => {
    e.preventDefault()
    const id = value.trim().split(/\s/)[0]
    if (id) void navigate(`/stocks/${encodeURIComponent(id)}`)
    setValue("")
  }
  return (
    <form onSubmit={submit} className="w-full sm:w-48">
      <Input
        list="watchlist-ids"
        placeholder="股票代號，例如 2330"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        aria-label="股票代號"
      />
      <datalist id="watchlist-ids">
        {overview.data?.watchlist.map((w) => (
          <option key={w.stock_id} value={w.stock_id}>
            {w.name}
          </option>
        ))}
      </datalist>
    </form>
  )
}

// 開關選中的那一邊要一眼看得出來
const SWITCH_ON =
  "data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:hover:bg-primary data-[state=on]:hover:text-primary-foreground"

/** 還原模式的開關：指標、買賣判斷、圖改用還原價格算。每一頁都適用。 */
function PriceSwitch() {
  const adjusted = useAdjusted()
  return (
    <div className="flex items-center gap-2 text-sm" title={ADJUSTED_HELP}>
      <span className="text-muted-foreground">判斷用</span>
      <ToggleGroup
        type="single"
        size="sm"
        variant="outline"
        value={adjusted ? "adjusted" : "raw"}
        onValueChange={(v) => v && setAdjusted(v === "adjusted")}
        aria-label="判斷用的價格"
      >
        <ToggleGroupItem value="raw" className={SWITCH_ON}>
          實際成交價
        </ToggleGroupItem>
        <ToggleGroupItem value="adjusted" className={SWITCH_ON}>
          還原價格
        </ToggleGroupItem>
      </ToggleGroup>
    </div>
  )
}

/** 頁首的高度（手機上會換行，高度會變），給要貼在頁首下面的東西用（例如買賣點圖的日期列）。 */
function useHeaderHeight() {
  const ref = useRef<HTMLElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new ResizeObserver(() =>
      document.documentElement.style.setProperty("--header-height", `${el.offsetHeight}px`),
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])
  return ref
}

export function Layout({ email }: { email: string }) {
  const qc = useQueryClient()
  const header = useHeaderHeight()
  return (
    <div className="min-h-svh">
      <header ref={header} className="sticky top-0 z-20 border-b bg-background/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2">
          <Link to="/">
            <span className="font-semibold">Hermes</span>
          </Link>
          <nav className="flex gap-1 text-sm">
            {NAV.map(([to, label]) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  cn("rounded-md px-3 py-1.5 hover:bg-accent", isActive && "bg-accent font-medium")
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex w-full flex-wrap items-center gap-2 sm:w-auto">
            <PriceSwitch />
            {/* 手機上開關一行，搜尋框和登出一行 */}
            <div className="flex flex-1 items-center gap-2 sm:flex-none">
              <StockSearch />
              <span className="hidden text-xs text-muted-foreground lg:inline">{email}</span>
              <Button variant="ghost" size="icon" onClick={() => logout(qc)} title="登出" aria-label="登出">
                <LogOut />
              </Button>
            </div>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl space-y-4 px-4 py-6">
        <Outlet />
        <Disclaimer />
      </main>
    </div>
  )
}
