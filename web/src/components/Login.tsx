import { useEffect, useRef, useState } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ErrorBox } from "@/components/common"
import { api, setAccessToken, type TokenResponse } from "@/lib/api"

const GSI = "https://accounts.google.com/gsi/client"

function loadGsi(): Promise<void> {
  if (window.google?.accounts) return Promise.resolve()
  return new Promise((resolve, reject) => {
    const el = document.createElement("script")
    el.src = GSI
    el.async = true
    el.onload = () => resolve()
    el.onerror = () => reject(new Error("無法載入 Google 登入，請確認網路連線"))
    document.head.appendChild(el)
  })
}

/** Google 登入：只用用戶端 ID，拿到的 ID token 交給後端驗證。 */
export function Login() {
  const qc = useQueryClient()
  const button = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<unknown>(null)
  const config = useQuery({ queryKey: ["config"], queryFn: () => api<{ google_client_id: string }>("/api/config") })

  useEffect(() => {
    const clientId = config.data?.google_client_id
    if (!clientId) return
    loadGsi()
      .then(() => {
        const id = window.google!.accounts.id
        id.initialize({
          client_id: clientId,
          ux_mode: "popup",
          callback: async ({ credential }) => {
            try {
              setError(null)
              const tokens = await api<TokenResponse>("/api/auth/google", {
                method: "POST",
                body: JSON.stringify({ credential }),
              })
              setAccessToken(tokens.access_token)
              await qc.invalidateQueries()
            } catch (e) {
              setError(e)
            }
          },
        })
        if (button.current) id.renderButton(button.current, { theme: "outline", size: "large", locale: "zh-TW" })
      })
      .catch(setError)
  }, [config.data, qc])

  return (
    <div className="flex min-h-svh items-center justify-center p-4">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle className="text-xl">hermes 台股分析</CardTitle>
          <CardDescription>用 Google 帳號登入；只有允許名單裡的帳號能看。</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div ref={button} className="min-h-11" />
          {config.error ? <ErrorBox error={config.error} /> : null}
          {error ? <ErrorBox error={error} /> : null}
        </CardContent>
      </Card>
    </div>
  )
}

export async function logout(qc: ReturnType<typeof useQueryClient>) {
  try {
    await api("/api/auth/logout", { method: "POST" })
  } finally {
    setAccessToken(null)
  }
  window.google?.accounts.id.disableAutoSelect()
  // 清掉看過的資料，重新確認登入狀態（會回到登入頁）
  qc.removeQueries({ predicate: (q) => q.queryKey[0] !== "me" })
  await qc.resetQueries({ queryKey: ["me"] })
}
