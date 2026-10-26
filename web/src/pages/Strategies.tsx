import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ErrorBox, Loading } from "@/components/common"
import { useOverview } from "@/components/Layout"

export function StrategiesPage() {
  const q = useOverview()
  if (q.isPending) return <Loading />
  if (q.error) return <ErrorBox error={q.error} />
  const { strategies, indicator_sets } = q.data
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold">策略</h2>
        <p className="text-sm text-muted-foreground">網頁只能看；新增或修改策略要用 CLI。</p>
      </div>
      {strategies.length === 0 ? <p className="text-sm text-muted-foreground">還沒有策略。</p> : null}
      <div className="grid gap-4 md:grid-cols-2">
        {strategies.map((s) =>
          s.error !== null ? (
            <Card key={s.id} className="border-destructive/50">
              <CardHeader>
                <CardTitle>這個策略的設定有誤</CardTitle>
                <CardDescription className="whitespace-pre-line text-destructive">{s.error}</CardDescription>
              </CardHeader>
            </Card>
          ) : (
            <Card key={s.id}>
              <CardHeader>
                <CardTitle>{s.name}</CardTitle>
                {s.description ? <CardDescription>{s.description}</CardDescription> : null}
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <p>
                  <span className="text-up">買進</span>：{s.buy}
                </p>
                <p>
                  <span className="text-down">賣出</span>：{s.sell}
                </p>
                {s.forming !== null ? (
                  <p>
                    <span className="text-muted-foreground">形成中</span>：{s.forming}
                  </p>
                ) : null}
              </CardContent>
            </Card>
          ),
        )}
      </div>
      <div>
        <h2 className="text-lg font-semibold">指標組</h2>
        <p className="text-sm text-muted-foreground">個股的「技術指標」分頁會列出數值。</p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {indicator_sets.map((s) => (
          <Card key={s.id}>
            <CardHeader>
              <CardTitle>{s.error === null ? s.name : "這個指標組的設定有誤"}</CardTitle>
              <CardDescription className={s.error === null ? "" : "whitespace-pre-line text-destructive"}>
                {s.error === null ? s.description : s.error}
              </CardDescription>
            </CardHeader>
          </Card>
        ))}
      </div>
    </div>
  )
}
