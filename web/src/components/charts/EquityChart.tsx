import { useEffect, useRef } from "react"
import { LineSeries, PriceScaleMode, createChart, type Time } from "lightweight-charts"
import type { Curves, Num } from "@/lib/api"
import { LINE_COLORS, baseOptions, chartColors } from "@/components/charts/theme"

function line(dates: string[], values: Num[], scale = 1) {
  return dates.map((d, i) => {
    const v = values[i]
    return typeof v === "number" ? { time: d as Time, value: v * scale } : { time: d as Time }
  })
}

/** 回測：淨值（對數刻度）、回撤（%）、部位；和 CLI 的回測圖相同的三段。 */
export function EquityChart({ curves, dark }: { curves: Curves; dark: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!ref.current) return
    const c = chartColors(dark)
    const chart = createChart(ref.current, baseOptions(dark))
    const { date } = curves
    const pct = { type: "custom" as const, formatter: (v: number) => `${v.toFixed(1)}%`, minMove: 0.01 }

    const bench = chart.addSeries(LineSeries, {
      color: c.bench,
      lineWidth: 2,
      title: "買進持有",
      priceLineVisible: false,
    })
    bench.setData(line(date, curves.benchmark.equity))
    chart
      .addSeries(
        LineSeries,
        { color: c.bench, lineWidth: 1, priceFormat: pct, priceLineVisible: false, lastValueVisible: false },
        1,
      )
      .setData(line(date, curves.benchmark.drawdown, 100))

    // 多個策略時部位一個策略一條（由上到下和表格同順序），0、1 的線疊在一起看不出是誰
    const lanes = curves.series.length > 1
    curves.series.forEach((s, k) => {
      const color = LINE_COLORS[k % LINE_COLORS.length]
      const lane = curves.series.length - 1 - k
      const position = lanes ? s.position.map((v) => (typeof v === "number" ? lane + 0.8 * v : v)) : s.position
      chart
        .addSeries(LineSeries, { color, lineWidth: 2, title: s.name, priceLineVisible: false })
        .setData(line(date, s.equity))
      chart
        .addSeries(
          LineSeries,
          { color, lineWidth: 1, priceFormat: pct, priceLineVisible: false, lastValueVisible: false },
          1,
        )
        .setData(line(date, s.drawdown, 100))
      chart
        .addSeries(
          LineSeries,
          {
            color,
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: !lanes,
            // 分開畫的時候刻度的數字沒有意義，不顯示（把刻度整個藏起來的話，圖會畫不出來）
            ...(lanes ? { priceFormat: { type: "custom" as const, formatter: () => "", minMove: 0.01 } } : {}),
          },
          2,
        )
        .setData(line(date, position))
    })
    // 只有淨值用對數刻度；回撤是負數、部位是 0～1，要用一般刻度
    chart.priceScale("right", 0).applyOptions({ mode: PriceScaleMode.Logarithmic })
    chart.priceScale("right", 1).applyOptions({ mode: PriceScaleMode.Normal })
    chart.priceScale("right", 2).applyOptions({ mode: PriceScaleMode.Normal })
    const panes = chart.panes()
    panes[0]?.setStretchFactor(3)
    panes[1]?.setStretchFactor(1.2)
    panes[2]?.setStretchFactor(0.8)
    chart.timeScale().fitContent()
    return () => chart.remove()
  }, [curves, dark])
  return <div ref={ref} className="h-[560px] w-full" />
}
