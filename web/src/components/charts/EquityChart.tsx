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

    curves.series.forEach((s, k) => {
      const color = LINE_COLORS[k % LINE_COLORS.length]
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
        .addSeries(LineSeries, { color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false }, 2)
        .setData(line(date, s.position))
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
