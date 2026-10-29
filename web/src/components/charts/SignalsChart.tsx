import { useEffect, useRef, useState } from "react"
import {
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  LineStyle,
  createChart,
  createSeriesMarkers,
  createTextWatermark,
  type AutoscaleInfo,
  type SeriesMarker,
  type Time,
} from "lightweight-charts"
import type { Backtest, Num, Prices, Signals } from "@/lib/api"
import { price as priceText } from "@/lib/format"
import { LINE_COLORS, alpha, axisNumber, baseOptions, chartColors } from "@/components/charts/theme"

const PRICE_HEIGHT = 380
const PANE_HEIGHT = 200

function line(dates: string[], values: Num[]) {
  return dates.map((d, i) => {
    const v = values[i]
    return typeof v === "number" ? { time: d as Time, value: v } : { time: d as Time }
  })
}

/**
 * 實際交易的訊號日；最新一天收盤後的訊號隔天開盤才成交，還不在交易明細裡，也標出來。
 * 只用箭頭（紅色向上買進、綠色向下賣出），不加文字：訊號靠很近時文字會疊在一起。
 */
function tradeMarkers(b: Backtest, up: string, down: string): SeriesMarker<Time>[] {
  const markers: SeriesMarker<Time>[] = []
  for (const t of b.trades) {
    markers.push({ time: t.buy_signal, position: "belowBar", shape: "arrowUp", color: up })
    if (t.sell_signal) markers.push({ time: t.sell_signal, position: "aboveBar", shape: "arrowDown", color: down })
  }
  const marked = new Set(b.trades.flatMap((t) => [`${t.buy_signal}|buy`, `${t.sell_signal ?? ""}|sell`]))
  const action = b.latest?.action
  if (b.latest && action && !marked.has(`${b.latest.date}|${action}`))
    markers.push(
      action === "buy"
        ? { time: b.latest.date, position: "belowBar", shape: "arrowUp", color: up }
        : { time: b.latest.date, position: "aboveBar", shape: "arrowDown", color: down },
    )
  return markers.sort((x, y) => (x.time < y.time ? -1 : x.time > y.time ? 1 : 0))
}

// 價格欄位的名稱；指標的白話名稱由後端給（column_labels）
const COLUMN_LABELS: Record<string, string> = { close: "收盤價", volume_lots: "成交量（張）" }

/**
 * 一個策略在圖上分成幾格，每一格的 key 對應門檻、參考線的鍵：
 * 跟股價同單位的指標（均線、SAR…）和收盤價一格；有成交量門檻時成交量一格；其他指標、籌碼各自一格。
 * 什麼指標都沒有的策略也給收盤價一格，買賣點才有地方標。
 */
function strategyPanes(s: Signals): { key: string; columns: string[] }[] {
  const { overlays, panels, thresholds, reference_lines } = s.layout
  const others = Object.entries(panels).map(([key, columns]) => ({ key, columns }))
  const panes: { key: string; columns: string[] }[] = []
  if (overlays.length || thresholds.price || reference_lines.price || (!others.length && !thresholds.volume))
    panes.push({ key: "price", columns: ["close", ...overlays] })
  if (thresholds.volume) panes.push({ key: "volume", columns: ["volume_lots"] })
  return [...panes, ...others]
}

function signalsPaneCount(list: Signals[]): number {
  return list.reduce((n, s) => n + strategyPanes(s).length, 0)
}

/** 圖表給的時間可能是字串、年月日或秒數，都換成 YYYY-MM-DD。 */
function day(t: Time): string {
  if (typeof t === "string") return t
  if (typeof t === "number") return new Date(t * 1000).toISOString().slice(0, 10)
  return `${t.year}-${String(t.month).padStart(2, "0")}-${String(t.day).padStart(2, "0")}`
}

/**
 * 圖上方的日期列，捲動時貼在頁首下面：圖很高的時候日期軸在最下面看不到。
 * 游標指著圖時是那天的日期和開高低收；沒有指著圖時是畫面裡的日期範圍（拖動、縮放時跟著變）。
 */
function DateBar({
  price,
  hover,
  range,
}: {
  price: Prices["series"]
  hover: number | null
  range: [string, string] | null
}) {
  const cell = (label: string, v: Num) => (
    <span>
      <span className="text-muted-foreground">{label} </span>
      <span className="tabular-nums">{priceText(v)}</span>
    </span>
  )
  return (
    <div
      className="sticky z-10 flex min-h-8 flex-wrap items-center gap-x-4 gap-y-0.5 rounded-md bg-card/95 px-2 py-1 text-sm backdrop-blur"
      style={{ top: "var(--header-height, 0px)" }}
    >
      {hover !== null ? (
        <>
          <b className="tabular-nums">{price.date[hover]}</b>
          {cell("開", price.open[hover])}
          {cell("高", price.high[hover])}
          {cell("低", price.low[hover])}
          {cell("收", price.close[hover])}
        </>
      ) : range ? (
        <span className="text-muted-foreground">
          畫面裡是 <span className="text-foreground tabular-nums">{range[0]}</span> ～{" "}
          <span className="text-foreground tabular-nums">{range[1]}</span>；游標移到圖上看那天的日期和價格
        </span>
      ) : null}
    </div>
  )
}

/**
 * 歷史買賣點的圖：最上面是股價（K 線、成交量），下面每個策略一段，畫策略用到的指標數值；
 * 實線是買賣條件裡的門檻，虛線是策略設定檔寫的參考線。買賣點標在每個策略的第一格。
 * 全部共用一條時間軸和十字線，一次看完。沒有選策略時只畫股價。
 */
export function SignalsChart({
  price,
  strategies,
  backtests,
  dark,
  from,
  to,
}: {
  price: Prices["series"]
  strategies: Signals[]
  backtests: Backtest[]
  dark: boolean
  from: string
  to: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const paneCount = signalsPaneCount(strategies)
  const [hover, setHover] = useState<number | null>(null)
  const [range, setRange] = useState<[string, string] | null>(null)
  useEffect(() => {
    if (!ref.current) return
    const c = chartColors(dark)
    const dates = price.date
    const chart = createChart(ref.current, baseOptions(dark))
    const position = new Map(dates.map((d, i) => [d, i]))
    chart.subscribeCrosshairMove((param) =>
      setHover(param.time === undefined ? null : (position.get(day(param.time)) ?? null)),
    )
    chart.timeScale().subscribeVisibleTimeRangeChange((r) => setRange(r ? [day(r.from), day(r.to)] : null))

    const volume = chart.addSeries(HistogramSeries, {
      priceScaleId: "volume",
      priceFormat: { type: "volume" },
      lastValueVisible: false,
      priceLineVisible: false,
      title: "成交量（張）",
    })
    volume.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } })
    volume.setData(
      dates.map((d, i) => {
        const v = price.volume_lots[i]
        const up = (price.close[i] as number) >= (price.open[i] as number)
        return typeof v === "number" ? { time: d, value: v, color: alpha(up ? c.up : c.down, 0.35) } : { time: d }
      }),
    )
    const candles = chart.addSeries(CandlestickSeries, {
      upColor: c.up,
      downColor: c.down,
      borderUpColor: c.up,
      borderDownColor: c.down,
      wickUpColor: c.up,
      wickDownColor: c.down,
      priceLineVisible: false,
      priceFormat: { type: "custom", formatter: axisNumber, minMove: 0.01 },
    })
    candles.priceScale().applyOptions({ scaleMargins: { top: 0.05, bottom: 0.22 } })
    candles.setData(
      dates.map((d, i) => ({
        time: d,
        open: price.open[i] as number,
        high: price.high[i] as number,
        low: price.low[i] as number,
        close: price.close[i] as number,
      })),
    )

    let pane = 0
    for (const s of strategies) {
      const { thresholds, reference_lines } = s.layout
      const b = backtests.find((x) => x.strategy.id === s.strategy.id)
      strategyPanes(s).forEach(({ key, columns }, p) => {
        pane += 1
        const levels = [...(thresholds[key] ?? []), ...(reference_lines[key] ?? [])]
        // 指標的格子把門檻、參考線也算進刻度範圍，才一定看得到；價格那格不算，免得門檻離股價太遠把線壓扁
        const autoscale =
          key === "price" || !levels.length
            ? undefined
            : (original: () => AutoscaleInfo | null): AutoscaleInfo | null => {
                const r = original()
                if (!r?.priceRange) return r
                return {
                  ...r,
                  priceRange: {
                    minValue: Math.min(r.priceRange.minValue, ...levels),
                    maxValue: Math.max(r.priceRange.maxValue, ...levels),
                  },
                }
              }
        let color = 0
        const lines = columns
          .filter((col) => col in s.series)
          .map((col) => {
            const isClose = col === "close"
            const lineColor = isClose ? c.bench : LINE_COLORS[color++ % LINE_COLORS.length]
            // 線的名稱寫在格子左上角，不用 title：title 會畫在圖的右側，蓋住最近幾天的買賣點
            const series = chart.addSeries(
              LineSeries,
              {
                color: lineColor,
                lineWidth: isClose ? 1 : 2,
                priceLineVisible: false,
                priceFormat: { type: "custom", formatter: axisNumber, minMove: 0.0001 },
                autoscaleInfoProvider: autoscale,
              },
              pane,
            )
            series.setData(line(s.series.date, s.series[col] as Num[]))
            return { series, label: COLUMN_LABELS[col] ?? s.column_labels[col] ?? col, color: lineColor }
          })
        const host = lines[0]?.series
        if (!host) return
        createTextWatermark(chart.panes()[pane], {
          horzAlign: "left",
          vertAlign: "top",
          lines: [
            // 每個策略的第一格先寫策略名稱，看得出這一段是哪個策略
            ...(p === 0 ? [{ text: s.strategy.name, color: c.text, fontSize: 12 }] : []),
            ...lines.map((l) => ({ text: l.label, color: l.color, fontSize: 12 })),
          ],
        })
        for (const v of thresholds[key] ?? [])
          host.createPriceLine({ price: v, color: c.warn, lineWidth: 1, lineStyle: LineStyle.Solid })
        for (const v of reference_lines[key] ?? [])
          host.createPriceLine({ price: v, color: c.bench, lineWidth: 1, lineStyle: LineStyle.Dashed })
        if (p === 0 && b) createSeriesMarkers(host, tradeMarkers(b, c.up, c.down))
        chart.priceScale("right", pane).applyOptions({ scaleMargins: { top: 0.12, bottom: 0.08 } })
      })
    }
    const panes = chart.panes()
    panes[0]?.setStretchFactor(PRICE_HEIGHT)
    panes.slice(1).forEach((p) => p.setStretchFactor(PANE_HEIGHT))

    chart.timeScale().setVisibleRange({ from: from as Time, to: to as Time })
    return () => chart.remove()
  }, [price, strategies, backtests, dark, from, to])

  return (
    <>
      <DateBar price={price} hover={hover} range={range} />
      <div ref={ref} style={{ height: PRICE_HEIGHT + paneCount * PANE_HEIGHT }} className="w-full" />
    </>
  )
}
