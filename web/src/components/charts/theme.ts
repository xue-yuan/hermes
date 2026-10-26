import { ColorType, CrosshairMode, type DeepPartial, type ChartOptions } from "lightweight-charts"

// 圖表套件不認得 oklch，顏色用十六進位；紅漲綠跌（台股慣例）
export function chartColors(dark: boolean) {
  return dark
    ? {
        bg: "#171717",
        text: "#d4d4d4",
        grid: "#262626",
        border: "#404040",
        up: "#f05252",
        down: "#31c48d",
        warn: "#facc15",
        bench: "#a3a3a3",
      }
    : {
        bg: "#ffffff",
        text: "#404040",
        grid: "#f0f0f0",
        border: "#d4d4d4",
        up: "#d62728",
        down: "#16a34a",
        warn: "#ca8a04",
        bench: "#404040",
      }
}

export const LINE_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]

export function alpha(hex: string, a: number): string {
  const v = parseInt(hex.slice(1), 16)
  return `rgba(${(v >> 16) & 255}, ${(v >> 8) & 255}, ${v & 255}, ${a})`
}

export function baseOptions(dark: boolean): DeepPartial<ChartOptions> {
  const c = chartColors(dark)
  return {
    autoSize: true,
    layout: {
      background: { type: ColorType.Solid, color: c.bg },
      textColor: c.text,
      fontFamily: 'ui-sans-serif, system-ui, -apple-system, "PingFang TC", "Noto Sans TC", sans-serif',
      panes: { separatorColor: c.border, separatorHoverColor: alpha(c.border, 0.5) },
      attributionLogo: true,
    },
    grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
    crosshair: { mode: CrosshairMode.Normal },
    rightPriceScale: { borderColor: c.border },
    timeScale: { borderColor: c.border },
    localization: { locale: "zh-TW", dateFormat: "yyyy-MM-dd" },
  }
}
