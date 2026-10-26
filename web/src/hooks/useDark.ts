import { useEffect, useState } from "react"

/** 跟著系統的深色模式；html 加上 .dark 讓 Tailwind 和圖表一起換色。 */
export function useDark(): boolean {
  const media = typeof window !== "undefined" ? window.matchMedia("(prefers-color-scheme: dark)") : null
  const [dark, setDark] = useState(media?.matches ?? false)
  useEffect(() => {
    if (!media) return
    const onChange = (e: MediaQueryListEvent) => setDark(e.matches)
    media.addEventListener("change", onChange)
    return () => media.removeEventListener("change", onChange)
  }, [media])
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
  }, [dark])
  return dark
}
