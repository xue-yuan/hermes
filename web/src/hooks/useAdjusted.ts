import { useSyncExternalStore } from "react"

// 還原模式：指標、買賣判斷、圖改用還原價格算；每個模式只算、只顯示自己那種價格。
// 存在這個瀏覽器，換頁、重新整理、開新分頁都保留；瀏覽器不讓存時只影響這次打開的頁面。
const KEY = "hermes-adjusted"
const listeners = new Set<() => void>()

function read(): boolean {
  try {
    return localStorage.getItem(KEY) === "1"
  } catch {
    return false
  }
}

let current = read()

export function setAdjusted(on: boolean) {
  current = on
  try {
    if (on) localStorage.setItem(KEY, "1")
    else localStorage.removeItem(KEY)
  } catch {
    // 存不了：這次打開的頁面照樣切換
  }
  listeners.forEach((l) => l())
}

function subscribe(listener: () => void) {
  listeners.add(listener)
  // 別的分頁切換時跟著換
  const onStorage = (e: StorageEvent) => {
    if (e.key === KEY) {
      current = read()
      listener()
    }
  }
  window.addEventListener("storage", onStorage)
  return () => {
    listeners.delete(listener)
    window.removeEventListener("storage", onStorage)
  }
}

export function useAdjusted(): boolean {
  return useSyncExternalStore(subscribe, () => current)
}
