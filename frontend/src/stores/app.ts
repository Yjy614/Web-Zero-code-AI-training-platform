/** 全局应用状态（演示模式、菜单可见性等）。 */
import { defineStore } from 'pinia'
import { ref } from 'vue'
import {
  DEFAULT_MENU_VISIBILITY,
  fetchSettings,
  fetchSystemInfo,
  type MenuVisibility,
  type SystemInfo,
} from '@/api/system'

export const useAppStore = defineStore('app', () => {
  const systemInfo = ref<SystemInfo | null>(null)
  const demoMode = ref(true)
  const menuVisibility = ref<MenuVisibility>({ ...DEFAULT_MENU_VISIBILITY })

  async function loadSystemInfo() {
    const { data } = await fetchSystemInfo()
    systemInfo.value = data
    demoMode.value = data.demo_mode
  }

  async function loadMenuVisibility() {
    try {
      const { data } = await fetchSettings()
      const next = { ...DEFAULT_MENU_VISIBILITY }
      const raw = data.menu_visibility || {}
      for (const key of Object.keys(DEFAULT_MENU_VISIBILITY) as Array<keyof MenuVisibility>) {
        if (raw[key] !== undefined) next[key] = Boolean(raw[key])
      }
      menuVisibility.value = next
    } catch {
      menuVisibility.value = { ...DEFAULT_MENU_VISIBILITY }
    }
  }

  function setMenuVisibility(vis: MenuVisibility) {
    const next = { ...DEFAULT_MENU_VISIBILITY }
    for (const key of Object.keys(DEFAULT_MENU_VISIBILITY) as Array<keyof MenuVisibility>) {
      if (vis[key] !== undefined) next[key] = Boolean(vis[key])
    }
    menuVisibility.value = next
  }

  async function bootstrap() {
    await Promise.all([loadSystemInfo(), loadMenuVisibility()])
  }

  return {
    systemInfo,
    demoMode,
    menuVisibility,
    loadSystemInfo,
    loadMenuVisibility,
    setMenuVisibility,
    bootstrap,
  }
})
