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
      menuVisibility.value = { ...DEFAULT_MENU_VISIBILITY, ...(data.menu_visibility || {}) }
    } catch {
      menuVisibility.value = { ...DEFAULT_MENU_VISIBILITY }
    }
  }

  function setMenuVisibility(vis: MenuVisibility) {
    menuVisibility.value = { ...DEFAULT_MENU_VISIBILITY, ...vis }
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
