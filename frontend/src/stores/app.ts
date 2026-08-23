/** 全局应用状态（演示模式等）。 */
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { fetchSystemInfo, type SystemInfo } from '@/api/system'

export const useAppStore = defineStore('app', () => {
  const systemInfo = ref<SystemInfo | null>(null)
  const demoMode = ref(true)

  async function loadSystemInfo() {
    const { data } = await fetchSystemInfo()
    systemInfo.value = data
    demoMode.value = data.demo_mode
  }

  return { systemInfo, demoMode, loadSystemInfo }
})
