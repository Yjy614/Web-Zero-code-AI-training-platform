/** 登录态与用户信息。 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { loginApi, logoutApi, meApi, type UserInfo } from '@/api/auth'
import {
  clearAuthSession,
  clearLegacyPersistentAuth,
  getAuthToken,
  getAuthUserRaw,
  setAuthToken,
  setAuthUserRaw,
} from '@/utils/authStorage'
import { clearDetectWizardState } from '@/utils/wizardSession'

// 启动时清掉旧版 localStorage 登录，避免关浏览器后仍免密进入
clearLegacyPersistentAuth()

export const useAuthStore = defineStore('auth', () => {
  const token = ref<string | null>(getAuthToken())
  const user = ref<UserInfo | null>(readUser())

  const isLoggedIn = computed(() => Boolean(token.value))
  const isAdmin = computed(() => user.value?.role === 'admin')

  function readUser(): UserInfo | null {
    const raw = getAuthUserRaw()
    if (!raw) return null
    try {
      return JSON.parse(raw) as UserInfo
    } catch {
      return null
    }
  }

  async function login(username: string, password: string) {
    // 新登录从干净向导开始，不沿用上一会话进度
    clearDetectWizardState()
    const { data } = await loginApi(username, password)
    token.value = data.access_token
    user.value = data.user
    setAuthToken(data.access_token)
    setAuthUserRaw(JSON.stringify(data.user))
  }

  async function fetchMe() {
    if (!token.value) return
    const { data } = await meApi()
    user.value = data
    setAuthUserRaw(JSON.stringify(data))
  }

  async function logout() {
    try {
      if (token.value) await logoutApi()
    } catch {
      // 忽略登出接口失败，本地仍清理
    }
    token.value = null
    user.value = null
    clearAuthSession()
    clearDetectWizardState()
  }

  return { token, user, isLoggedIn, isAdmin, login, fetchMe, logout }
})
