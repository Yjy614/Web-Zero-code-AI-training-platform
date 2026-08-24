/** Axios 封装：统一鉴权与错误中文提示。 */
import axios from 'axios'
import { ElMessage } from 'element-plus'
import { clearAuthSession, getAuthToken } from '@/utils/authStorage'
import { clearDetectWizardState } from '@/utils/wizardSession'

const http = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
})

http.interceptors.request.use((config) => {
  const token = getAuthToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

async function extractErrorMessage(error: unknown): Promise<string> {
  const fallback = '请求失败，请稍后重试'
  const err = error as {
    response?: { status?: number; data?: unknown }
    message?: string
  }
  const data = err?.response?.data
  if (!data) return err?.message || fallback

  // blob 下载失败时，错误体也是 Blob（JSON）
  if (typeof Blob !== 'undefined' && data instanceof Blob) {
    try {
      const text = await data.text()
      const parsed = JSON.parse(text) as { detail?: { message?: string } | string; message?: string }
      if (typeof parsed.detail === 'string') return parsed.detail
      if (parsed.detail && typeof parsed.detail === 'object' && parsed.detail.message) {
        return parsed.detail.message
      }
      if (parsed.message) return parsed.message
    } catch {
      // ignore
    }
    if (err.response?.status === 404) return '文件不存在或路径错误'
    return fallback
  }

  const detail = (data as { detail?: unknown; message?: string }).detail
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object' && (detail as { message?: string }).message) {
    return String((detail as { message?: string }).message)
  }
  if ((data as { message?: string }).message) return String((data as { message?: string }).message)
  return fallback
}

http.interceptors.response.use(
  (res) => res,
  async (error) => {
    const message = await extractErrorMessage(error)
    if (error?.response?.status === 401) {
      clearAuthSession()
      clearDetectWizardState()
      if (!location.pathname.includes('/login')) {
        location.href = '/login'
      }
    } else {
      ElMessage.error(message)
    }
    return Promise.reject(error)
  },
)

export default http
