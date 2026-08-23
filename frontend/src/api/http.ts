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

http.interceptors.response.use(
  (res) => res,
  (error) => {
    const detail = error?.response?.data?.detail
    let message = '请求失败，请稍后重试'
    if (typeof detail === 'string') {
      message = detail
    } else if (detail && typeof detail === 'object' && detail.message) {
      message = detail.message
    } else if (error?.response?.data?.message) {
      message = error.response.data.message
    }
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
