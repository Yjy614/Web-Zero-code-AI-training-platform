/** 登录态仅存会话级存储：关闭浏览器后需重新登录。 */

const TOKEN_KEY = 'access_token'
const USER_KEY = 'user_info'

/** 清除历史上误用 localStorage 持久化的登录态 */
export function clearLegacyPersistentAuth() {
  try {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  } catch {
    // ignore
  }
}

export function getAuthToken(): string | null {
  try {
    return sessionStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function setAuthToken(token: string) {
  sessionStorage.setItem(TOKEN_KEY, token)
}

export function getAuthUserRaw(): string | null {
  try {
    return sessionStorage.getItem(USER_KEY)
  } catch {
    return null
  }
}

export function setAuthUserRaw(json: string) {
  sessionStorage.setItem(USER_KEY, json)
}

export function clearAuthSession() {
  try {
    sessionStorage.removeItem(TOKEN_KEY)
    sessionStorage.removeItem(USER_KEY)
  } catch {
    // ignore
  }
  clearLegacyPersistentAuth()
}
