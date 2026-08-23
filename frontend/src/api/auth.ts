/** 认证相关 API。 */
import http from './http'

export interface UserInfo {
  id: number
  username: string
  role: 'admin' | 'user' | string
}

export interface LoginResult {
  access_token: string
  token_type: string
  user: UserInfo
}

export function loginApi(username: string, password: string) {
  return http.post<LoginResult>('/auth/login', { username, password })
}

export function logoutApi() {
  return http.post('/auth/logout')
}

export function meApi() {
  return http.get<UserInfo>('/auth/me')
}
