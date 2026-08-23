/** 用户管理 API（管理员）。 */
import http from './http'

export interface UserRow {
  id: number
  username: string
  role: 'admin' | 'user' | string
}

export function listUsers() {
  return http.get<UserRow[]>('/users')
}

export function createUser(payload: { username: string; password: string; role: string }) {
  return http.post<UserRow>('/users', payload)
}

export function updateUser(id: number, payload: { password?: string; role?: string }) {
  return http.patch<UserRow>(`/users/${id}`, payload)
}

export function resetUserPassword(id: number, password: string) {
  return http.post<UserRow>(`/users/${id}/reset-password`, { password })
}

export function deleteUser(id: number) {
  return http.delete(`/users/${id}`)
}
