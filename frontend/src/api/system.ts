/** 系统信息与设置 API。 */
import http from './http'

export interface SystemInfo {
  api_version: string
  demo_mode: boolean
  job_runner: string
  storage_configured: boolean
  forbid_weight_download: boolean
}

export interface ModelEndpointConfig {
  base_url: string
  api_key: string
  model: string
  timeout: number
  api_key_set?: boolean
}

export interface AppSettings {
  demo_mode: boolean
  job_runner: string
  forbid_weight_download: boolean
  free_step_nav: boolean
  llm: ModelEndpointConfig
  vision: ModelEndpointConfig
}

export function fetchSystemInfo() {
  return http.get<SystemInfo>('/system/info')
}

export interface DeviceItem {
  id: string
  label: string
}

export function fetchDevices() {
  return http.get<{ devices: DeviceItem[]; cuda_available: boolean; gpu_count: number }>(
    '/system/devices',
  )
}

export function fetchSettings() {
  return http.get<AppSettings>('/settings')
}

export function updateSettings(payload: {
  demo_mode?: boolean
  llm?: Partial<ModelEndpointConfig>
  vision?: Partial<ModelEndpointConfig>
}) {
  return http.put<AppSettings>('/settings', payload)
}
