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
  /** DeepSeek 思考模式开关（仅 LLM） */
  thinking_enabled?: boolean
  /** low | high | max */
  reasoning_effort?: string
}

/** 侧栏功能入口可见性（管理员可配；全员生效） */
export interface MenuVisibility {
  detect_wizard: boolean
  segment_wizard: boolean
  pose_wizard: boolean
  datasets: boolean
  models: boolean
  infer: boolean
  agent_chat: boolean
  agent_orchestrate: boolean
  weights: boolean
  settings: boolean
}

export const DEFAULT_MENU_VISIBILITY: MenuVisibility = {
  detect_wizard: true,
  segment_wizard: true,
  pose_wizard: true,
  datasets: true,
  models: true,
  infer: true,
  agent_chat: true,
  agent_orchestrate: true,
  weights: true,
  settings: true,
}

/** 菜单 key → 路由 path */
export const MENU_PATH_BY_KEY: Record<keyof MenuVisibility, string> = {
  detect_wizard: '/app/detect/wizard',
  segment_wizard: '/app/segment/wizard',
  pose_wizard: '/app/pose/wizard',
  datasets: '/app/resources/datasets',
  models: '/app/resources/models',
  infer: '/app/resources/infer',
  agent_chat: '/app/agent/chat',
  agent_orchestrate: '/app/agent/orchestrate',
  weights: '/app/resources/weights',
  settings: '/app/settings',
}

export const MENU_LABEL_BY_KEY: Record<keyof MenuVisibility, string> = {
  detect_wizard: '目标检测训练',
  segment_wizard: '实例分割训练',
  pose_wizard: '姿态估计训练',
  datasets: '数据集管理',
  models: '模型库',
  infer: '推理试用',
  agent_chat: 'AI Agent',
  agent_orchestrate: '流程编排',
  weights: '基础模型权重仓库',
  settings: '系统设置',
}

export interface AppSettings {
  demo_mode: boolean
  job_runner: string
  forbid_weight_download: boolean
  free_step_nav: boolean
  menu_visibility: MenuVisibility
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
  menu_visibility?: Partial<MenuVisibility>
}) {
  return http.put<AppSettings>('/settings', payload)
}

/** 根据当前 path 解析菜单 key；用户管理等返回 null（不受此开关控制） */
export function menuKeyFromPath(path: string): keyof MenuVisibility | null {
  const entries = Object.entries(MENU_PATH_BY_KEY) as Array<[keyof MenuVisibility, string]>
  for (const [key, p] of entries) {
    if (path === p || path.startsWith(p + '/')) return key
  }
  return null
}

export function firstVisibleMenuPath(vis: MenuVisibility): string {
  for (const key of Object.keys(MENU_PATH_BY_KEY) as Array<keyof MenuVisibility>) {
    if (vis[key]) return MENU_PATH_BY_KEY[key]
  }
  return '/app/settings'
}
