/** 主动学习 API */
import http from './http'
import type { JobItem } from './tasks'

export interface AlSession {
  id: string
  status: string
  model_id: number
  model_name?: string
  task_type: string
  target_dataset_id: number
  target_dataset_name?: string
  staging_dataset_id: number
  staging_dataset_name?: string
  job_id?: number | null
  retrain_task_id?: number | null
  summary?: {
    total?: number
    easy_count?: number
    hard_count?: number
    empty_count?: number
    written_labels?: number
  }
  items?: Array<{
    image_name: string
    difficulty: string
    score: number
    max_conf?: number | null
    reason?: string
  }>
  merged_count?: number
  finetune_weight?: string
  error?: string | null
}

export interface AlHint {
  model_id: number
  model_name?: string
  task_type: string
  supported: boolean
  has_pt: boolean
  can_start: boolean
  default_target_dataset_id: number | null
  default_target_dataset_name: string | null
  target_image_count?: number | null
  target_class_count?: number | null
  message: string | null
}

export function getAlHint(modelId: number) {
  return http.get<AlHint>(`/active-learn/models/${modelId}/hint`)
}

/** 原数据集由服务端按模型血缘锁定，无需/不可传其它集 */
export function createAlSession(modelId: number) {
  return http.post<{ session: AlSession }>('/active-learn/sessions', {
    model_id: modelId,
  })
}

export function getAlSession(sessionId: string) {
  return http.get<{ session: AlSession }>(`/active-learn/sessions/${sessionId}`)
}

export function startAlScreen(sessionId: string) {
  return http.post<{ job: JobItem; session: AlSession }>(`/active-learn/sessions/${sessionId}/screen`)
}

export function mergeAlSession(sessionId: string) {
  return http.post<{ merged: number; session: AlSession; message: string }>(
    `/active-learn/sessions/${sessionId}/merge`,
  )
}

/** 取消本轮并删除临时数据集 */
export function abandonAlSession(sessionId: string) {
  return http.post<{ session: AlSession; message: string }>(
    `/active-learn/sessions/${sessionId}/abandon`,
  )
}

/** 清理悬空的主动学习临时集 */
export function cleanupAlOrphans() {
  return http.post<{ removed: number[]; count: number }>('/active-learn/cleanup-orphans')
}

export function prepareAlRetrain(
  sessionId: string,
  body?: { epochs?: number; batch?: number; device?: string; imgsz?: number },
) {
  return http.post<{
    task_id: number
    pretrained_weight: string
    dataset_id: number
    session: AlSession
    message: string
  }>(`/active-learn/sessions/${sessionId}/prepare-retrain`, body || {})
}
