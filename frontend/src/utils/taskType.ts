/** 任务类型展示文案（detect / segment / pose）。 */

export type AppTaskType = 'detect' | 'segment' | 'pose'

const LABELS: Record<string, string> = {
  detect: '目标检测',
  segment: '实例分割',
  pose: '姿态估计',
}

export function normalizeAppTaskType(taskType?: string | null): AppTaskType {
  const t = (taskType || 'detect').trim().toLowerCase()
  if (t === 'segment' || t === 'pose') return t
  return 'detect'
}

export function taskTypeLabel(taskType?: string | null): string {
  const t = normalizeAppTaskType(taskType)
  return LABELS[t] || t
}
