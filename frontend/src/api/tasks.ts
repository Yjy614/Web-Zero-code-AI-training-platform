/** 训练任务 / Job / 资源 API。 */
import http from './http'

export interface TrainConfig {
  train_ratio: number
  val_ratio: number
  test_ratio: number
  epochs: number
  batch: number
  imgsz: number
  /** 训练策略：stable | fast | finetune */
  train_strategy: string
  device: string
  pretrained_weight: string
  augment: boolean
}

export interface TaskItem {
  id: number
  name: string
  owner_id: number
  dataset_id: number
  status: string
  step: number
  config: Partial<TrainConfig> & Record<string, unknown>
  model_path: string
  metrics: Record<string, unknown>
  created_at?: string
  updated_at?: string
}

export interface JobItem {
  id: number
  task_id: number
  type: string
  status: string
  progress: number
  message: string
  result: Record<string, unknown>
}

export interface WeightItem {
  name: string
  path: string
  size: number
  task_type?: string
}

export interface ModelItem {
  id: number
  name: string
  path: string
  task_type: string
  owner_id: number
  task_id?: number | null
  metrics: Record<string, unknown>
  created_at?: string
  has_pt?: boolean
  has_onnx?: boolean
  /** 训练血缘：原数据集与续训父模型 */
  dataset_id?: number | null
  dataset_name?: string | null
  parent_model_id?: number | null
  parent_weight?: string | null
  /** 展示用：父模型权重文件名，如 ties.pt */
  parent_weight_label?: string | null
}

export function listTasks() {
  return http.get<TaskItem[]>('/tasks')
}

export function createTask(name: string, datasetId: number) {
  return http.post<TaskItem>('/tasks', { name, dataset_id: datasetId })
}

export function getTask(id: number) {
  return http.get<TaskItem>(`/tasks/${id}`)
}

export function patchTask(id: number, payload: { step?: number; status?: string; config?: TrainConfig }) {
  return http.patch<TaskItem>(`/tasks/${id}`, payload)
}

export function splitTask(id: number) {
  return http.post(`/tasks/${id}/split`)
}

export function startTrain(id: number) {
  return http.post<JobItem>(`/tasks/${id}/train`)
}

export function startEval(id: number) {
  return http.post<JobItem>(`/tasks/${id}/eval`)
}

export interface AiAdviceResult {
  advice: string
  source: 'llm' | 'fallback' | string
  context?: Record<string, unknown>
  error?: string | null
}

/** 评估步：基于数据量与超参请求 AI 优化建议（超时单独加长）。 */
export function fetchAiAdvice(id: number) {
  return http.post<AiAdviceResult>(`/tasks/${id}/ai-advice`, null, { timeout: 120000 })
}

export function startExport(id: number, formats: string[] = ['pt', 'onnx']) {
  return http.post<JobItem>(`/tasks/${id}/export`, { formats })
}

export function getJob(id: number) {
  return http.get<JobItem>(`/jobs/${id}`)
}

/** 某任务下的 Job 列表（新到旧）；type 可选：train / eval / export */
export function listTaskJobs(taskId: number, type?: string) {
  return http.get<JobItem[]>(`/tasks/${taskId}/jobs`, {
    params: type ? { type } : undefined,
  })
}

export function cancelJob(id: number) {
  return http.post(`/jobs/${id}/cancel`)
}

export interface WeightTaskTypeItem {
  task_type: string
  label: string
  enabled: boolean
}

export function listWeightTaskTypes() {
  return http.get<{ items: WeightTaskTypeItem[] }>('/weights/task-types')
}

export function listWeights(taskType = 'detect') {
  return http.get<WeightItem[]>('/weights', { params: { task_type: taskType } })
}

export function listModels(taskType?: string) {
  return http.get<ModelItem[]>('/models', { params: taskType ? { task_type: taskType } : undefined })
}

export function deleteModel(modelId: number) {
  return http.delete<{ message: string; id: number; name: string }>(`/models/${modelId}`)
}

/** 启动仅导出 ONNX 的任务（若已存在则返回 completed）。 */
export function exportModelOnnx(modelId: number) {
  return http.post<JobItem>(`/models/${modelId}/export-onnx`)
}

export function uploadWeight(file: File, taskType = 'detect') {
  const form = new FormData()
  form.append('file', file)
  form.append('task_type', taskType)
  return http.post<WeightItem>('/weights/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export function deleteWeight(name: string, taskType = 'detect') {
  return http.delete(`/weights/${encodeURIComponent(name)}`, { params: { task_type: taskType } })
}

export interface DetectItem {
  class_id: number
  class_name: string
  confidence: number
  bbox_xyxy: number[]
  polygon?: number[][] | null
  keypoints?: number[][] | null
}

export interface PredictResult {
  task_type: string
  model_format: string
  count: number
  detections: DetectItem[]
  image_base64: string
  image_mime: string
  width: number
  height: number
  conf: number
  iou: number
  imgsz: number
}

/** 模型库推理试用：上传图片，返回可视化与检测列表。 */
export function predictModel(
  modelId: number,
  file: File,
  opts?: { conf?: number; iou?: number; imgsz?: number },
) {
  const form = new FormData()
  form.append('file', file)
  form.append('conf', String(opts?.conf ?? 0.25))
  form.append('iou', String(opts?.iou ?? 0.45))
  form.append('imgsz', String(opts?.imgsz ?? 640))
  return http.post<PredictResult>(`/models/${modelId}/predict`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 5 * 60 * 1000,
  })
}

/** 若 blob 实为接口错误 JSON 则抛出；HTML/文本报告不能当错误。 */
async function rejectIfErrorJsonBlob(blob: Blob) {
  const type = (blob.type || '').toLowerCase()
  if (!type.includes('json')) return
  let msg = '下载失败'
  try {
    const parsed = JSON.parse(await blob.text()) as {
      detail?: { message?: string } | string
      message?: string
    }
    if (typeof parsed.detail === 'string') msg = parsed.detail
    else if (parsed.detail && typeof parsed.detail === 'object' && parsed.detail.message) {
      msg = parsed.detail.message
    } else if (parsed.message) msg = parsed.message
  } catch {
    // ignore parse
  }
  throw new Error(msg)
}

function saveBlobAsFile(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.rel = 'noopener'
  a.style.display = 'none'
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 2000)
}

/** 下载模型库中的模型文件（format: pt | onnx）。 */
export async function downloadModel(modelId: number, filename = 'best.pt', format: 'pt' | 'onnx' = 'pt') {
  const res = await http.get(`/models/${modelId}/download`, {
    params: { format },
    responseType: 'blob',
    timeout: 10 * 60 * 1000,
  })
  const blob = res.data as Blob
  await rejectIfErrorJsonBlob(blob)
  saveBlobAsFile(blob, filename)
}

/** 带鉴权下载产物。 */
export async function downloadTaskFile(taskId: number, kind: 'report' | 'export' | 'weight', name?: string) {
  const res = await http.get(`/tasks/${taskId}/download/${kind}`, {
    params: name ? { name } : undefined,
    responseType: 'blob',
    timeout: 10 * 60 * 1000,
  })
  const blob = res.data as Blob
  await rejectIfErrorJsonBlob(blob)
  const filename =
    (name || (kind === 'weight' ? 'best.pt' : kind === 'report' ? 'eval_report.html' : 'export.bin'))
      .split(/[/\\]/)
      .pop() || 'download.bin'
  saveBlobAsFile(blob, filename)
}
