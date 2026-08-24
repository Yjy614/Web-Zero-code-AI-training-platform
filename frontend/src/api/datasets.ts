/** 数据集相关 API。 */
import http from './http'

export interface DatasetItem {
  id: number
  name: string
  path: string
  task_type: string
  owner_id: number
  classes: string[]
  /** 图片总数（含不参与训练的排除项） */
  image_count: number
  /** 参与训练的活跃图片数 */
  active_count?: number
  created_at?: string
  updated_at?: string
}

export interface ImageItem {
  name: string
  has_label: boolean
  status: 'active' | 'removed' | string
}

export interface BBox {
  class_id: number
  x_center: number
  y_center: number
  width: number
  height: number
}

export interface PolygonPoint {
  x: number
  y: number
}

export interface PolygonInstance {
  class_id: number
  points: PolygonPoint[]
}

export interface AnnotationData {
  image: string
  boxes: BBox[]
  polygons: PolygonInstance[]
}

export interface CleanResult {
  kept: number
  removed: number
  removed_files: string[]
  message: string
}

export function listDatasets(taskType = 'detect') {
  return http.get<DatasetItem[]>('/datasets', { params: { task_type: taskType } })
}

export function createDataset(name: string, taskType = 'detect') {
  return http.post<DatasetItem>('/datasets', { name, task_type: taskType })
}

export function getDataset(id: number) {
  return http.get<DatasetItem>(`/datasets/${id}`)
}

export function deleteDataset(id: number) {
  return http.delete(`/datasets/${id}`)
}

/** 下载数据集 zip 包。 */
export async function downloadDataset(id: number, name: string) {
  const res = await http.get(`/datasets/${id}/download`, { responseType: 'blob' })
  const url = URL.createObjectURL(res.data)
  const a = document.createElement('a')
  a.href = url
  a.download = `${name || 'dataset'}.zip`
  a.click()
  URL.revokeObjectURL(url)
}

export function uploadImages(id: number, files: File[], onProgress?: (p: number) => void) {
  const form = new FormData()
  files.forEach((f) => form.append('files', f))
  return http.post(`/datasets/${id}/images`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (!onProgress || !e.total) return
      onProgress(Math.round((e.loaded / e.total) * 100))
    },
  })
}

export function uploadZip(id: number, file: File, onProgress?: (p: number) => void) {
  const form = new FormData()
  form.append('file', file)
  return http.post(`/datasets/${id}/images/zip`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (!onProgress || !e.total) return
      onProgress(Math.round((e.loaded / e.total) * 100))
    },
  })
}

export function listImages(id: number) {
  return http.get<ImageItem[]>(`/datasets/${id}/images`)
}

export function cleanDataset(id: number) {
  return http.post<CleanResult>(`/datasets/${id}/clean`)
}

export function restoreDataset(id: number, names: string[]) {
  return http.post<{ restored: number; message: string; image_count: number }>(
    `/datasets/${id}/restore`,
    { names },
  )
}

export function deleteDatasetImages(id: number, names: string[]) {
  return http.post<{ deleted: number; message: string; image_count: number }>(
    `/datasets/${id}/images/delete`,
    { names },
  )
}

export interface PrelabelStatus {
  labeled_count: number
  unlabeled_count: number
  class_count: number
  can_prelabel: boolean
  min_labeled: number
  min_unlabeled: number
  last_written: string[]
  block_reason: string | null
  running_job_id: number | null
}

export function getPrelabelStatus(id: number) {
  return http.get<PrelabelStatus>(`/datasets/${id}/prelabel/status`)
}

export function startPrelabel(id: number) {
  return http.post<{
    id: number
    task_id: number
    type: string
    status: string
    progress: number
    message: string
    result: Record<string, unknown>
  }>(`/datasets/${id}/prelabel`)
}

export function revertPrelabel(id: number) {
  return http.post<{ reverted: number; message: string }>(`/datasets/${id}/prelabel/revert`)
}

export function getClasses(id: number) {
  return http.get<{ classes: string[] }>(`/datasets/${id}/classes`)
}

export function putClasses(id: number, classes: string[], removedIndices: number[] = []) {
  return http.put<{ classes: string[]; purge?: { files_touched: number; annotations_removed: number } }>(
    `/datasets/${id}/classes`,
    { classes, removed_indices: removedIndices },
  )
}

export function getAnnotation(id: number, image: string) {
  return http.get<AnnotationData>(`/datasets/${id}/annotations/${encodeURIComponent(image)}`)
}

export function putAnnotation(
  id: number,
  image: string,
  payload: { boxes?: BBox[]; polygons?: PolygonInstance[] },
) {
  return http.put<AnnotationData>(`/datasets/${id}/annotations/${encodeURIComponent(image)}`, {
    boxes: payload.boxes || [],
    polygons: payload.polygons || [],
  })
}

/** 带鉴权拉取图片并转为 Object URL。 */
export async function fetchImageObjectUrl(id: number, image: string, removed = false) {
  const res = await http.get(`/datasets/${id}/images/${encodeURIComponent(image)}/file`, {
    params: { removed },
    responseType: 'blob',
  })
  let blob = res.data as Blob
  // 错误时后端可能返回 JSON，axios 仍当 blob
  if (blob.type && /json|text/.test(blob.type)) {
    let msg = '图片下载失败'
    try {
      const text = await blob.text()
      const parsed = JSON.parse(text) as { detail?: { message?: string } | string }
      if (typeof parsed.detail === 'string') msg = parsed.detail
      else if (parsed.detail && typeof parsed.detail === 'object' && parsed.detail.message) {
        msg = parsed.detail.message
      }
    } catch {
      // ignore
    }
    throw new Error(msg)
  }
  // 部分环境 octet-stream 无法解码，按扩展名强制图片 MIME
  if (!blob.type || blob.type === 'application/octet-stream') {
    const ext = image.split('.').pop()?.toLowerCase() || 'jpg'
    const mime =
      ext === 'png'
        ? 'image/png'
        : ext === 'webp'
          ? 'image/webp'
          : ext === 'bmp'
            ? 'image/bmp'
            : 'image/jpeg'
    blob = new Blob([blob], { type: mime })
  }
  return URL.createObjectURL(blob)
}
