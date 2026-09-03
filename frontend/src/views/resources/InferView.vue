<script setup lang="ts">
/**
 * 推理试用：从模型库选模型，上传图片，服务端推理并展示可视化结果。
 */
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Picture, VideoPlay } from '@element-plus/icons-vue'
import {
  listModels,
  predictModel,
  type ModelItem,
  type PredictResult,
} from '@/api/tasks'
import { taskTypeLabel } from '@/utils/taskType'

const route = useRoute()

const TASK_FILTERS = [
  { value: 'all', label: '全部类型' },
  { value: 'detect', label: '目标检测' },
  { value: 'segment', label: '实例分割' },
  { value: 'pose', label: '姿态估计' },
]

const loadingModels = ref(false)
const predicting = ref(false)
const models = ref<ModelItem[]>([])
const filterType = ref('all')
const modelId = ref<number | null>(null)
const conf = ref(0.25)
const iou = ref(0.45)
const imgsz = ref(640)

function applyTypeFromQuery() {
  const t = String(route.query.type || '').trim().toLowerCase()
  if (t === 'detect' || t === 'segment' || t === 'pose' || t === 'all') {
    filterType.value = t
  }
}
const file = ref<File | null>(null)
const previewUrl = ref('')
const result = ref<PredictResult | null>(null)

const usableModels = computed(() => {
  let list = models.value.filter((m) => m.has_pt !== false || m.has_onnx)
  if (filterType.value !== 'all') {
    list = list.filter((m) => (m.task_type || 'detect') === filterType.value)
  }
  return list
})

const selectedModel = computed(() => usableModels.value.find((m) => m.id === modelId.value) || null)

function displayName(name?: string) {
  const raw = (name || '').trim()
  if (!raw) return 'model'
  return raw.replace(/-best$/i, '')
}

function revokePreview() {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = ''
  }
}

async function loadModels() {
  loadingModels.value = true
  try {
    const { data } = await listModels(filterType.value === 'all' ? undefined : filterType.value)
    models.value = data
    if (modelId.value && !usableModels.value.some((m) => m.id === modelId.value)) {
      modelId.value = usableModels.value[0]?.id ?? null
    } else if (!modelId.value && usableModels.value.length) {
      modelId.value = usableModels.value[0].id
    }
  } finally {
    loadingModels.value = false
  }
}

function onFileChange(uploadFile: { raw?: File } | undefined) {
  const raw = uploadFile?.raw
  if (!raw) return
  if (!raw.type.startsWith('image/')) {
    ElMessage.warning('请上传图片文件')
    return
  }
  file.value = raw
  revokePreview()
  previewUrl.value = URL.createObjectURL(raw)
  result.value = null
}

function clearFile() {
  file.value = null
  revokePreview()
  result.value = null
}

async function onPredict() {
  if (!modelId.value) {
    ElMessage.warning('请先选择模型')
    return
  }
  if (!file.value) {
    ElMessage.warning('请先上传图片')
    return
  }
  predicting.value = true
  try {
    const { data } = await predictModel(modelId.value, file.value, {
      conf: conf.value,
      iou: iou.value,
      imgsz: imgsz.value,
    })
    result.value = data
    ElMessage.success(data.count > 0 ? `推理完成，检出 ${data.count} 个目标` : '推理完成，未检出目标')
  } catch {
    // 拦截器已提示
  } finally {
    predicting.value = false
  }
}

const resultImageSrc = computed(() => {
  if (!result.value?.image_base64) return ''
  const mime = result.value.image_mime || 'image/jpeg'
  return `data:${mime};base64,${result.value.image_base64}`
})

watch(filterType, () => {
  void loadModels()
})

onMounted(() => {
  applyTypeFromQuery()
  void loadModels()
})

onUnmounted(() => {
  revokePreview()
})
</script>

<template>
  <section class="page" v-loading="loadingModels">
    <header class="hero">
      <div class="hero-text">
        <p class="eyebrow">资源管理</p>
        <h2>推理试用</h2>
        <p class="desc">选择模型库中的训练产物，上传图片查看检测或分割结果（优先使用 PT，其次 ONNX）。</p>
      </div>
    </header>

    <div class="layout">
      <aside class="panel controls">
        <h3>参数</h3>
        <div class="field">
          <label>任务类型</label>
          <div class="type-pills">
            <button
              v-for="t in TASK_FILTERS"
              :key="t.value"
              type="button"
              class="pill"
              :class="{ active: filterType === t.value }"
              @click="filterType = t.value"
            >
              {{ t.label }}
            </button>
          </div>
        </div>

        <div class="field">
          <label>模型</label>
          <el-select
            v-model="modelId"
            filterable
            placeholder="选择模型"
            class="full"
            :disabled="!usableModels.length"
          >
            <el-option
              v-for="m in usableModels"
              :key="m.id"
              :label="`${displayName(m.name)} · ${taskTypeLabel(m.task_type)}`"
              :value="m.id"
            />
          </el-select>
          <p v-if="!usableModels.length" class="hint">暂无可用模型，请先完成训练。</p>
          <p v-else-if="selectedModel" class="hint">
            {{ selectedModel.has_pt !== false ? 'PT 可用' : '无 PT' }}
            ·
            {{ selectedModel.has_onnx ? 'ONNX 可用' : '无 ONNX' }}
          </p>
        </div>

        <div class="field">
          <label>置信度阈值 {{ conf.toFixed(2) }}</label>
          <el-slider v-model="conf" :min="0.05" :max="0.95" :step="0.05" />
        </div>
        <div class="field">
          <label>IoU {{ iou.toFixed(2) }}</label>
          <el-slider v-model="iou" :min="0.1" :max="0.9" :step="0.05" />
        </div>
        <div class="field">
          <label>推理尺寸</label>
          <el-select v-model="imgsz" class="full">
            <el-option :value="320" label="320" />
            <el-option :value="640" label="640" />
            <el-option :value="960" label="960" />
            <el-option :value="1280" label="1280" />
          </el-select>
        </div>

        <div class="field">
          <label>上传图片</label>
          <el-upload
            class="upload-compact"
            drag
            :auto-upload="false"
            :show-file-list="false"
            accept="image/*"
            :on-change="(f: any) => onFileChange(f)"
          >
            <div class="upload-inner">
              <el-icon :size="20"><Picture /></el-icon>
              <span>拖拽或点击选择图片</span>
            </div>
          </el-upload>
          <div v-if="file" class="file-row">
            <span :title="file.name">{{ file.name }}</span>
            <el-button link type="danger" @click="clearFile">清除</el-button>
          </div>
        </div>

        <el-button
          type="primary"
          class="run-btn"
          :icon="VideoPlay"
          :loading="predicting"
          :disabled="!modelId || !file"
          @click="onPredict"
        >
          开始推理
        </el-button>
      </aside>

      <div class="panel preview">
        <div class="preview-head">
          <h3>结果预览</h3>
          <span v-if="result" class="meta">
            检出 {{ result.count }} · {{ result.model_format.toUpperCase() }} ·
            {{ result.width }}×{{ result.height }}
          </span>
        </div>

        <div class="preview-body">
          <img v-if="resultImageSrc" :src="resultImageSrc" alt="推理结果" class="result-img" />
          <img v-else-if="previewUrl" :src="previewUrl" alt="原图预览" class="result-img dim" />
          <div v-else class="empty-preview">
            <el-icon :size="36"><Picture /></el-icon>
            <p>上传图片并点击「开始推理」后，结果将显示在这里</p>
          </div>
        </div>

        <div v-if="result?.detections?.length" class="table-wrap">
          <el-table :data="result.detections" size="small" max-height="220" stripe>
            <el-table-column type="index" width="48" label="#" />
            <el-table-column prop="class_name" label="类别" min-width="100" />
            <el-table-column label="置信度" width="90">
              <template #default="{ row }">{{ Number(row.confidence).toFixed(3) }}</template>
            </el-table-column>
            <el-table-column label="框 [x1,y1,x2,y2]" min-width="180">
              <template #default="{ row }">
                {{ (row.bbox_xyxy || []).map((v: number) => Math.round(v)).join(', ') }}
              </template>
            </el-table-column>
            <el-table-column label="掩膜" width="72">
              <template #default="{ row }">{{ row.polygon?.length ? '有' : '—' }}</template>
            </el-table-column>
            <el-table-column label="关键点" width="80">
              <template #default="{ row }">
                {{ row.keypoints?.length ? `${row.keypoints.length} 点` : '—' }}
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.page {
  width: 100%;
  max-width: none;
}
.hero {
  margin-bottom: 1.1rem;
  padding: 1.15rem 1.25rem;
  border-radius: var(--radius-lg);
  background:
    linear-gradient(135deg, rgba(232, 244, 242, 0.95), rgba(243, 246, 248, 0.6)),
    radial-gradient(circle at 90% 10%, rgba(61, 155, 143, 0.18), transparent 45%);
  border: 1px solid var(--line);
}
.eyebrow {
  margin: 0 0 0.25rem;
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--brand-soft);
  font-weight: 600;
}
.hero h2 {
  margin: 0;
  font-family: var(--font-display);
  font-size: 1.45rem;
  color: var(--brand-deep);
}
.desc {
  margin: 0.4rem 0 0;
  color: var(--ink-muted);
  line-height: 1.55;
  max-width: 40rem;
}
.layout {
  display: grid;
  grid-template-columns: minmax(260px, 320px) 1fr;
  gap: 0.9rem;
  align-items: start;
}
.panel {
  background: var(--surface-elevated);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  padding: 1rem;
}
.controls h3,
.preview-head h3 {
  margin: 0 0 0.85rem;
  font-size: 1rem;
  color: var(--brand-deep);
}
.field {
  margin-bottom: 0.9rem;
}
.field label {
  display: block;
  margin-bottom: 0.35rem;
  font-size: 0.82rem;
  color: var(--ink-faint);
}
.full {
  width: 100%;
}
.type-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}
.pill {
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink-muted);
  border-radius: 999px;
  padding: 0.3rem 0.7rem;
  cursor: pointer;
  font: inherit;
  font-size: 0.8rem;
}
.pill.active {
  background: var(--brand-mist);
  border-color: var(--brand-soft);
  color: var(--brand-deep);
  font-weight: 600;
}
.hint {
  margin: 0.35rem 0 0;
  font-size: 0.75rem;
  color: var(--ink-faint);
}
.upload-compact :deep(.el-upload) {
  width: 100%;
}
.upload-compact :deep(.el-upload-dragger) {
  width: 100%;
  height: auto;
  padding: 0.55rem 0.7rem;
  display: flex;
  align-items: center;
  justify-content: center;
}
.upload-inner {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  color: var(--ink-muted);
  font-size: 0.85rem;
  line-height: 1.2;
}
.upload-inner span {
  white-space: nowrap;
}
.file-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  margin-top: 0.45rem;
  font-size: 0.8rem;
  color: var(--ink-muted);
}
.file-row span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.run-btn {
  width: 100%;
  margin-top: 0.25rem;
}
.preview-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}
.preview-head h3 {
  margin: 0;
}
.meta {
  font-size: 0.8rem;
  color: var(--ink-faint);
}
.preview-body {
  min-height: 320px;
  display: grid;
  place-items: center;
  background: #f7fafb;
  border: 1px solid var(--line);
  border-radius: 10px;
  overflow: hidden;
}
.result-img {
  max-width: 100%;
  max-height: min(62vh, 640px);
  object-fit: contain;
  display: block;
}
.result-img.dim {
  opacity: 0.85;
}
.empty-preview {
  text-align: center;
  color: var(--ink-faint);
  padding: 2rem 1rem;
}
.empty-preview p {
  margin: 0.6rem 0 0;
  font-size: 0.9rem;
}
.table-wrap {
  margin-top: 0.85rem;
}
@media (max-width: 900px) {
  .layout {
    grid-template-columns: 1fr;
  }
}
</style>
