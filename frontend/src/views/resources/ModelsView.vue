<script setup lang="ts">
/**
 * 模型库：训练产出卡片列表，按任务类型筛选。
 * 卡片：右上角删除；下方 PT / ONNX 按钮样式统一；转 ONNX 时按钮内淡进度条。
 */
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Box, Close, Download } from '@element-plus/icons-vue'
import {
  deleteModel,
  downloadModel,
  exportModelOnnx,
  getJob,
  listModels,
  type ModelItem,
} from '@/api/tasks'
import { taskTypeLabel } from '@/utils/taskType'

const route = useRoute()

const TASK_FILTERS = [
  { value: 'all', label: '全部' },
  { value: 'detect', label: '目标检测' },
  { value: 'segment', label: '实例分割' },
  { value: 'pose', label: '姿态估计' },
]

const loading = ref(false)
const models = ref<ModelItem[]>([])
const activeType = ref('all')
const query = ref('')

function applyTypeFromQuery() {
  const t = String(route.query.type || '').trim().toLowerCase()
  if (t === 'detect' || t === 'segment' || t === 'pose' || t === 'all') {
    activeType.value = t
  }
}
/** 各模型 ONNX 转格式进度：modelId -> { progress, message } */
const onnxBusy = reactive<Record<number, { progress: number; message: string }>>({})
const pollTimers = new Map<number, number>()

const filtered = computed(() => {
  let list = models.value
  if (activeType.value !== 'all') {
    list = list.filter((m) => (m.task_type || 'detect') === activeType.value)
  }
  const q = query.value.trim().toLowerCase()
  if (q) list = list.filter((m) => m.name.toLowerCase().includes(q))
  return list
})

const latestTime = computed(() => {
  if (!filtered.value.length) return '—'
  const times = filtered.value
    .map((m) => m.created_at || '')
    .filter(Boolean)
    .sort()
  const last = times[times.length - 1]
  return last ? formatTime(last) : '—'
})

const detectCount = computed(
  () => filtered.value.filter((m) => (m.task_type || 'detect') === 'detect').length,
)

function stopOnnxPoll(modelId: number) {
  const t = pollTimers.get(modelId)
  if (t) {
    window.clearInterval(t)
    pollTimers.delete(modelId)
  }
}

function clearOnnxBusy(modelId: number) {
  stopOnnxPoll(modelId)
  delete onnxBusy[modelId]
}

async function load() {
  loading.value = true
  try {
    const { data } = await listModels(activeType.value === 'all' ? undefined : activeType.value)
    models.value = data
  } finally {
    loading.value = false
  }
}

async function onDownloadPt(row: ModelItem) {
  if (row.has_pt === false) {
    ElMessage.warning('PT 模型文件不存在')
    return
  }
  try {
    await downloadModel(row.id, `${displayModelName(row.name)}.pt`, 'pt')
    ElMessage.success('已开始下载 PT 模型')
  } catch {
    // 拦截器提示
  }
}

function pollOnnxJob(modelId: number, jobId: number, row: ModelItem) {
  stopOnnxPoll(modelId)
  pollTimers.set(
    modelId,
    window.setInterval(async () => {
      try {
        const { data } = await getJob(jobId)
        onnxBusy[modelId] = {
          progress: Math.min(100, Number(data.progress) || 0),
          message: data.message || '正在转格式…',
        }
        if (['completed', 'failed', 'cancelled'].includes(data.status)) {
          clearOnnxBusy(modelId)
          if (data.status === 'completed') {
            row.has_onnx = true
            try {
              await downloadModel(modelId, `${displayModelName(row.name)}.onnx`, 'onnx')
              ElMessage.success('ONNX 已就绪并开始下载')
              await load()
            } catch {
              ElMessage.warning('转格式完成，但下载失败，请稍后重试')
            }
          } else if (data.status === 'failed') {
            ElMessage.error(data.message || 'ONNX 转格式失败')
          } else {
            ElMessage.warning(data.message || 'ONNX 转格式已取消')
          }
        }
      } catch {
        // 轮询失败下次再试
      }
    }, 1000),
  )
}

async function onDownloadOnnx(row: ModelItem) {
  if (onnxBusy[row.id]) return
  if (row.has_pt === false) {
    ElMessage.warning('PT 模型文件不存在，无法导出 ONNX')
    return
  }
  // 已有 ONNX：直接下载
  if (row.has_onnx) {
    try {
      await downloadModel(row.id, `${displayModelName(row.name)}.onnx`, 'onnx')
      ElMessage.success('已开始下载 ONNX 模型')
      return
    } catch {
      // 可能文件被删，走转格式
      row.has_onnx = false
    }
  }

  onnxBusy[row.id] = { progress: 2, message: '准备转格式…' }
  try {
    const { data } = await exportModelOnnx(row.id)
    if (data.status === 'completed' || data.id === 0) {
      clearOnnxBusy(row.id)
      row.has_onnx = true
      await downloadModel(row.id, `${displayModelName(row.name)}.onnx`, 'onnx')
      ElMessage.success('已开始下载 ONNX 模型')
      await load()
      return
    }
    onnxBusy[row.id] = {
      progress: Math.min(100, Number(data.progress) || 5),
      message: data.message || '正在转 ONNX…',
    }
    pollOnnxJob(row.id, data.id, row)
  } catch {
    clearOnnxBusy(row.id)
  }
}

async function onDelete(row: ModelItem) {
  try {
    await ElMessageBox.confirm(
      `确定删除模型「${displayModelName(row.name)}」？删除后不可恢复。`,
      '删除模型',
      {
        type: 'warning',
        confirmButtonText: '删除',
        cancelButtonText: '取消',
      },
    )
  } catch {
    return
  }
  try {
    clearOnnxBusy(row.id)
    await deleteModel(row.id)
    ElMessage.success('已删除（含该版本的 runs / exports / reports）')
    await load()
  } catch {
    // 拦截器提示
  }
}

/** 展示用模型名：去掉历史遗留的 -best 后缀 */
function displayModelName(name?: string) {
  const raw = (name || '').trim()
  if (!raw) return 'model'
  return raw.replace(/-best$/i, '')
}

function formatTime(v?: string) {
  if (!v) return '—'
  const raw = v.trim()
  // 后端多为无时区的 UTC；无 Z/偏移时按 UTC 解析，再格式化为北京时间
  const hasTz = /([zZ]|[+-]\d{2}:?\d{2})$/.test(raw)
  const normalized = raw.includes('T') ? raw : raw.replace(' ', 'T')
  const d = new Date(hasTz ? normalized : `${normalized}Z`)
  if (Number.isNaN(d.getTime())) {
    return raw.replace('T', ' ').slice(0, 19)
  }
  return d.toLocaleString('sv-SE', { timeZone: 'Asia/Shanghai' })
}

function mapText(row: ModelItem) {
  const v = row.metrics?.map50
  if (v === undefined || v === null || v === '') return '—'
  return String(v)
}

function lossText(row: ModelItem) {
  const direct = row.metrics?.loss
  if (direct !== undefined && direct !== null && direct !== '') return String(direct)
  const hist = row.metrics?.history
  if (Array.isArray(hist) && hist.length) {
    const last = hist[hist.length - 1] as { loss?: unknown }
    if (last?.loss !== undefined && last?.loss !== null && last?.loss !== '') {
      return String(last.loss)
    }
  }
  return '—'
}

watch(activeType, () => {
  query.value = ''
  void load()
})

onMounted(() => {
  applyTypeFromQuery()
  void load()
})

onUnmounted(() => {
  for (const id of [...pollTimers.keys()]) stopOnnxPoll(id)
})
</script>

<template>
  <section class="page" v-loading="loading">
    <header class="hero">
      <div class="hero-text">
        <p class="eyebrow">资源管理</p>
        <h2>模型库</h2>
        <p class="desc">训练产出的模型记录，可下载 PT / ONNX，或删除不需要的模型。</p>
      </div>
    </header>

    <div class="stats">
      <div class="stat">
        <span>模型数</span>
        <strong>{{ filtered.length }}</strong>
      </div>
      <div class="stat">
        <span>检测模型</span>
        <strong>{{ detectCount }}</strong>
      </div>
      <div class="stat">
        <span>最近产出</span>
        <strong class="stat-time">{{ latestTime }}</strong>
      </div>
    </div>

    <div class="toolbar">
      <div class="type-pills">
        <button
          v-for="t in TASK_FILTERS"
          :key="t.value"
          type="button"
          class="pill"
          :class="{ active: activeType === t.value }"
          @click="activeType = t.value"
        >
          {{ t.label }}
        </button>
      </div>
      <el-input v-model="query" clearable placeholder="搜索模型名称" class="search" />
    </div>

    <div v-if="filtered.length" class="grid">
      <article v-for="row in filtered" :key="row.id" class="card">
        <button
          type="button"
          class="card-close"
          title="删除模型"
          @click="onDelete(row)"
        >
          <el-icon :size="14"><Close /></el-icon>
        </button>
        <div class="card-top">
          <div class="card-icon" aria-hidden="true">
            <el-icon :size="22"><Box /></el-icon>
          </div>
          <div class="card-title">
            <h3 :title="displayModelName(row.name)">{{ displayModelName(row.name) }}</h3>
            <div class="meta">
              <span
                class="tag"
                :class="{
                  'tag-segment': (row.task_type || 'detect') === 'segment',
                  'tag-pose': (row.task_type || 'detect') === 'pose',
                  'tag-detect': !['segment', 'pose'].includes(row.task_type || 'detect'),
                }"
              >
                {{ taskTypeLabel(row.task_type) }}
              </span>
              <span class="meta-time">{{ formatTime(row.created_at) }}</span>
            </div>
          </div>
        </div>
        <div class="metric-row">
          <div class="metric">
            <span>mAP@0.5</span>
            <strong>{{ mapText(row) }}</strong>
          </div>
          <div class="metric">
            <span>Loss</span>
            <strong>{{ lossText(row) }}</strong>
          </div>
        </div>
        <div class="card-actions">
          <el-button
            type="primary"
            plain
            class="action-btn"
            :icon="Download"
            :disabled="row.has_pt === false"
            @click="onDownloadPt(row)"
          >
            下载PT
          </el-button>
          <div class="onnx-wrap" :class="{ busy: Boolean(onnxBusy[row.id]) }">
            <span
              v-if="onnxBusy[row.id]"
              class="onnx-bar"
              :style="{ width: `${Math.max(6, onnxBusy[row.id].progress)}%` }"
            />
            <el-button
              type="primary"
              plain
              class="action-btn onnx-el-btn"
              :icon="onnxBusy[row.id] ? undefined : Download"
              :disabled="Boolean(onnxBusy[row.id]) || row.has_pt === false"
              :title="
                onnxBusy[row.id]
                  ? onnxBusy[row.id].message || '正在转格式…'
                  : row.has_onnx
                    ? '下载ONNX模型'
                    : '转ONNX并下载'
              "
              @click="onDownloadOnnx(row)"
            >
              <template v-if="onnxBusy[row.id]">
                {{ Math.round(onnxBusy[row.id].progress) }}%
              </template>
              <template v-else-if="row.has_onnx">下载ONNX</template>
              <template v-else>转ONNX</template>
            </el-button>
          </div>
        </div>
      </article>
    </div>

    <div v-else class="empty">
      <div class="empty-icon" aria-hidden="true"><el-icon :size="28"><Box /></el-icon></div>
      <p>
        <template v-if="query">无匹配「{{ query }}」的模型。</template>
        <template v-else-if="activeType === 'segment'">暂无实例分割训练产出模型。</template>
        <template v-else-if="activeType === 'pose'">暂无姿态估计训练产出模型。</template>
        <template v-else-if="activeType === 'detect'">暂无目标检测训练产出模型。</template>
        <template v-else>完成训练后，模型会出现在这里。</template>
      </p>
    </div>
  </section>
</template>

<style scoped>
.page {
  width: 100%;
  max-width: none;
}
.hero {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: flex-start;
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
  max-width: 36rem;
}
.stats {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
  margin-bottom: 1rem;
}
.stat {
  background: var(--surface-elevated);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  padding: 0.85rem 1rem;
}
.stat span {
  display: block;
  font-size: 0.78rem;
  color: var(--ink-faint);
}
.stat strong {
  display: block;
  margin-top: 0.25rem;
  font-family: var(--font-display);
  font-size: 1.2rem;
  color: var(--brand-deep);
}
.stat-time {
  font-size: 1rem !important;
  font-weight: 600;
}
.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.9rem;
}
.type-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}
.pill {
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink-muted);
  border-radius: 999px;
  padding: 0.4rem 0.9rem;
  cursor: pointer;
  font: inherit;
  font-size: 0.88rem;
}
.pill.active {
  background: var(--brand-mist);
  border-color: var(--brand-soft);
  color: var(--brand-deep);
  font-weight: 600;
}
.search {
  width: min(240px, 100%);
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 300px));
  gap: 0.85rem;
  justify-content: start;
  align-items: stretch;
}
.card {
  position: relative;
  display: grid;
  /* 三段固定结构：头 / 指标 / 操作，保证同行卡片横向对齐 */
  grid-template-rows: 48px auto auto;
  gap: 0.75rem;
  width: 100%;
  max-width: 300px;
  height: 100%;
  padding: 0.95rem;
  background: var(--surface-elevated);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  box-sizing: border-box;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
  border-color: #b7cdd0;
  box-shadow: var(--shadow-soft);
}
.card-close {
  position: absolute;
  top: 0.45rem;
  right: 0.45rem;
  z-index: 2;
  width: 1.55rem;
  height: 1.55rem;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: #c45656;
  display: grid;
  place-items: center;
  cursor: pointer;
  padding: 0;
  transition: background 0.15s ease, color 0.15s ease;
}
.card-close:hover {
  background: #fdecec;
  color: #a33b3b;
}
.card-top {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  min-height: 48px;
  height: 48px;
  padding-right: 1.4rem;
  min-width: 0;
}
.card-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: var(--brand-mist);
  color: var(--brand);
  display: grid;
  place-items: center;
  flex-shrink: 0;
}
.card-title {
  min-width: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 0.28rem;
}
.card-title h3 {
  margin: 0;
  font-size: 1.02rem;
  line-height: 1.25;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.meta {
  display: flex;
  gap: 0.45rem;
  align-items: center;
  min-width: 0;
  color: var(--ink-faint);
  font-size: 0.78rem;
  line-height: 1.2;
  white-space: nowrap;
}
.meta-time {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tag {
  display: inline-flex;
  flex-shrink: 0;
  align-items: center;
  justify-content: center;
  min-width: 4.2rem;
  padding: 0.1rem 0.4rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  line-height: 1.2;
}
/* 任务类型标签：尺寸统一，仅用色区分类型 */
.tag-detect {
  background: #eef2f6;
  color: #3d4f5f;
}
.tag-segment {
  background: var(--brand-mist);
  color: var(--brand-deep);
}
.tag-pose {
  background: #eef6f4;
  color: #1f6b5c;
}
.metric-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.55rem;
}
.metric {
  background: #f7fafb;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.55rem 0.7rem;
  min-height: 3.35rem;
  box-sizing: border-box;
}
.metric span {
  display: block;
  font-size: 0.72rem;
  color: var(--ink-faint);
  line-height: 1.2;
}
.metric strong {
  display: block;
  margin-top: 0.2rem;
  font-family: var(--font-display);
  font-size: 1.05rem;
  line-height: 1.2;
  color: var(--brand-deep);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.card-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  align-items: stretch;
  height: 32px;
}
.card-actions :deep(.action-btn) {
  width: 100%;
  height: 32px;
  margin: 0;
  padding: 0 0.35rem;
  font-size: 0.8rem;
  white-space: nowrap;
}
.onnx-wrap {
  position: relative;
  overflow: hidden;
  border-radius: var(--el-border-radius-base, 4px);
  min-width: 0;
  height: 32px;
}
.onnx-wrap.busy {
  cursor: wait;
}
.onnx-wrap :deep(.onnx-el-btn) {
  position: relative;
  z-index: 1;
}
.onnx-wrap.busy :deep(.onnx-el-btn) {
  color: var(--brand-deep);
  border-color: #b7cdd0;
  background: rgba(243, 248, 249, 0.55);
}
.onnx-bar {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  z-index: 0;
  background: rgba(61, 155, 143, 0.22);
  transition: width 0.35s ease;
  pointer-events: none;
}
.empty {
  text-align: center;
  padding: 2.8rem 1.2rem;
  border: 1px dashed var(--line);
  border-radius: var(--radius-lg);
  background: rgba(255, 255, 255, 0.7);
  color: var(--ink-muted);
}
.empty-icon {
  width: 56px;
  height: 56px;
  margin: 0 auto 0.75rem;
  border-radius: 16px;
  background: var(--brand-mist);
  color: var(--brand);
  display: grid;
  place-items: center;
}
.empty p {
  margin: 0;
}
@media (max-width: 720px) {
  .hero {
    flex-direction: column;
  }
  .stats {
    grid-template-columns: 1fr;
  }
  .search {
    width: 100%;
  }
}
</style>
