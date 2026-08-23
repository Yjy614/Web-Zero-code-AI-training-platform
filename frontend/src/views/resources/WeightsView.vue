<script setup lang="ts">
/**
 * 基础模型权重仓库：按任务类型分栏，卡片式管理。
 */
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, Upload, Cpu } from '@element-plus/icons-vue'
import {
  deleteWeight,
  listWeightTaskTypes,
  listWeights,
  uploadWeight,
  type WeightItem,
  type WeightTaskTypeItem,
} from '@/api/tasks'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const loading = ref(false)
const weights = ref<WeightItem[]>([])
const taskTypes = ref<WeightTaskTypeItem[]>([
  { task_type: 'detect', label: '目标检测', enabled: true },
  { task_type: 'segment', label: '实例分割', enabled: false },
])
const activeType = ref('detect')
const query = ref('')

const currentMeta = computed(
  () => taskTypes.value.find((t) => t.task_type === activeType.value) || taskTypes.value[0],
)
const canUpload = computed(() => auth.isAdmin && Boolean(currentMeta.value?.enabled))
const filtered = computed(() => {
  const q = query.value.trim().toLowerCase()
  if (!q) return weights.value
  return weights.value.filter((w) => w.name.toLowerCase().includes(q))
})
const totalSize = computed(() => weights.value.reduce((s, w) => s + (w.size || 0), 0))

async function loadTypes() {
  try {
    const { data } = await listWeightTaskTypes()
    if (data.items?.length) taskTypes.value = data.items
  } catch {
    // 默认分栏
  }
  if (!taskTypes.value.some((t) => t.task_type === activeType.value)) {
    activeType.value = taskTypes.value.find((t) => t.enabled)?.task_type || 'detect'
  }
}

async function load() {
  loading.value = true
  try {
    const { data } = await listWeights(activeType.value)
    weights.value = data
  } finally {
    loading.value = false
  }
}

async function onUpload(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file || !canUpload.value) return
  loading.value = true
  try {
    await uploadWeight(file, activeType.value)
    ElMessage.success(`已上传到「${currentMeta.value?.label || activeType.value}」`)
    await load()
  } finally {
    loading.value = false
  }
}

async function onDelete(row: WeightItem) {
  if (!auth.isAdmin) return
  try {
    await ElMessageBox.confirm(
      `确定删除「${currentMeta.value?.label}」下的权重 ${row.name}？`,
      '删除权重',
      { type: 'warning' },
    )
  } catch {
    return
  }
  loading.value = true
  try {
    await deleteWeight(row.name, activeType.value)
    ElMessage.success('已删除')
    await load()
  } finally {
    loading.value = false
  }
}

function formatSize(n: number) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(2)} MB`
}

function extLabel(name: string) {
  const ext = name.split('.').pop()?.toUpperCase() || 'PT'
  return ext
}

watch(activeType, () => {
  query.value = ''
  void load()
})

onMounted(async () => {
  await loadTypes()
  await load()
})
</script>

<template>
  <section class="page" v-loading="loading">
    <header class="hero">
      <div class="hero-text">
        <p class="eyebrow">资源管理</p>
        <h2>基础模型权重仓库</h2>
      </div>
      <label v-if="canUpload" class="upload-btn">
        <el-icon><Upload /></el-icon>
        上传到当前类型
        <input type="file" accept=".pt,.pth,.onnx" hidden @change="onUpload" />
      </label>
    </header>

    <div class="stats">
      <div class="stat">
        <span>当前类型</span>
        <strong>{{ currentMeta?.label || activeType }}</strong>
      </div>
      <div class="stat">
        <span>文件数</span>
        <strong>{{ weights.length }}</strong>
      </div>
      <div class="stat">
        <span>合计大小</span>
        <strong>{{ formatSize(totalSize) }}</strong>
      </div>
    </div>

    <div class="toolbar">
      <div class="type-pills" role="tablist">
        <button
          v-for="t in taskTypes"
          :key="t.task_type"
          type="button"
          class="pill"
          :class="{ active: activeType === t.task_type, muted: !t.enabled }"
          @click="activeType = t.task_type"
        >
          {{ t.label }}
          <em v-if="!t.enabled">预留</em>
        </button>
      </div>
      <el-input v-model="query" clearable placeholder="搜索文件名" class="search" />
    </div>

    <p v-if="!currentMeta?.enabled" class="notice">
      「{{ currentMeta?.label }}」训练模块尚未开放，目录已预留；可先上传权重，功能上线后即可选用。
    </p>

    <div v-if="filtered.length" class="grid">
      <article v-for="row in filtered" :key="row.name" class="card">
        <div class="card-icon" aria-hidden="true">
          <el-icon :size="22"><Cpu /></el-icon>
          <span class="ext">{{ extLabel(row.name) }}</span>
        </div>
        <div class="card-body">
          <h3 :title="row.name">{{ row.name }}</h3>
          <div class="meta">
            <span class="tag">{{ row.task_type || activeType }}</span>
            <span>{{ formatSize(row.size) }}</span>
          </div>
        </div>
        <div class="card-actions">
          <el-button
            v-if="auth.isAdmin"
            type="danger"
            plain
            :icon="Delete"
            @click="onDelete(row)"
          >
            删除
          </el-button>
        </div>
      </article>
    </div>

    <div v-else class="empty">
      <div class="empty-icon" aria-hidden="true"><el-icon :size="28"><Cpu /></el-icon></div>
      <p>
        「{{ currentMeta?.label || activeType }}」下暂无权重
        <template v-if="query">（无匹配「{{ query }}」）</template>
      </p>
      <label v-if="canUpload && !query" class="upload-btn ghost">
        <el-icon><Upload /></el-icon>
        上传权重
        <input type="file" accept=".pt,.pth,.onnx" hidden @change="onUpload" />
      </label>
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
.upload-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.55rem 1rem;
  background: var(--brand);
  color: #fff;
  border-radius: 10px;
  cursor: pointer;
  font-size: 0.9rem;
  flex-shrink: 0;
  border: none;
}
.upload-btn.ghost {
  background: transparent;
  color: var(--brand);
  border: 1px solid var(--brand);
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
.pill em {
  font-style: normal;
  margin-left: 0.35rem;
  font-size: 0.75rem;
  opacity: 0.7;
}
.pill.active {
  background: var(--brand-mist);
  border-color: var(--brand-soft);
  color: var(--brand-deep);
  font-weight: 600;
}
.pill.muted:not(.active) {
  opacity: 0.75;
}
.search {
  width: min(240px, 100%);
}
.notice {
  margin: 0 0 0.9rem;
  padding: 0.7rem 0.9rem;
  border-radius: 10px;
  background: #f7f3e8;
  color: #5a4a2a;
  font-size: 0.88rem;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 280px));
  gap: 0.85rem;
  justify-content: start;
}
.card {
  display: grid;
  grid-template-columns: auto 1fr;
  grid-template-rows: auto auto;
  gap: 0.65rem 0.75rem;
  width: 100%;
  max-width: 280px;
  padding: 0.95rem;
  background: var(--surface-elevated);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  align-content: start;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
  border-color: #b7cdd0;
  box-shadow: var(--shadow-soft);
}
.card-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: var(--brand-mist);
  color: var(--brand);
  display: grid;
  place-items: center;
  position: relative;
}
.ext {
  position: absolute;
  right: -4px;
  bottom: -4px;
  font-size: 0.62rem;
  font-weight: 700;
  background: var(--brand-deep);
  color: #fff;
  padding: 0.1rem 0.28rem;
  border-radius: 4px;
}
.card-body {
  min-width: 0;
}
.card-body h3 {
  margin: 0;
  font-size: 0.98rem;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.meta {
  display: flex;
  gap: 0.55rem;
  align-items: center;
  margin-top: 0.35rem;
  color: var(--ink-faint);
  font-size: 0.82rem;
}
.tag {
  display: inline-flex;
  padding: 0.12rem 0.45rem;
  border-radius: 999px;
  background: var(--brand-mist);
  color: var(--brand-deep);
  font-size: 0.75rem;
  font-weight: 600;
}
.card-actions {
  grid-column: 1 / -1;
  display: flex;
  justify-content: stretch;
}
.card-actions :deep(.el-button) {
  width: 100%;
  margin: 0;
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
  margin: 0 0 1rem;
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
