<script setup lang="ts">
/**
 * 数据集管理：卡片列表 + 任务类型筛选。
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Collection, Download, Delete, Right, Plus } from '@element-plus/icons-vue'
import {
  createDataset,
  deleteDataset,
  downloadDataset,
  listDatasets,
  type DatasetItem,
} from '@/api/datasets'

const TASK_FILTERS = [
  { value: 'detect', label: '目标检测' },
  { value: 'segment', label: '实例分割' },
]

const router = useRouter()
const loading = ref(false)
const datasets = ref<DatasetItem[]>([])
const createVisible = ref(false)
const newName = ref('')
const activeType = ref('detect')
const query = ref('')

const filtered = computed(() => {
  const q = query.value.trim().toLowerCase()
  let list = datasets.value.filter((d) => (d.task_type || 'detect') === activeType.value)
  if (q) list = list.filter((d) => d.name.toLowerCase().includes(q))
  return list
})
const totalImages = computed(() =>
  filtered.value.reduce((s, d) => s + (d.image_count || 0), 0),
)

async function load() {
  loading.value = true
  try {
    const { data } = await listDatasets(activeType.value)
    datasets.value = data
  } finally {
    loading.value = false
  }
}

async function onCreate() {
  if (!newName.value.trim()) {
    ElMessage.warning('请输入名称')
    return
  }
  if (activeType.value !== 'detect') {
    ElMessage.warning('当前仅开放目标检测数据集创建')
    return
  }
  const { data } = await createDataset(newName.value.trim())
  ElMessage.success('已创建')
  createVisible.value = false
  newName.value = ''
  await load()
  router.push({ path: '/app/detect/wizard', query: { datasetId: String(data.id) } })
}

async function onDelete(row: DatasetItem) {
  try {
    await ElMessageBox.confirm(`确认删除数据集「${row.name}」？磁盘文件将一并删除。`, '删除确认', {
      type: 'warning',
    })
  } catch {
    return
  }
  await deleteDataset(row.id)
  ElMessage.success('已删除')
  await load()
}

async function onDownload(row: DatasetItem) {
  try {
    await downloadDataset(row.id, row.name)
    ElMessage.success('已开始下载')
  } catch {
    // 拦截器提示
  }
}

function openWizard(row: DatasetItem) {
  if ((row.task_type || 'detect') !== 'detect') {
    ElMessage.info('该任务类型向导尚未开放')
    return
  }
  router.push({ path: '/app/detect/wizard', query: { datasetId: String(row.id) } })
}

/** 类别数量；无类别时不展示 */
function classCount(row: DatasetItem) {
  return (row.classes || []).filter((c) => String(c).trim()).length
}

function metaTitle(row: DatasetItem) {
  const n = classCount(row)
  const names = (row.classes || []).map((c) => String(c).trim()).filter(Boolean).join('、')
  const base = `${row.image_count} 张图 · ${n} 种类别`
  return names ? `${base}：${names}` : base
}

watch(activeType, () => {
  query.value = ''
  void load()
})

onMounted(load)
</script>

<template>
  <section class="page" v-loading="loading">
    <header class="hero">
      <div class="hero-text">
        <p class="eyebrow">资源管理</p>
        <h2>数据集管理</h2>
        <p class="desc">管理服务器侧数据集，进入向导继续标注训练，或打包下载到本地。</p>
      </div>
      <el-button
        type="primary"
        :icon="Plus"
        :disabled="activeType !== 'detect'"
        @click="createVisible = true"
      >
        新建数据集
      </el-button>
    </header>

    <div class="stats">
      <div class="stat">
        <span>当前类型</span>
        <strong>{{ TASK_FILTERS.find((t) => t.value === activeType)?.label }}</strong>
      </div>
      <div class="stat">
        <span>数据集</span>
        <strong>{{ filtered.length }}</strong>
      </div>
      <div class="stat">
        <span>图片合计</span>
        <strong>{{ totalImages }}</strong>
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
      <el-input v-model="query" clearable placeholder="搜索数据集名称" class="search" />
    </div>

    <div v-if="filtered.length" class="grid">
      <article v-for="row in filtered" :key="row.id" class="card">
        <div class="card-top">
          <div class="card-icon" aria-hidden="true">
            <el-icon :size="22"><Collection /></el-icon>
          </div>
          <div class="card-title">
            <h3 :title="row.name">{{ row.name }}</h3>
            <div class="meta">
              <span class="tag">{{ row.task_type || 'detect' }}</span>
              <span class="meta-line" :title="metaTitle(row)">
                <span>{{ row.image_count }} 张图</span>
                <span class="meta-sep" aria-hidden="true" />
                <span>{{ classCount(row) }} 种类别</span>
              </span>
            </div>
          </div>
        </div>
        <div
          class="classes"
          :title="classCount(row) ? (row.classes || []).filter(Boolean).join('、') : undefined"
        >
          <span v-for="c in row.classes || []" :key="c" class="chip">{{ c }}</span>
        </div>
        <div class="card-actions">
          <el-button type="primary" plain :icon="Right" @click="openWizard(row)">进入向导</el-button>
          <el-button :icon="Download" @click="onDownload(row)">下载</el-button>
          <el-button type="danger" plain :icon="Delete" @click="onDelete(row)">删除</el-button>
        </div>
      </article>
    </div>

    <div v-else class="empty">
      <div class="empty-icon" aria-hidden="true"><el-icon :size="28"><Collection /></el-icon></div>
      <p>
        <template v-if="activeType === 'segment'">实例分割数据集能力即将开放。</template>
        <template v-else-if="query">无匹配「{{ query }}」的数据集。</template>
        <template v-else>暂无数据集。可在此新建，或前往训练向导创建并上传。</template>
      </p>
      <el-button
        v-if="activeType === 'detect' && !query"
        type="primary"
        :icon="Plus"
        @click="createVisible = true"
      >
        新建数据集
      </el-button>
    </div>

    <el-dialog v-model="createVisible" title="新建检测数据集" width="420px">
      <el-input
        v-model="newName"
        placeholder="名称（中文/字母/数字/下划线/短横线）"
        @keyup.enter="onCreate"
      />
      <template #footer>
        <el-button @click="createVisible = false">取消</el-button>
        <el-button type="primary" @click="onCreate">创建</el-button>
      </template>
    </el-dialog>
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
    radial-gradient(circle at 90% 10%, rgba(26, 95, 122, 0.16), transparent 45%);
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
  grid-template-columns: repeat(auto-fill, minmax(260px, 280px));
  gap: 0.85rem;
  justify-content: start;
}
.card {
  display: grid;
  gap: 0.75rem;
  width: 100%;
  max-width: 280px;
  padding: 0.95rem;
  background: var(--surface-elevated);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  /* 同行卡片被撑高时，内容顶对齐，避免中间区域被拉长 */
  align-content: start;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
  border-color: #b7cdd0;
  box-shadow: var(--shadow-soft);
}
.card-top {
  display: flex;
  gap: 0.75rem;
  align-items: flex-start;
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
}
.card-title h3 {
  margin: 0;
  font-size: 1.02rem;
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
  min-width: 0;
}
.meta-line {
  min-width: 0;
  flex: 1;
  display: inline-flex;
  align-items: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.meta-sep {
  display: inline-block;
  width: 0.65rem;
  flex-shrink: 0;
}
.tag {
  display: inline-flex;
  flex-shrink: 0;
  padding: 0.12rem 0.45rem;
  border-radius: 999px;
  background: var(--brand-mist);
  color: var(--brand-deep);
  font-size: 0.75rem;
  font-weight: 600;
}
.classes {
  display: flex;
  flex-wrap: nowrap;
  gap: 0.35rem;
  align-items: center;
  overflow: hidden;
  max-width: 100%;
  /* 无类别时也占一行，保证卡片高度一致 */
  min-height: 1.55rem;
}
.chip {
  display: inline-flex;
  flex-shrink: 0;
  align-items: center;
  font-size: 0.75rem;
  line-height: 1.25;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  background: #f0f4f7;
  color: var(--ink-muted);
  border: 1px solid var(--line);
  white-space: nowrap;
}
.card-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}
.card-actions :deep(.el-button) {
  flex: 1;
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
