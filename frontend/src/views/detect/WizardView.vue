<script setup lang="ts">
import { computed, onActivated, onDeactivated, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import AnnotateStepPanel from '@/components/annotate/AnnotateStepPanel.vue'
import CleanThumb from '@/components/CleanThumb.vue'
import TrainFlowPanel from '@/components/wizard/TrainFlowPanel.vue'
import { getWizardSteps, type WizardTaskType } from '@/composables/useWizardSteps'
import {
  cleanDataset,
  createDataset,
  deleteDatasetImages,
  fetchImageObjectUrl,
  getDataset,
  listDatasets,
  listImages,
  restoreDataset,
  uploadImages,
  uploadZip,
  type CleanResult,
  type DatasetItem,
  type ImageItem,
} from '@/api/datasets'
import { wizardStateKey } from '@/utils/wizardSession'

defineOptions({ name: 'DetectWizard' })

const route = useRoute()
const router = useRouter()

/**
 * 任务类型在实例创建时冻结。
 * KeepAlive 失活后 useRoute() 会变成「当前全局路由」：若用 computed(route.meta)，
 * 点到其他菜单会误把 segment→detect（或点到分割页把 detect→segment），
 * 再触发重置逻辑，配置/步骤全部丢失。
 */
const taskType: WizardTaskType =
  route.meta.taskType === 'segment' || String(route.name || '').includes('segment')
    ? 'segment'
    : 'detect'
const steps = computed(() => getWizardSteps(taskType))
const wizardTitle = computed(() =>
  taskType === 'segment' ? '实例分割训练' : '目标检测训练',
)
const WIZARD_STATE_KEY = wizardStateKey(taskType)

const active = ref(0)
const loading = ref(false)
const datasets = ref<DatasetItem[]>([])
const currentId = ref<number | null>(null)
const current = ref<DatasetItem | null>(null)
const newName = ref('')
const uploadProgress = ref(0)
const images = ref<ImageItem[]>([])
const cleanResult = ref<CleanResult | null>(null)
const annotatePanelRef = ref<{ flushSave: () => Promise<void> } | null>(null)

/** 清洗页预览与多选 */
const cleanFilter = ref<'active' | 'removed'>('active')
const cleanPreviewIndex = ref(0)
const cleanPreviewUrl = ref('')
const cleanSelected = ref<Set<string>>(new Set())
let cleanPreviewRevoke: string | null = null
/** 当前大图对应的文件名，用于避免重复拉取导致闪烁 */
let cleanPreviewName = ''
let cleanPreviewSeq = 0

const activeImages = computed(() => images.value.filter((i) => i.status === 'active'))
const removedImages = computed(() => images.value.filter((i) => i.status === 'removed'))
const removedCount = computed(() => removedImages.value.length)
const cleanList = computed(() => (cleanFilter.value === 'active' ? activeImages.value : removedImages.value))
const cleanPreviewItem = computed(() => cleanList.value[cleanPreviewIndex.value] || null)
const cleanSelectedCount = computed(() => cleanSelected.value.size)
const canGoClean = computed(() => Boolean(currentId.value && (current.value?.image_count || 0) > 0))
const canGoAnnotate = computed(() => Boolean(currentId.value && activeImages.value.length > 0))

function isCleanSelected(name: string) {
  return cleanSelected.value.has(name)
}

function toggleCleanSelect(name: string) {
  const next = new Set(cleanSelected.value)
  if (next.has(name)) next.delete(name)
  else next.add(name)
  cleanSelected.value = next
}

function selectAllClean() {
  cleanSelected.value = new Set(cleanList.value.map((i) => i.name))
}

function clearCleanSelection() {
  cleanSelected.value = new Set()
}

async function refreshDatasets() {
  const { data } = await listDatasets(taskType)
  datasets.value = data
}

async function refreshCurrent() {
  if (!currentId.value) {
    current.value = null
    return
  }
  const { data } = await getDataset(currentId.value)
  current.value = data
}

async function refreshImages() {
  if (!currentId.value) {
    images.value = []
    return
  }
  const { data } = await listImages(currentId.value)
  images.value = data
}

async function selectDataset(id: number) {
  currentId.value = id
  await refreshCurrent()
  await refreshImages()
  persistWizardState()
}

async function onCreate() {
  if (!newName.value.trim()) {
    ElMessage.warning('请输入数据集名称')
    return
  }
  loading.value = true
  try {
    const { data } = await createDataset(newName.value.trim(), taskType)
    ElMessage.success('数据集已创建')
    newName.value = ''
    await refreshDatasets()
    await selectDataset(data.id)
  } finally {
    loading.value = false
  }
}

async function onUploadFiles(fileList: File[]) {
  if (!currentId.value || !fileList.length) return
  loading.value = true
  uploadProgress.value = 0
  try {
    const { data } = await uploadImages(currentId.value, fileList, (p) => {
      uploadProgress.value = p
    })
    ElMessage.success(`已上传 ${data.saved} 张图片`)
    await refreshCurrent()
    await refreshImages()
  } finally {
    loading.value = false
  }
}

async function onUploadZip(file: File) {
  if (!currentId.value) return
  loading.value = true
  uploadProgress.value = 0
  try {
    const { data } = await uploadZip(currentId.value, file, (p) => {
      uploadProgress.value = p
    })
    ElMessage.success(`ZIP 解压完成，新增 ${data.saved} 张`)
    await refreshCurrent()
    await refreshImages()
  } finally {
    loading.value = false
  }
}

function onFileInput(e: Event) {
  const input = e.target as HTMLInputElement
  const files = input.files ? Array.from(input.files) : []
  input.value = ''
  onUploadFiles(files)
}

function onZipInput(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (file) onUploadZip(file)
}

async function goStep(step: number) {
  if (step === active.value) return
  if (step > 0 && !currentId.value) {
    ElMessage.warning('请先创建或选择数据集')
    return
  }
  if (step === 1 && !canGoClean.value) {
    ElMessage.warning('请先上传图片')
    return
  }
  if (step === 2 && !canGoAnnotate.value) {
    ElMessage.warning('请先完成导入并保留至少一张图片')
    return
  }
  if (step >= 3 && !canGoAnnotate.value) {
    ElMessage.warning('请先完成导入')
    return
  }
  // 离开标注步时自动保存
  if (active.value === 2) {
    await annotatePanelRef.value?.flushSave()
  }
  active.value = step
  if (step === 1) {
    await prepareCleanStep()
  }
  persistWizardState()
}

function persistWizardState() {
  try {
    sessionStorage.setItem(
      WIZARD_STATE_KEY,
      JSON.stringify({
        datasetId: currentId.value,
        step: active.value,
      }),
    )
  } catch {
    // 忽略存储失败
  }
}

function readWizardState(): { datasetId: number | null; step: number } | null {
  try {
    const raw = sessionStorage.getItem(WIZARD_STATE_KEY)
    if (!raw) return null
    const data = JSON.parse(raw) as { datasetId?: number | null; step?: number }
    return {
      datasetId: data.datasetId != null ? Number(data.datasetId) : null,
      step: Number(data.step) || 0,
    }
  } catch {
    return null
  }
}

/** 消费并清除 URL 查询参数，避免步骤/数据集暴露在地址栏可被篡改。 */
function clearWizardQuery() {
  if (!Object.keys(route.query).length) return
  router.replace({ path: route.path, query: {} })
}

function clampReachableStep(step: number): number {
  const target = Math.min(Math.max(0, step), steps.value.length - 1)
  if (target === 0) return 0
  if (!canGoClean.value) return 0
  if (target >= 2 && !canGoAnnotate.value) return 1
  return target
}

async function hydrateWizard() {
  await refreshDatasets()
  const saved = readWizardState()
  const qid = Number(route.query.datasetId) || 0

  let targetDataset = 0
  let targetStep = 0

  if (qid) {
    // 从数据集管理带入：仅用 datasetId 作一次性入口，步骤不读 URL
    targetDataset = qid
    targetStep = saved?.datasetId === qid ? saved.step ?? 0 : 0
  } else if (saved?.datasetId) {
    // 同一次登录内从其他菜单返回：恢复会话进度（登出后会清空）
    targetDataset = saved.datasetId
    targetStep = saved.step ?? 0
  }

  if (targetDataset) {
    currentId.value = targetDataset
    await refreshCurrent()
    await refreshImages()
    const clamped = clampReachableStep(targetStep)
    active.value = clamped
    if (active.value === 1) await prepareCleanStep()
    // 仅在未因临时条件降级时写回，避免把已保存的第4步冲成更早步骤
    if (clamped >= targetStep) persistWizardState()
    else {
      try {
        sessionStorage.setItem(
          WIZARD_STATE_KEY,
          JSON.stringify({ datasetId: currentId.value, step: targetStep }),
        )
      } catch {
        // 忽略存储失败
      }
    }
  }

  clearWizardQuery()
}

watch(
  () => [active.value, currentId.value] as const,
  () => persistWizardState(),
)

async function onClean() {
  if (!currentId.value) return
  loading.value = true
  try {
    const { data } = await cleanDataset(currentId.value)
    cleanResult.value = data
    ElMessage.success(data.message)
    clearCleanSelection()
    await refreshCurrent()
    await refreshImages()
    cleanFilter.value = data.removed > 0 ? 'removed' : 'active'
    cleanPreviewIndex.value = 0
    await loadCleanPreview()
  } finally {
    loading.value = false
  }
}

async function onRestoreSelected() {
  if (!currentId.value || cleanSelectedCount.value === 0) {
    ElMessage.warning('请先勾选要恢复的图片')
    return
  }
  const excludedNames = new Set(removedImages.value.map((i) => i.name))
  const names = [...cleanSelected.value].filter((n) => excludedNames.has(n))
  if (!names.length) {
    ElMessage.warning('请在「不参与训练」中勾选要恢复的图片')
    return
  }
  loading.value = true
  try {
    const { data } = await restoreDataset(currentId.value, names)
    ElMessage.success(data.message)
    cleanResult.value = null
    clearCleanSelection()
    await refreshCurrent()
    await refreshImages()
    if (removedCount.value === 0) cleanFilter.value = 'active'
    cleanPreviewIndex.value = 0
    await loadCleanPreview()
  } finally {
    loading.value = false
  }
}

async function onDeleteSelected() {
  if (!currentId.value || cleanSelectedCount.value === 0) {
    ElMessage.warning('请先勾选要删除的图片')
    return
  }
  const names = [...cleanSelected.value]
  try {
    await ElMessageBox.confirm(
      `确定永久删除已选中的 ${names.length} 张图片吗？此操作不可恢复。`,
      '确认删除',
      { type: 'warning', confirmButtonText: '确定删除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  loading.value = true
  try {
    const { data } = await deleteDatasetImages(currentId.value, names)
    ElMessage.success(data.message)
    cleanResult.value = null
    clearCleanSelection()
    await refreshCurrent()
    await refreshImages()
    if (cleanFilter.value === 'removed' && removedCount.value === 0) {
      cleanFilter.value = 'active'
    }
    cleanPreviewIndex.value = 0
    await loadCleanPreview()
  } finally {
    loading.value = false
  }
}

function revokeCleanPreview() {
  if (cleanPreviewRevoke) {
    URL.revokeObjectURL(cleanPreviewRevoke)
    cleanPreviewRevoke = null
  }
  cleanPreviewName = ''
}

async function loadCleanPreview() {
  const item = cleanPreviewItem.value
  if (!currentId.value || !item) {
    revokeCleanPreview()
    cleanPreviewUrl.value = ''
    return
  }
  // 同一张图已在显示，跳过请求，避免主图闪一下
  if (cleanPreviewUrl.value && cleanPreviewName === item.name) return

  const seq = ++cleanPreviewSeq
  // 排除项仍在 images/，无需 removed=true
  const url = await fetchImageObjectUrl(currentId.value, item.name, false)
  // 忽略过期请求（快速连点缩略图）
  if (seq !== cleanPreviewSeq) {
    URL.revokeObjectURL(url)
    return
  }
  // 新图就绪后再替换，避免先清空造成闪烁
  revokeCleanPreview()
  cleanPreviewRevoke = url
  cleanPreviewName = item.name
  cleanPreviewUrl.value = url
}

async function selectCleanThumb(index: number) {
  if (index === cleanPreviewIndex.value && cleanPreviewUrl.value) return
  cleanPreviewIndex.value = index
  await loadCleanPreview()
}

async function switchCleanFilter(filter: 'active' | 'removed') {
  cleanFilter.value = filter
  cleanPreviewIndex.value = 0
  clearCleanSelection()
  await loadCleanPreview()
}

async function prepareCleanStep() {
  await refreshImages()
  await refreshCurrent()
  if (cleanFilter.value === 'removed' && removedCount.value === 0) {
    cleanFilter.value = 'active'
  }
  if (cleanPreviewIndex.value >= cleanList.value.length) {
    cleanPreviewIndex.value = 0
  }
  await loadCleanPreview()
}

onMounted(async () => {
  loading.value = true
  try {
    await hydrateWizard()
  } finally {
    loading.value = false
  }
})

/** 失活前再落盘一次，避免未触发 watch 时丢步骤 */
onDeactivated(() => {
  persistWizardState()
})

/** 从其他菜单返回时：刷新数据集列表；若带入新的 datasetId 则切换 */
onActivated(async () => {
  try {
    await refreshDatasets()
    // 当前选中的数据集已被删除时清空选择
    if (currentId.value && !datasets.value.some((d) => d.id === currentId.value)) {
      currentId.value = null
      current.value = null
      images.value = []
      cleanResult.value = null
      active.value = 0
      persistWizardState()
    } else if (currentId.value) {
      await refreshCurrent()
    }
  } catch {
    // 列表刷新失败不阻断后续切换
  }

  const qid = Number(route.query.datasetId)
  if (qid && qid !== currentId.value) {
    loading.value = true
    try {
      // 确认列表中仍存在该数据集
      if (datasets.value.some((d) => d.id === qid)) {
        active.value = 0
        await selectDataset(qid)
        if (active.value === 1) await prepareCleanStep()
      }
    } finally {
      loading.value = false
    }
  }
  clearWizardQuery()
})

onUnmounted(() => {
  revokeCleanPreview()
})
</script>

<template>
  <section class="wizard" v-loading="loading">
    <div class="wizard-head">
      <div>
        <h2>{{ wizardTitle }}向导</h2>
        <p>
          当前数据集：
          <strong>{{ current?.name || '未选择' }}</strong>
          <span v-if="current">（{{ current.image_count }} 张图）</span>
        </p>
      </div>
    </div>

    <ol class="step-bar">
      <li
        v-for="(step, index) in steps"
        :key="step.title"
        :class="{ active: index === active, done: index < active }"
        @click="goStep(index)"
      >
        <span class="idx">{{ index + 1 }}</span>
        <div>
          <strong>{{ step.title }}</strong>
          <small>{{ step.desc }}</small>
        </div>
      </li>
    </ol>

    <div class="panel">
      <!-- 步骤1 导入 -->
      <div v-if="active === 0" class="step-body">
        <h3>步骤 1 · 导入</h3>
        <div class="form-row">
          <el-input v-model="newName" placeholder="新建数据集名称，如 line_defect" clearable />
          <el-button type="primary" @click="onCreate">创建数据集</el-button>
        </div>

        <div class="select-block">
          <label>或选择已有数据集</label>
          <el-select
            :model-value="currentId"
            placeholder="选择数据集"
            filterable
            style="width: 100%"
            @change="selectDataset"
          >
            <el-option v-for="d in datasets" :key="d.id" :label="`${d.name}（${d.image_count} 张）`" :value="d.id" />
          </el-select>
        </div>

        <div v-if="currentId" class="upload-block">
          <p>选择图片或 ZIP 上传到服务器，保存到当前数据集。</p>
          <div class="actions">
            <label class="file-btn">
              选择图片（可多选）
              <input type="file" accept="image/*" multiple hidden @change="onFileInput" />
            </label>
            <label class="file-btn ghost">
              上传 ZIP
              <input type="file" accept=".zip" hidden @change="onZipInput" />
            </label>
          </div>
          <el-progress v-if="uploadProgress > 0 && uploadProgress < 100" :percentage="uploadProgress" />
          <p class="muted">当前共 {{ current?.image_count || 0 }} 张图片</p>
          <el-button type="primary" :disabled="!canGoClean" @click="goStep(1)">下一步：清洗</el-button>
        </div>
      </div>

      <!-- 步骤2 清洗 -->
      <div v-else-if="active === 1" class="step-body">
        <h3>步骤 2 · 清洗</h3>
        <p class="muted">
          长边归一（1280）+ 感知哈希去重。重复/异常图仅标记为「不参与训练」，文件仍保留在数据集中；只有确认删除才会真正移除。
        </p>
        <div class="actions toolbar">
          <el-button type="primary" @click="onClean">一键清洗</el-button>
          <el-button @click="selectAllClean" :disabled="!cleanList.length">全选</el-button>
          <el-button @click="clearCleanSelection" :disabled="cleanSelectedCount === 0">取消全选</el-button>
          <el-button @click="onRestoreSelected" :disabled="cleanSelectedCount === 0">恢复已选中</el-button>
          <el-button type="danger" plain @click="onDeleteSelected" :disabled="cleanSelectedCount === 0">
            删除已选中
          </el-button>
          <el-button @click="goStep(0)">上一步</el-button>
          <el-button type="primary" plain :disabled="!canGoAnnotate" @click="goStep(2)">下一步：标注</el-button>
        </div>
        <div v-if="cleanResult" class="result">
          <p>
            本次清洗：保留 {{ cleanResult.kept }} 张参与训练，本轮排除 {{ cleanResult.removed }} 张（未删除）
          </p>
        </div>

        <div class="clean-preview">
          <div class="clean-tabs">
            <button
              type="button"
              class="tab"
              :class="{ active: cleanFilter === 'active' }"
              @click="switchCleanFilter('active')"
            >
              参与训练（{{ activeImages.length }}）
            </button>
            <button
              type="button"
              class="tab"
              :class="{ active: cleanFilter === 'removed' }"
              @click="switchCleanFilter('removed')"
            >
              不参与训练（{{ removedCount }}）
            </button>
          </div>

          <div class="clean-main">
            <div class="clean-stage">
              <img v-if="cleanPreviewUrl" :src="cleanPreviewUrl" :alt="cleanPreviewItem?.name || '预览'" />
              <p v-else class="muted">暂无图片可预览</p>
              <div v-if="cleanPreviewItem" class="clean-caption">
                {{ cleanPreviewIndex + 1 }} / {{ cleanList.length }} · {{ cleanPreviewItem.name }}
                <span v-if="cleanFilter === 'removed'" class="tag-removed">不参与训练</span>
                <span v-if="cleanSelectedCount" class="tag-selected">已选 {{ cleanSelectedCount }}</span>
              </div>
            </div>
            <div class="thumb-rail">
              <div
                v-for="(item, idx) in cleanList"
                :key="item.name"
                class="thumb"
                :class="{ active: idx === cleanPreviewIndex, selected: isCleanSelected(item.name) }"
              >
                <label class="thumb-check" @click.stop>
                  <input
                    type="checkbox"
                    :checked="isCleanSelected(item.name)"
                    @change="toggleCleanSelect(item.name)"
                  />
                </label>
                <button type="button" class="thumb-body" :title="item.name" @click="selectCleanThumb(idx)">
                <CleanThumb
                  :dataset-id="currentId!"
                  :name="item.name"
                />
                  <span>{{ item.name }}</span>
                </button>
              </div>
              <p v-if="!cleanList.length" class="muted">当前列表为空</p>
            </div>
          </div>
        </div>
      </div>

      <!-- 步骤3 标注（独立组件，后期可平行替换为分割标注） -->
      <AnnotateStepPanel
        v-else-if="active === 2 && currentId"
        ref="annotatePanelRef"
        :dataset-id="currentId"
        :task-type="taskType"
        @back="goStep(1)"
        @next="goStep(3)"
      />

      <!-- 步骤 4~7：配置 / 训练 / 评估 / 导出 -->
      <TrainFlowPanel
        v-else-if="active >= 3 && currentId && current"
        :step="active"
        :dataset-id="currentId"
        :dataset-name="current.name"
        :image-count="current.active_count ?? current.image_count"
        :task-type="taskType"
        @update:step="goStep"
      />
    </div>
  </section>
</template>

<style scoped>
.wizard {
  display: grid;
  gap: 1.1rem;
}
.wizard-head h2 {
  margin: 0;
  font-family: var(--font-display);
  font-size: 1.45rem;
}
.wizard-head p {
  margin: 0.4rem 0 0;
  color: var(--ink-muted);
}
.step-bar {
  list-style: none;
  margin: 0;
  padding: 0.9rem;
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 0.45rem;
  background: rgba(255, 255, 255, 0.78);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
}
.step-bar li {
  display: flex;
  gap: 0.55rem;
  align-items: flex-start;
  padding: 0.45rem;
  border-radius: 10px;
  opacity: 0.55;
  cursor: pointer;
}
.step-bar li.active {
  opacity: 1;
  background: var(--brand-mist);
}
.step-bar li.done {
  opacity: 0.85;
}
.idx {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  flex-shrink: 0;
  font-size: 0.75rem;
  font-weight: 700;
  color: #fff;
  background: var(--brand);
}
.step-bar strong {
  display: block;
  font-size: 0.88rem;
}
.step-bar small {
  color: var(--ink-faint);
  font-size: 0.72rem;
}
.panel {
  min-height: 360px;
  background: var(--surface-elevated);
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-soft);
  padding: 1.5rem;
}
.step-body h3 {
  margin: 0 0 0.85rem;
  font-family: var(--font-display);
}
.form-row {
  display: flex;
  gap: 0.6rem;
  margin-bottom: 1rem;
}
.form-row.compact {
  margin-bottom: 0;
}
.select-block {
  margin-bottom: 1.1rem;
}
.select-block label {
  display: block;
  margin-bottom: 0.35rem;
  color: var(--ink-muted);
  font-size: 0.88rem;
}
.upload-block code {
  font-size: 0.86em;
  background: var(--brand-mist);
  padding: 0.1em 0.35em;
  border-radius: 4px;
}
.actions {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
  margin: 0.8rem 0;
  align-items: center;
}
.actions.toolbar {
  margin-bottom: 0.6rem;
}
.file-btn {
  display: inline-flex;
  align-items: center;
  padding: 0.55rem 0.95rem;
  background: var(--brand);
  color: #fff;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.9rem;
}
.file-btn.ghost {
  background: transparent;
  color: var(--brand);
  border: 1px solid var(--brand);
}
.muted {
  color: var(--ink-muted);
  font-size: 0.9rem;
  line-height: 1.6;
}
.result {
  margin: 0.8rem 0;
  padding: 0.85rem 1rem;
  background: var(--brand-mist);
  border-radius: 10px;
}
.clean-preview {
  margin-top: 1rem;
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  overflow: hidden;
  background: #fff;
}
.clean-tabs {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--line);
}
.clean-tabs .tab {
  flex: 1;
  border: 0;
  background: transparent;
  padding: 0.7rem 0.85rem;
  cursor: pointer;
  color: var(--ink-muted);
  font-size: 0.9rem;
}
.clean-tabs .tab.active {
  color: var(--brand-deep);
  font-weight: 600;
  box-shadow: inset 0 -2px 0 var(--brand);
  background: var(--brand-mist);
}
.clean-main {
  display: grid;
  grid-template-columns: 1fr 200px;
  min-height: 360px;
}
.clean-stage {
  display: grid;
  place-items: center;
  padding: 1rem;
  background:
    linear-gradient(45deg, #eef2f5 25%, transparent 25%),
    linear-gradient(-45deg, #eef2f5 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #eef2f5 75%),
    linear-gradient(-45deg, transparent 75%, #eef2f5 75%);
  background-size: 20px 20px;
  background-position: 0 0, 0 10px, 10px -10px, -10px 0;
  position: relative;
}
.clean-stage img {
  max-width: 100%;
  max-height: 420px;
  object-fit: contain;
  border-radius: 6px;
  background: #111;
}
.clean-caption {
  position: absolute;
  left: 0.75rem;
  right: 0.75rem;
  bottom: 0.75rem;
  padding: 0.35rem 0.55rem;
  border-radius: 6px;
  background: rgba(28, 43, 54, 0.72);
  color: #fff;
  font-size: 0.8rem;
  display: flex;
  gap: 0.5rem;
  align-items: center;
  word-break: break-all;
}
.tag-removed {
  margin-left: auto;
  background: #c4473a;
  border-radius: 4px;
  padding: 0.1rem 0.4rem;
  font-size: 0.72rem;
  flex-shrink: 0;
}
.tag-selected {
  background: var(--brand);
  border-radius: 4px;
  padding: 0.1rem 0.4rem;
  font-size: 0.72rem;
  flex-shrink: 0;
}
.thumb-rail {
  border-left: 1px solid var(--line);
  padding: 0.65rem;
  overflow: auto;
  max-height: 460px;
  display: grid;
  gap: 0.45rem;
  align-content: start;
}
.thumb {
  position: relative;
  border: 1px solid var(--line);
  background: #fff;
  border-radius: 8px;
  padding: 0.35rem;
  text-align: left;
}
.thumb.active {
  border-color: var(--brand);
  box-shadow: 0 0 0 1px var(--brand);
}
.thumb.selected {
  background: var(--brand-mist);
}
.thumb-check {
  position: absolute;
  top: 0.35rem;
  left: 0.35rem;
  z-index: 2;
  display: flex;
  padding: 0.1rem;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 3px;
  cursor: pointer;
}
.thumb-body {
  display: block;
  width: 100%;
  border: 0;
  background: transparent;
  padding: 0;
  cursor: pointer;
  text-align: left;
}
.thumb span {
  display: block;
  margin-top: 0.25rem;
  font-size: 0.72rem;
  color: var(--ink-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
@media (max-width: 1100px) {
  .step-bar {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .clean-main {
    grid-template-columns: 1fr;
  }
  .thumb-rail {
    border-left: 0;
    border-top: 1px solid var(--line);
    max-height: 200px;
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  }
}
</style>
