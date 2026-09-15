<script setup lang="ts">
/**
 * 主动学习一条龙：确认原集 → 上传新图 → 筛难例 → 复核 → 并入 → 接着模型续训
 * 原数据集由模型训练血缘锁定，不可换其它集。
 */
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ArrowLeft } from '@element-plus/icons-vue'
import {
  abandonAlSession,
  cleanupAlOrphans,
  createAlSession,
  getAlHint,
  getAlSession,
  mergeAlSession,
  prepareAlRetrain,
  startAlScreen,
  type AlHint,
  type AlSession,
} from '@/api/activeLearn'
import { uploadImages, uploadZip, getDataset, listImages } from '@/api/datasets'
import { getJob, startTrain } from '@/api/tasks'
import AnnotateStepPanel from '@/components/annotate/AnnotateStepPanel.vue'

const route = useRoute()
const router = useRouter()

const modelId = computed(() => Number(route.query.modelId) || 0)
const loading = ref(false)
const step = ref(0) // 0确认原集 1上传 2筛图 3复核 4并入续训
const hint = ref<AlHint | null>(null)
const session = ref<AlSession | null>(null)
const uploading = ref(false)
const uploadPercent = ref(0)
const stagingImageCount = ref(0)
const stagingImageNames = ref<string[]>([])
const screening = ref(false)
const screenProgress = ref(0)
const screenMessage = ref('')
const retraining = ref(false)
/** 本页内复核：hard / easy，不跳转训练向导 */
const reviewMode = ref<'hard' | 'easy' | null>(null)
const annotateRef = ref<{ flushSave: () => Promise<void> } | null>(null)
/** 去向导续训时保留临时集，不在卸载时取消 */
const preserveSessionOnLeave = ref(false)
let pollTimer: number | null = null

const ACTIVE_STATUSES = new Set(['created', 'uploaded', 'screening', 'screened'])

const hardItems = computed(() => (session.value?.items || []).filter((x) => x.difficulty === 'hard'))
const easyItems = computed(() => (session.value?.items || []).filter((x) => x.difficulty === 'easy'))
const reviewNames = computed(() => {
  if (reviewMode.value === 'hard') return hardItems.value.map((x) => x.image_name)
  if (reviewMode.value === 'easy') return easyItems.value.map((x) => x.image_name)
  return []
})
const reviewTitle = computed(() =>
  reviewMode.value === 'hard' ? '复核难例' : reviewMode.value === 'easy' ? '快速检查简单例' : '',
)

const sessionTaskType = computed(() => {
  const t = (session.value?.task_type || hint.value?.task_type || 'detect').toLowerCase()
  if (t === 'segment' || t === 'pose') return t
  return 'detect'
})

function wizardPathFor(tt: string) {
  if (tt === 'segment') return '/app/segment/wizard'
  if (tt === 'pose') return '/app/pose/wizard'
  return '/app/detect/wizard'
}

const lockedDatasetLabel = computed(() => {
  const h = hint.value
  if (!h?.default_target_dataset_id) return '未找到原训练数据集'
  const parts = [`${h.default_target_dataset_name || '未命名'}（#${h.default_target_dataset_id}）`]
  if (h.target_image_count != null) parts.push(`${h.target_image_count} 张图`)
  if (h.target_class_count != null) parts.push(`${h.target_class_count} 类`)
  return parts.join(' · ')
})

function stopPoll() {
  if (pollTimer) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

function stepFromStatus(status?: string) {
  const s = status || ''
  if (s === 'merged' || s === 'retraining' || s === 'done') return 4
  if (s === 'screened') return 3
  if (s === 'screening') return 2
  if (s === 'created' || s === 'uploaded') return 1
  return 0
}

function isSessionActive(s?: AlSession | null) {
  return Boolean(s?.id && ACTIVE_STATUSES.has(String(s.status || '')))
}

/** 刷新临时集已上传图片数量（原生文件框清空后仍显示「未选择文件」，以此为准） */
async function refreshStagingImages() {
  const sid = session.value?.staging_dataset_id
  if (!sid) {
    stagingImageCount.value = 0
    stagingImageNames.value = []
    return
  }
  try {
    const [dsRes, imgRes] = await Promise.all([getDataset(sid), listImages(sid)])
    const imgs = imgRes.data || []
    stagingImageNames.value = imgs.map((x) => x.name).slice(0, 20)
    stagingImageCount.value =
      Number(dsRes.data.active_count ?? dsRes.data.image_count ?? imgs.length) || imgs.length
  } catch {
    stagingImageCount.value = 0
    stagingImageNames.value = []
  }
}

async function abandonCurrent(silent = false) {
  if (!session.value?.id || !isSessionActive(session.value)) return
  try {
    await abandonAlSession(session.value.id)
    if (!silent) ElMessage.info('已取消本轮，临时数据集已删除')
  } catch {
    //
  } finally {
    session.value = null
    const q = { ...route.query } as Record<string, string>
    delete q.sessionId
    await router.replace({ query: q })
  }
}

async function bootstrap() {
  if (!modelId.value) {
    ElMessage.error('缺少模型 ID')
    router.push('/app/resources/models')
    return
  }
  loading.value = true
  try {
    await cleanupAlOrphans().catch(() => undefined)
    const { data } = await getAlHint(modelId.value)
    hint.value = data
    if (!data.can_start) {
      ElMessage.warning(data.message || '当前模型暂不支持主动学习')
    }
    const sid = String(route.query.sessionId || '').trim()
    if (sid) {
      try {
        const r = await getAlSession(sid)
        session.value = r.data.session
        step.value = stepFromStatus(r.data.session.status)
        if (r.data.session.staging_dataset_id) {
          await refreshStagingImages()
        }
      } catch {
        session.value = null
        step.value = 0
      }
    }
  } catch {
    // 拦截器
  } finally {
    loading.value = false
  }
}

async function onCreateSession() {
  if (!hint.value?.can_start || !hint.value.default_target_dataset_id) {
    ElMessage.warning(hint.value?.message || '无法锁定原数据集')
    return
  }
  loading.value = true
  try {
    const { data } = await createAlSession(modelId.value)
    session.value = data.session
    stagingImageCount.value = 0
    stagingImageNames.value = []
    step.value = 1
    await router.replace({
      query: { ...route.query, modelId: String(modelId.value), sessionId: data.session.id },
    })
    ElMessage.success('已创建本轮主动学习，请上传新图片')
  } catch {
    //
  } finally {
    loading.value = false
  }
}

async function onFiles(ev: Event) {
  const input = ev.target as HTMLInputElement
  const files = Array.from(input.files || [])
  // 清空原生框：浏览器会回到「未选择文件」，不代表未上传成功
  input.value = ''
  if (!files.length || !session.value?.staging_dataset_id) return
  uploading.value = true
  uploadPercent.value = 0
  try {
    const zips = files.filter((f) => /\.zip$/i.test(f.name))
    const imgs = files.filter((f) => !/\.zip$/i.test(f.name))
    const sid = session.value.staging_dataset_id
    let saved = 0
    for (const z of zips) {
      const zr = await uploadZip(sid, z, (p) => {
        uploadPercent.value = p
      })
      saved += Number((zr.data as { saved?: number })?.saved || 0)
    }
    if (imgs.length) {
      const ir = await uploadImages(sid, imgs, (p) => {
        uploadPercent.value = p
      })
      saved += Number((ir.data as { saved?: number })?.saved || imgs.length)
    }
    const refreshed = await getAlSession(session.value.id)
    session.value = refreshed.data.session
    // 标记已上传，便于恢复步骤
    if (session.value.status === 'created') {
      session.value = { ...session.value, status: 'uploaded' }
    }
    await refreshStagingImages()
    ElMessage.success(
      stagingImageCount.value > 0
        ? `上传完成，临时集现有 ${stagingImageCount.value} 张（本批约 ${saved || files.length} 个文件）`
        : '上传请求已完成，但临时集仍无图片，请检查文件格式',
    )
  } catch {
    //
  } finally {
    uploading.value = false
  }
}

async function goScreenStep() {
  await refreshStagingImages()
  if (stagingImageCount.value <= 0) {
    ElMessage.warning('请先上传至少一张图片（「未选择文件」是浏览器清空选择框后的正常显示）')
    return
  }
  step.value = 2
}

async function onScreen() {
  if (!session.value) return
  await refreshStagingImages()
  if (stagingImageCount.value <= 0) {
    ElMessage.warning('临时集没有图片，请返回上一步上传')
    step.value = 1
    return
  }
  screening.value = true
  screenProgress.value = 0
  screenMessage.value = '排队中…'
  try {
    const { data } = await startAlScreen(session.value.id)
    session.value = data.session
    const jobId = data.job.id
    stopPoll()
    pollTimer = window.setInterval(async () => {
      try {
        const jr = await getJob(jobId)
        const job = jr.data
        screenProgress.value = Number(job.progress || 0)
        screenMessage.value = job.message || ''
        if (['completed', 'failed', 'cancelled'].includes(job.status)) {
          stopPoll()
          screening.value = false
          const refreshed = await getAlSession(session.value!.id)
          session.value = refreshed.data.session
          if (job.status === 'completed') {
            step.value = 3
            ElMessage.success(job.message || '筛图完成')
          } else {
            ElMessage.error(job.message || '筛图失败')
          }
        }
      } catch {
        stopPoll()
        screening.value = false
      }
    }, 1000)
  } catch {
    screening.value = false
  }
}

function openReview(preferHard: boolean) {
  if (!session.value?.staging_dataset_id) return
  const mode = preferHard ? 'hard' : 'easy'
  const names =
    mode === 'hard' ? hardItems.value.map((x) => x.image_name) : easyItems.value.map((x) => x.image_name)
  if (!names.length) {
    ElMessage.info(mode === 'hard' ? '本轮没有难例' : '本轮没有简单例')
    return
  }
  reviewMode.value = mode
}

async function closeReview() {
  try {
    await annotateRef.value?.flushSave()
  } catch {
    //
  }
  reviewMode.value = null
  ElMessage.success('标注已保存，可继续复核另一类，或并入原集续训')
}

async function onMergeAndRetrain() {
  if (!session.value) return
  try {
    await ElMessageBox.confirm(
      '将把本轮图片与标注并入原数据集，并准备「接着当前模型继续训练」。确认后仍需再点一次启动训练。',
      '并入并准备续训',
      { type: 'warning', confirmButtonText: '继续', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  retraining.value = true
  try {
    const merged = await mergeAlSession(session.value.id)
    session.value = merged.data.session
    ElMessage.success(merged.data.message)
    const prep = await prepareAlRetrain(session.value.id, { epochs: 50, batch: 8, device: 'cpu' })
    session.value = prep.data.session
    const taskId = prep.data.task_id
    await ElMessageBox.confirm(
      `续训任务 #${taskId} 已就绪（权重：接着当前模型）。是否现在启动训练？`,
      '确认开训',
      { type: 'info', confirmButtonText: '确认开训', cancelButtonText: '稍后' },
    )
    const job = await startTrain(taskId)
    ElMessage.success(`训练已启动 Job #${job.data.id}`)
    step.value = 4
    preserveSessionOnLeave.value = true
    router.push({
      path: wizardPathFor(String(prep.data.task_type || sessionTaskType.value)),
      query: { datasetId: String(prep.data.dataset_id), step: '3' },
    })
  } catch {
    // 取消确认不提示
  } finally {
    retraining.value = false
  }
}

async function onCancelRound() {
  if (!isSessionActive(session.value)) {
    step.value = 0
    return
  }
  try {
    await ElMessageBox.confirm(
      '取消后将删除本轮临时数据集（未并入原集的图片会丢失）。',
      '取消本轮主动学习',
      { type: 'warning', confirmButtonText: '确认取消', cancelButtonText: '继续本轮' },
    )
  } catch {
    return
  }
  await abandonCurrent()
  step.value = 0
}

async function backModels() {
  if (reviewMode.value) {
    await closeReview()
  }
  if (isSessionActive(session.value)) {
    try {
      await ElMessageBox.confirm(
        '离开将取消本轮主动学习，并删除尚未并入的临时数据集。',
        '返回模型库',
        { type: 'warning', confirmButtonText: '离开并删除', cancelButtonText: '留在本页' },
      )
    } catch {
      return
    }
    await abandonCurrent(true)
  }
  router.push('/app/resources/models')
}

onMounted(() => {
  void bootstrap()
})
onUnmounted(() => {
  stopPoll()
  // 去向导复核 / 续训时保留；其它离开路径由返回/取消显式处理
  if (!preserveSessionOnLeave.value && isSessionActive(session.value) && session.value?.id) {
    void abandonAlSession(session.value.id).catch(() => undefined)
  }
})

watch(
  () => route.query.modelId,
  async () => {
    if (isSessionActive(session.value)) {
      await abandonCurrent(true)
    }
    step.value = 0
    session.value = null
    void bootstrap()
  },
)
</script>

<template>
  <section class="al-page" :class="{ reviewing: Boolean(reviewMode) }" v-loading="loading">
    <header class="head">
      <button type="button" class="back" @click="reviewMode ? closeReview() : backModels()">
        <el-icon><ArrowLeft /></el-icon>
        {{ reviewMode ? '返回主动学习' : '返回模型库' }}
      </button>
      <div>
        <h2>{{ reviewMode ? `主动学习 · ${reviewTitle}` : '主动学习' }}</h2>
        <p class="lead">
          <template v-if="reviewMode">
            在本页核对并修正框；完成后点「返回主动学习」继续并入，不会进入训练向导。
          </template>
          <template v-else>上传新图 → 筛难例 → 人工复核 → 并入原集 → 接着模型续训</template>
        </p>
      </div>
    </header>

    <!-- 页内复核：复用标注画布，但不进入训练向导流程 -->
    <div v-if="reviewMode && session?.staging_dataset_id" class="review-shell">
      <div class="review-toolbar">
        <span
          >共 {{ reviewNames.length }} 张 ·
          {{ reviewMode === 'hard' ? '请优先仔细核对' : '可快速过一遍草稿框' }}</span
        >
        <el-button type="primary" @click="closeReview">完成复核，返回主动学习</el-button>
      </div>
      <AnnotateStepPanel
        :key="`${session.staging_dataset_id}-${reviewMode}-${sessionTaskType}`"
        ref="annotateRef"
        embedded
        hide-prelabel
        :task-type="sessionTaskType"
        :dataset-id="session.staging_dataset_id"
        :only-names="reviewNames"
        back-label="完成复核，返回主动学习"
        @back="closeReview"
      />
    </div>

    <template v-else>
      <el-steps :active="step" finish-status="success" align-center class="steps">
        <el-step title="确认原数据集" />
        <el-step title="上传新图" />
        <el-step title="筛图" />
        <el-step title="复核" />
        <el-step title="并入续训" />
      </el-steps>

      <!-- 步骤 0：原集由模型血缘锁定，只确认不更换 -->
      <div v-if="step === 0" class="panel">
        <p v-if="hint && !hint.can_start" class="warn">{{ hint.message }}</p>
        <p class="hint">
          本轮新图复核后会并入该模型的<strong>原训练数据集</strong>，再接着当前模型续训（不可更换其它数据集）。
        </p>
        <div class="locked-ds">
          <span class="locked-label">原数据集</span>
          <strong>{{ lockedDatasetLabel }}</strong>
          <span v-if="hint?.model_name" class="muted-inline">模型：{{ hint.model_name }}</span>
        </div>
        <div class="actions">
          <el-button
            type="primary"
            :disabled="!hint?.can_start || !hint?.default_target_dataset_id"
            @click="onCreateSession"
          >
            确认并开始本轮
          </el-button>
        </div>
      </div>

      <!-- 步骤 1 -->
      <div v-else-if="step === 1" class="panel">
        <p class="hint">
          本轮图片暂存于临时集（不进入数据集管理），确认并入后才会写入原集「{{
            session?.target_dataset_name
          }}」。
        </p>
        <div class="upload-status" :class="{ ok: stagingImageCount > 0 }">
          <template v-if="stagingImageCount > 0">
            已上传 <strong>{{ stagingImageCount }}</strong> 张到临时集，可继续追加。
          </template>
          <template v-else>尚未上传图片，请选择图片或 zip。</template>
        </div>
        <ul v-if="stagingImageNames.length" class="name-preview">
          <li v-for="n in stagingImageNames" :key="n">{{ n }}</li>
          <li v-if="stagingImageCount > stagingImageNames.length" class="muted">
            …共 {{ stagingImageCount }} 张
          </li>
        </ul>
        <label class="file-btn" :class="{ disabled: uploading }">
          <input type="file" multiple accept="image/*,.zip" :disabled="uploading" @change="onFiles" />
          <el-button type="primary" plain :loading="uploading" tag="span">
            {{ uploading ? '上传中…' : '选择并上传图片' }}
          </el-button>
        </label>
        <p class="field-note">支持多选图片或 zip；可多次追加。上传成功后请看上方「已上传 N 张」。</p>
        <el-progress v-if="uploading" :percentage="uploadPercent" />
        <div class="actions">
          <el-button @click="onCancelRound">取消本轮</el-button>
          <el-button type="primary" :disabled="uploading || stagingImageCount <= 0" @click="goScreenStep">
            下一步：筛图
          </el-button>
        </div>
      </div>

      <!-- 步骤 2 -->
      <div v-else-if="step === 2" class="panel">
        <p class="hint">将用当前模型对本轮新图推理，并划分难例 / 简单例。</p>
        <el-button type="primary" :loading="screening" @click="onScreen">开始筛图</el-button>
        <div v-if="screening" class="screen-prog">
          <el-progress :percentage="Math.round(screenProgress)" />
          <span>{{ screenMessage }}</span>
        </div>
        <div class="actions">
          <el-button :disabled="screening" @click="onCancelRound">取消本轮</el-button>
        </div>
      </div>

      <!-- 步骤 3 -->
      <div v-else-if="step === 3" class="panel">
        <div class="summary" v-if="session?.summary">
          <p>
            共 {{ session.summary.total || 0 }} 张：难例
            <strong>{{ session.summary.hard_count || 0 }}</strong>
            ，简单例
            <strong>{{ session.summary.easy_count || 0 }}</strong>
            （已写草稿 {{ session.summary.written_labels || 0 }} 张）
          </p>
        </div>
        <div class="two-col">
          <div>
            <h3>难例优先（{{ hardItems.length }}）</h3>
            <ul class="list">
              <li v-for="it in hardItems.slice(0, 12)" :key="it.image_name">
                {{ it.image_name }}
                <span class="muted">{{ it.reason }}</span>
              </li>
            </ul>
            <el-button type="primary" :disabled="!hardItems.length" @click="openReview(true)">
              在本页复核难例
            </el-button>
          </div>
          <div>
            <h3>简单例快过（{{ easyItems.length }}）</h3>
            <ul class="list">
              <li v-for="it in easyItems.slice(0, 12)" :key="it.image_name">
                {{ it.image_name }}
                <span class="muted">{{ it.reason }}</span>
              </li>
            </ul>
            <el-button :disabled="!easyItems.length" @click="openReview(false)">在本页检查简单例</el-button>
          </div>
        </div>
        <div class="actions">
          <el-button @click="onCancelRound">取消本轮</el-button>
          <el-button type="success" :loading="retraining" @click="onMergeAndRetrain">
            我已复核完 → 并入原集并准备续训
          </el-button>
        </div>
      </div>

      <!-- 步骤 4 -->
      <div v-else class="panel">
        <p>续训已启动或已准备。可到检测向导查看训练进度。</p>
        <el-button type="primary" @click="backModels">返回模型库</el-button>
      </div>
    </template>
  </section>
</template>

<style scoped>
.al-page {
  max-width: 960px;
  margin: 0 auto;
  padding: 1.25rem 1.5rem 2.5rem;
}
.al-page.reviewing {
  max-width: min(1280px, 100%);
}
.review-shell {
  display: grid;
  gap: 0.75rem;
}
.review-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  border: 1px solid #d7e3ea;
  border-radius: 10px;
  background: #f7fafc;
  color: #4a5c68;
  font-size: 0.9rem;
}
.head {
  display: grid;
  gap: 0.75rem;
  margin-bottom: 1.25rem;
}
.back {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  border: none;
  background: transparent;
  color: var(--brand-deep, #1f4e5f);
  cursor: pointer;
  width: fit-content;
  padding: 0;
  font-size: 0.9rem;
}
h2 {
  margin: 0;
  font-size: 1.45rem;
}
.lead {
  margin: 0.25rem 0 0;
  color: #5a6a78;
}
.steps {
  margin: 1.25rem 0 1.5rem;
}
.panel {
  background: #fff;
  border: 1px solid #e4ebf0;
  border-radius: 12px;
  padding: 1.25rem;
  display: grid;
  gap: 1rem;
}
.hint {
  margin: 0;
  color: #4a5c68;
}
.warn {
  color: #a94442;
  margin: 0;
}
.locked-ds {
  display: grid;
  gap: 0.35rem;
  padding: 0.85rem 1rem;
  border: 1px solid #d7e3ea;
  border-radius: 10px;
  background: #f7fafc;
}
.locked-label {
  font-size: 0.8rem;
  color: #6a7c88;
}
.muted-inline {
  color: #80919d;
  font-size: 0.85rem;
}
.upload-status {
  padding: 0.75rem 1rem;
  border-radius: 10px;
  border: 1px dashed #c5d3dc;
  background: #fafcfd;
  color: #6a7c88;
}
.upload-status.ok {
  border-style: solid;
  border-color: #9ec9b0;
  background: #f3faf6;
  color: #2f5d45;
}
.upload-status strong {
  font-size: 1.05rem;
}
.name-preview {
  margin: 0;
  padding-left: 1.1rem;
  max-height: 140px;
  overflow: auto;
  font-size: 0.85rem;
  color: #4a5c68;
}
.file-btn {
  display: inline-flex;
  width: fit-content;
  cursor: pointer;
  position: relative;
}
.file-btn.disabled {
  cursor: not-allowed;
  pointer-events: none;
}
.file-btn input[type='file'] {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
  width: 100%;
  height: 100%;
}
.field-note {
  margin: 0;
  font-size: 0.8rem;
  color: #80919d;
  line-height: 1.4;
}
.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
}
.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}
@media (max-width: 800px) {
  .two-col {
    grid-template-columns: 1fr;
  }
}
.list {
  margin: 0.5rem 0 0.75rem;
  padding-left: 1.1rem;
  max-height: 220px;
  overflow: auto;
  font-size: 0.88rem;
}
.muted {
  display: block;
  color: #80919d;
  font-size: 0.78rem;
}
.screen-prog {
  display: grid;
  gap: 0.35rem;
}
.summary strong {
  color: var(--brand-deep, #1f4e5f);
}
</style>
