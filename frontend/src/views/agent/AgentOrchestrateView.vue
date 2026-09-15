<script setup lang="ts">
/**
 * AI 流程编排：自然语言生成计划，逐步确认执行。
 * 标注不由 NL 直接生成；抽检门禁必须人工确认。
 * 切换侧栏时 KeepAlive 缓存；计划 ID 写入 session，刷新后可恢复。
 */
defineOptions({ name: 'AgentOrchestrate' })
import { computed, onActivated, onDeactivated, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { MagicStick, VideoPlay, Right, Close } from '@element-plus/icons-vue'
import {
  ackAgentJob,
  cancelAgentPlan,
  createAgentPlan,
  getAgentPlan,
  patchAgentPlan,
  runAgentStep,
  type AgentPlan,
  type AgentStep,
} from '@/api/agent'
import { getJob, listWeights, type WeightItem } from '@/api/tasks'
import { fetchDevices, type DeviceItem } from '@/api/system'
import { listDatasets, type DatasetItem } from '@/api/datasets'
import { agentOrchestrateStateKey } from '@/utils/wizardSession'

const router = useRouter()
const STATE_KEY = agentOrchestrateStateKey()

const loading = ref(false)
const running = ref(false)
const prompt = ref('用当前数据集训练 50 轮，训完后评估并导出 ONNX')
const taskType = ref<'detect' | 'segment' | 'pose'>('detect')
const datasetId = ref<number | null>(null)

const datasets = ref<DatasetItem[]>([])
const plan = ref<AgentPlan | null>(null)
const lastMessage = ref('')
const editEpochs = ref(50)
const editBatch = ref(8)
const editDevice = ref('cpu')
const editWeight = ref('')
const weightOptions = ref<WeightItem[]>([])
const deviceOptions = ref<DeviceItem[]>([{ id: 'cpu', label: 'CPU' }])
const patching = ref(false)
let pollTimer: number | null = null
let restoring = false

const REMOVABLE = new Set(['prelabel', 'review_labels', 'eval', 'export', 'summary'])

const filteredDatasets = computed(() =>
  datasets.value.filter((d) => (d.task_type || 'detect') === taskType.value),
)

const planCompleted = computed(() => plan.value?.status === 'completed')

const configStep = computed(() => plan.value?.steps.find((s) => s.type === 'apply_config') || null)

const planEditable = computed(() => {
  if (!plan.value || planCompleted.value) return false
  return !['cancelled'].includes(plan.value.status)
})

function canRemoveStep(s: AgentStep) {
  if (!planEditable.value) return false
  if (!REMOVABLE.has(s.type)) return false
  return !['completed', 'running'].includes(s.status)
}

const currentStep = computed<AgentStep | null>(() => {
  if (!plan.value?.steps?.length) return null
  const i = plan.value.current_index
  return plan.value.steps[i] || null
})

/** 仅在「写入训练配置」成为当前步时展示微调卡片 */
const canEditConfig = computed(() => {
  if (!plan.value || planCompleted.value) return false
  if (['cancelled'].includes(plan.value.status)) return false
  const s = currentStep.value
  return Boolean(s && s.type === 'apply_config' && !['completed', 'running', 'skipped'].includes(s.status))
})

/** 已完成/已跳过的步骤（当前步之前） */
const pastSteps = computed(() => {
  if (!plan.value?.steps?.length) return []
  const cur = plan.value.current_index
  return plan.value.steps
    .map((s, idx) => ({ s, idx }))
    .filter(({ idx }) => idx < cur)
})

/** 后续尚未开始的步骤 */
const upcomingSteps = computed(() => {
  if (!plan.value?.steps?.length) return []
  if (planCompleted.value) return []
  const cur = plan.value.current_index
  return plan.value.steps
    .map((s, idx) => ({ s, idx }))
    .filter(({ idx }) => idx > cur)
})

const canAct = computed(() => {
  if (!plan.value || running.value) return false
  if (['completed', 'cancelled'].includes(plan.value.status)) return false
  const s = currentStep.value
  if (!s) return false
  if (s.status === 'running' && s.job_id) return false
  return ['ready', 'pending', 'failed'].includes(s.status)
})

function stopPoll() {
  if (pollTimer) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

function saveUiState() {
  if (restoring) return
  try {
    sessionStorage.setItem(
      STATE_KEY,
      JSON.stringify({
        prompt: prompt.value,
        taskType: taskType.value,
        datasetId: datasetId.value,
        planId: plan.value?.id || null,
        lastMessage: lastMessage.value,
      }),
    )
  } catch {
    // 忽略存储失败
  }
}

async function restoreUiState() {
  restoring = true
  try {
    const raw = sessionStorage.getItem(STATE_KEY)
    if (!raw) return
    const data = JSON.parse(raw) as {
      prompt?: string
      taskType?: 'detect' | 'segment' | 'pose'
      datasetId?: number | null
      planId?: string | null
      lastMessage?: string
    }
    if (data.prompt) prompt.value = data.prompt
    if (data.taskType === 'detect' || data.taskType === 'segment' || data.taskType === 'pose') {
      taskType.value = data.taskType
    }
    if (typeof data.datasetId === 'number') datasetId.value = data.datasetId
    if (data.lastMessage) lastMessage.value = data.lastMessage
    await loadDatasets()
    if (data.planId) {
      try {
        const { data: p } = await getAgentPlan(data.planId)
        plan.value = p
        resumeJobPollIfNeeded()
      } catch {
        plan.value = null
      }
    }
  } catch {
    // 忽略损坏状态
  } finally {
    restoring = false
  }
}

function resumeJobPollIfNeeded() {
  const runningStep = plan.value?.steps.find((s) => s.status === 'running' && s.job_id)
  if (runningStep?.job_id) startJobPoll(runningStep.job_id)
}

async function loadDatasets() {
  const { data } = await listDatasets(taskType.value)
  datasets.value = data
  if (!filteredDatasets.value.some((d) => d.id === datasetId.value)) {
    datasetId.value = filteredDatasets.value[0]?.id ?? null
  }
}

async function loadWeightsForType(tt: string) {
  try {
    const { data } = await listWeights(tt)
    weightOptions.value = data
  } catch {
    weightOptions.value = []
  }
}

async function loadDevices() {
  try {
    const { data } = await fetchDevices()
    deviceOptions.value = data.devices?.length ? data.devices : [{ id: 'cpu', label: 'CPU' }]
    if (!deviceOptions.value.some((d) => d.id === editDevice.value)) {
      editDevice.value = deviceOptions.value[0]?.id || 'cpu'
    }
  } catch {
    deviceOptions.value = [{ id: 'cpu', label: 'CPU' }]
    editDevice.value = 'cpu'
  }
}

function syncEditFromPlan() {
  const s = configStep.value
  if (!s) return
  editEpochs.value = Number(s.params.epochs) || 50
  editBatch.value = Number(s.params.batch) || 8
  const rawDevice = String(s.params.device || 'cpu')
  // 兼容旧计划里的 "0" → cuda:0（若本机确有 GPU）
  if (rawDevice === '0' && deviceOptions.value.some((d) => d.id === 'cuda:0')) {
    editDevice.value = 'cuda:0'
  } else if (deviceOptions.value.some((d) => d.id === rawDevice)) {
    editDevice.value = rawDevice
  } else {
    editDevice.value = deviceOptions.value[0]?.id || 'cpu'
  }
  editWeight.value = String(s.params.pretrained_weight || '')
}

watch(
  () => plan.value?.id,
  async () => {
    if (!plan.value) return
    await loadDevices()
    syncEditFromPlan()
    await loadWeightsForType(plan.value.task_type)
  },
)

watch(canEditConfig, async (show) => {
  if (!show) return
  await loadDevices()
  syncEditFromPlan()
  if (plan.value?.task_type) await loadWeightsForType(plan.value.task_type)
})

async function onApplyConfigPatch() {
  if (!plan.value || !canEditConfig.value) return
  patching.value = true
  try {
    const { data } = await patchAgentPlan(plan.value.id, {
      epochs: editEpochs.value,
      batch: editBatch.value,
      device: editDevice.value,
      pretrained_weight: editWeight.value || undefined,
    })
    plan.value = data
    syncEditFromPlan()
    saveUiState()
    ElMessage.success('配置已更新')
  } catch {
    // 拦截器
  } finally {
    patching.value = false
  }
}

async function onRemoveStep(stepId: string) {
  if (!plan.value) return
  patching.value = true
  try {
    const { data } = await patchAgentPlan(plan.value.id, { remove_step_ids: [stepId] })
    plan.value = data
    syncEditFromPlan()
    saveUiState()
    ElMessage.success('已移除该步骤')
  } catch {
    //
  } finally {
    patching.value = false
  }
}

watch(taskType, () => {
  if (restoring) return
  void loadDatasets().then(saveUiState)
})

watch([prompt, datasetId, lastMessage], () => saveUiState())
watch(
  () => plan.value?.id,
  () => saveUiState(),
)
watch(
  () => plan.value?.current_index,
  () => saveUiState(),
)
watch(
  () => plan.value?.status,
  () => saveUiState(),
)

function statusLabel(s: string) {
  const map: Record<string, string> = {
    pending: '等待',
    ready: '待确认',
    running: '进行中',
    completed: '完成',
    failed: '失败',
    skipped: '已跳过',
  }
  return map[s] || s
}

function stepClass(s: AgentStep, idx: number) {
  return {
    active: plan.value && idx === plan.value.current_index,
    done: s.status === 'completed' || s.status === 'skipped',
    fail: s.status === 'failed',
    run: s.status === 'running',
  }
}

async function onCreatePlan() {
  if (!datasetId.value) {
    ElMessage.warning('请先选择数据集')
    return
  }
  if (!prompt.value.trim()) {
    ElMessage.warning('请输入需求描述')
    return
  }
  loading.value = true
  stopPoll()
  lastMessage.value = ''
  try {
    const { data } = await createAgentPlan({
      prompt: prompt.value.trim(),
      task_type: taskType.value,
      dataset_id: datasetId.value,
    })
    plan.value = data
    saveUiState()
    ElMessage.success(data.source === 'llm' ? '已生成计划（大模型）' : '已生成计划（模板）')
  } catch {
    // 拦截器
  } finally {
    loading.value = false
  }
}

function startJobPoll(jobId: number) {
  stopPoll()
  pollTimer = window.setInterval(async () => {
    try {
      const { data: job } = await getJob(jobId)
      lastMessage.value = job.message || job.status
      if (!['pending', 'running'].includes(job.status)) {
        stopPoll()
        const { data } = await ackAgentJob(plan.value!.id)
        plan.value = data.plan
        lastMessage.value = data.message || job.message
        saveUiState()
        if (job.status === 'completed') {
          ElMessage.success('步骤 Job 已完成')
        } else {
          ElMessage.error(job.message || 'Job 失败/取消')
        }
      }
    } catch {
      // 下次再试
    }
  }, 400)
}

async function onRun(action: 'run' | 'skip' = 'run') {
  if (!plan.value || !canAct.value) return
  running.value = true
  try {
    const { data } = await runAgentStep(plan.value.id, action)
    plan.value = data.plan
    lastMessage.value = data.message
    saveUiState()
    const runningStep = plan.value.steps.find((s) => s.status === 'running' && s.job_id)
    if (runningStep?.job_id) {
      ElMessage.info(data.message || 'Job 已启动')
      startJobPoll(runningStep.job_id)
    } else if (action === 'skip') {
      ElMessage.success(data.message || '已跳过')
    } else {
      ElMessage.success(data.message || '步骤完成')
    }
  } catch {
    if (plan.value) {
      try {
        const { data: p } = await getAgentPlan(plan.value.id)
        plan.value = p
        saveUiState()
      } catch {
        /* ignore */
      }
    }
  } finally {
    running.value = false
  }
}

async function onCancel() {
  if (!plan.value) return
  stopPoll()
  try {
    const { data } = await cancelAgentPlan(plan.value.id)
    plan.value = data
    saveUiState()
    ElMessage.warning('计划已取消')
  } catch {
    //
  }
}

function goAnnotate() {
  if (!datasetId.value) {
    ElMessage.warning('请先选择数据集')
    return
  }
  saveUiState()
  const path =
    taskType.value === 'segment'
      ? '/app/segment/wizard'
      : taskType.value === 'pose'
        ? '/app/pose/wizard'
        : '/app/detect/wizard'
  // step=2：直接进入标注步；返回编排页后再点「重新检查」
  router.push({
    path,
    query: { datasetId: String(datasetId.value), step: '2' },
  })
}

function goModels() {
  const tt = plan.value?.task_type || taskType.value
  router.push({ path: '/app/resources/models', query: { type: tt } })
}

function goInfer() {
  const tt = plan.value?.task_type || taskType.value
  router.push({ path: '/app/resources/infer', query: { type: tt } })
}

const showLabelRemediation = computed(() => {
  const s = currentStep.value
  if (!s || !plan.value) return false
  if (s.status !== 'failed') return false
  return s.type === 'check_labels' || s.type === 'review_labels'
})

async function onRecheckLabels() {
  await onRun('run')
}

onMounted(async () => {
  await restoreUiState()
  if (!datasets.value.length) await loadDatasets()
})

onActivated(() => {
  // 从其他页回来：刷新计划并恢复 Job 轮询
  void (async () => {
    if (plan.value?.id) {
      try {
        const { data: p } = await getAgentPlan(plan.value.id)
        plan.value = p
      } catch {
        /* 计划可能已失效，保留本地视图 */
      }
    }
    resumeJobPollIfNeeded()
  })()
})

onDeactivated(() => {
  saveUiState()
  stopPoll()
})

onUnmounted(() => stopPoll())
</script>

<template>
  <section class="page" v-loading="loading">
    <header class="hero">
      <div class="hero-text">
        <p class="eyebrow">AI Agent</p>
        <h2>流程编排</h2>
        <p class="desc">
          用自然语言生成训练计划，逐步确认后调用现有能力。标注不会由一句话直接生成；预标注后须人工抽检门禁。
        </p>
      </div>
    </header>

    <div class="layout">
      <aside class="panel">
        <h3>需求</h3>
        <div class="field">
          <label>任务类型</label>
          <div class="pills">
            <button type="button" class="pill" :class="{ active: taskType === 'detect' }" @click="taskType = 'detect'">
              目标检测
            </button>
            <button
              type="button"
              class="pill"
              :class="{ active: taskType === 'segment' }"
              @click="taskType = 'segment'"
            >
              实例分割
            </button>
            <button type="button" class="pill" :class="{ active: taskType === 'pose' }" @click="taskType = 'pose'">
              姿态估计
            </button>
          </div>
        </div>
        <div class="field">
          <label>数据集</label>
          <el-select v-model="datasetId" filterable placeholder="选择数据集" class="full">
            <el-option v-for="d in filteredDatasets" :key="d.id" :label="d.name" :value="d.id" />
          </el-select>
          <p v-if="!filteredDatasets.length" class="hint">该类型下暂无数据集</p>
        </div>
        <div class="field">
          <label>需求描述</label>
          <el-input
            v-model="prompt"
            type="textarea"
            :rows="5"
            placeholder="例如：训练 50 轮，训完后评估并导出 ONNX；需要预标注就写上「预标注」"
          />
          <p class="hint">轮数、是否评估/导出/预标注会从这段描述里解析，无需再单独勾选。</p>
        </div>
        <el-button type="primary" class="full-btn" :icon="MagicStick" :loading="loading" @click="onCreatePlan">
          生成计划
        </el-button>
        <el-button class="full-btn" @click="goAnnotate">去标注页</el-button>
      </aside>

      <div class="panel main">
        <div class="main-head">
          <h3>执行计划</h3>
          <div v-if="plan" class="meta">
            <span>来源：{{ plan.source === 'llm' ? '大模型' : '模板' }}</span>
            <span>·</span>
            <span>{{ plan.status }}</span>
            <span v-if="plan.task_id">· 任务 #{{ plan.task_id }}</span>
          </div>
        </div>

        <p v-if="plan?.summary" class="summary">{{ plan.summary }}</p>
        <p v-else class="empty">生成计划后，在此逐步确认执行。</p>

        <ul v-if="plan?.hints?.length" class="hints">
          <li v-for="(h, i) in plan.hints" :key="i">{{ h }}</li>
        </ul>

        <div v-if="plan" class="steps-flow">
          <!-- 已完成：紧凑一行 -->
          <ul v-if="pastSteps.length" class="steps-past">
            <li v-for="{ s, idx } in pastSteps" :key="s.id" class="step-past">
              <span class="past-idx">{{ idx + 1 }}.</span>
              <span class="past-title">{{ s.title }}</span>
              <span class="badge">{{ statusLabel(s.status) }}</span>
            </li>
          </ul>

          <!-- 当前步：完整卡片 -->
          <div
            v-if="currentStep && !planCompleted"
            class="step current-focus"
            :class="stepClass(currentStep, plan.current_index)"
          >
            <div class="step-top">
              <strong>{{ plan.current_index + 1 }}. {{ currentStep.title }}</strong>
              <span class="badge">{{ statusLabel(currentStep.status) }}</span>
              <span v-if="currentStep.human_gate" class="gate">需确认</span>
              <el-button
                v-if="canRemoveStep(currentStep)"
                link
                type="danger"
                size="small"
                :loading="patching"
                @click="onRemoveStep(currentStep.id)"
              >
                移除
              </el-button>
            </div>
            <p class="step-desc">{{ currentStep.description }}</p>

            <!-- 写入训练配置：内嵌微调表单 -->
            <div v-if="canEditConfig" class="tune-card">
              <div class="tune-row">
                <label>Epochs</label>
                <el-input-number
                  v-model="editEpochs"
                  class="tune-num"
                  :min="1"
                  :max="500"
                  size="small"
                  controls-position="right"
                />
                <label>Batch</label>
                <el-input-number
                  v-model="editBatch"
                  class="tune-num"
                  :min="1"
                  :max="128"
                  size="small"
                  controls-position="right"
                />
                <label>设备</label>
                <el-select v-model="editDevice" class="tune-device" size="small" placeholder="探测">
                  <el-option
                    v-for="d in deviceOptions"
                    :key="d.id"
                    :label="d.label"
                    :value="d.id"
                  />
                </el-select>
                <label>权重</label>
                <el-select v-model="editWeight" class="tune-weight" filterable size="small">
                  <el-option v-for="w in weightOptions" :key="w.name" :label="w.name" :value="w.name" />
                </el-select>
                <el-button type="primary" size="small" :loading="patching" @click="onApplyConfigPatch">
                  应用
                </el-button>
              </div>
              <p class="hint">确认参数后点「应用」写入本步；再点下方「执行当前步骤」生效。</p>
            </div>

            <p v-if="currentStep.error" class="err">{{ currentStep.error }}</p>
            <div
              v-if="
                currentStep.status === 'failed' &&
                (currentStep.type === 'check_labels' || currentStep.type === 'review_labels')
              "
              class="remedy"
            >
              <el-button type="primary" size="small" @click="goAnnotate">去标注</el-button>
              <el-button type="warning" plain size="small" :loading="running" @click="onRecheckLabels">
                重新检查
              </el-button>
              <span class="hint">补全类别与标注后，点「重新检查」继续编排</span>
            </div>
            <p v-if="currentStep.job_id" class="hint">Job #{{ currentStep.job_id }}</p>
          </div>

          <!-- 后续：标题摘要，可移除可选步骤 -->
          <div v-if="upcomingSteps.length" class="steps-upcoming">
            <div class="upcoming-label">后续 {{ upcomingSteps.length }} 步</div>
            <ul class="upcoming-list">
              <li v-for="{ s, idx } in upcomingSteps" :key="s.id" class="upcoming-item">
                <span class="upcoming-title">{{ idx + 1 }}. {{ s.title }}</span>
                <button
                  v-if="canRemoveStep(s)"
                  type="button"
                  class="upcoming-remove"
                  :disabled="patching"
                  @click="onRemoveStep(s.id)"
                >
                  移除
                </button>
              </li>
            </ul>
          </div>
        </div>

        <div v-if="plan && !planCompleted" class="actions">
          <el-button
            type="primary"
            :icon="VideoPlay"
            :disabled="!canAct"
            :loading="running"
            @click="onRun('run')"
          >
            {{ currentStep?.human_gate ? '确认并执行' : '执行当前步骤' }}
          </el-button>
          <el-button
            :icon="Right"
            :disabled="!canAct || currentStep?.type === 'check_labels' || currentStep?.type === 'ensure_task'"
            @click="onRun('skip')"
          >
            跳过
          </el-button>
          <el-button
            :icon="Close"
            :disabled="plan.status === 'cancelled'"
            @click="onCancel"
          >
            取消计划
          </el-button>
        </div>

        <div v-if="planCompleted" class="done-card">
          <h4>编排完成</h4>
          <p class="done-text">
            {{ plan?.summary || '本轮步骤已全部完成。' }}
          </p>
          <p v-if="plan?.task_id" class="done-meta">训练任务 #{{ plan.task_id }}</p>
          <div class="done-actions">
            <el-button type="primary" @click="goModels">去模型库</el-button>
            <el-button @click="goInfer">去推理试用</el-button>
            <el-button plain @click="goAnnotate">回标注页</el-button>
          </div>
        </div>

        <p v-if="lastMessage" class="last-msg">{{ lastMessage }}</p>
        <p v-if="currentStep?.type === 'review_labels' && !planCompleted" class="hint review-tip">
          请先到标注页抽检，确认质量后再点「确认并执行」。
        </p>
        <p v-if="showLabelRemediation" class="hint review-tip">
          标注不足或缺少类别时，可点步骤内的「去标注」，完成后再「重新检查」。
        </p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.page {
  width: 100%;
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
  max-width: 42rem;
}
.layout {
  display: grid;
  grid-template-columns: minmax(260px, 340px) 1fr;
  gap: 0.9rem;
  align-items: start;
}
.panel {
  background: var(--surface-elevated);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  padding: 1rem;
}
.panel h3 {
  margin: 0 0 0.85rem;
  font-size: 1rem;
  color: var(--brand-deep);
}
.field {
  margin-bottom: 0.85rem;
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
.full-btn {
  width: 100%;
  margin: 0 0 0.5rem !important;
}
.pills {
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
.checks {
  display: grid;
  gap: 0.25rem;
  margin-bottom: 0.85rem;
}
.hint {
  margin: 0.35rem 0 0;
  font-size: 0.75rem;
  color: var(--ink-faint);
}
.main-head {
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  align-items: baseline;
  flex-wrap: wrap;
}
.meta {
  font-size: 0.8rem;
  color: var(--ink-faint);
}
.summary {
  margin: 0.5rem 0 0.85rem;
  color: var(--ink-muted);
  line-height: 1.5;
}
.empty {
  color: var(--ink-faint);
}
.hints {
  margin: 0 0 0.75rem;
  padding: 0.55rem 0.75rem 0.55rem 1.25rem;
  border-radius: 8px;
  background: #fff8e8;
  border: 1px solid #f0d9a8;
  color: #8a6418;
  font-size: 0.82rem;
  line-height: 1.45;
}
.tune-card {
  margin: 0.65rem 0 0;
  padding: 0.65rem 0.7rem;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: #f7fafb;
}
.tune-card .hint {
  margin: 0.15rem 0 0;
}
.tune-row {
  display: grid;
  grid-template-columns: auto 1fr auto 1fr auto 1fr auto 1.2fr auto;
  align-items: center;
  column-gap: 0.4rem;
  row-gap: 0.35rem;
  margin-bottom: 0.4rem;
}
.tune-row label {
  font-size: 0.8rem;
  color: var(--ink-muted);
  white-space: nowrap;
}
.tune-num,
.tune-device,
.tune-weight {
  width: 100%;
  min-width: 0;
}
.tune-num :deep(.el-input-number),
.tune-num.el-input-number {
  width: 100%;
}
.tune-device :deep(.el-select__wrapper),
.tune-weight :deep(.el-select__wrapper) {
  width: 100%;
}
.tune-row .el-button {
  justify-self: end;
  white-space: nowrap;
}
@media (max-width: 900px) {
  .tune-row {
    grid-template-columns: auto 1fr auto 1fr;
  }
  .tune-row .el-button {
    grid-column: 1 / -1;
    justify-self: start;
  }
}
.steps-flow {
  display: grid;
  gap: 0.55rem;
}
.steps-past {
  list-style: none;
  margin: 0;
  padding: 0.35rem 0.55rem;
  border-radius: 8px;
  background: #f3f6f8;
  border: 1px dashed var(--line);
  display: grid;
  gap: 0.25rem;
}
.step-past {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.8rem;
  color: var(--ink-muted);
}
.past-idx {
  opacity: 0.7;
}
.past-title {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.step.current-focus {
  border-color: var(--brand-soft);
  background: var(--brand-mist);
  box-shadow: 0 1px 0 rgba(20, 60, 80, 0.04);
}
.steps-upcoming {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  padding: 0.55rem 0.65rem;
  border-radius: 8px;
  border: 1px dashed #d7e0e6;
  background: #fafcfc;
}
.upcoming-label {
  font-size: 0.78rem;
  line-height: 1.2;
  color: var(--ink-muted);
}
.upcoming-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem;
}
.upcoming-item {
  display: inline-flex;
  align-items: center;
  height: 28px;
  padding: 0 0.55rem;
  border-radius: 6px;
  background: #eef3f6;
  color: #4a5c6a;
  font-size: 0.8rem;
  line-height: 1;
  gap: 0.35rem;
}
.upcoming-title {
  white-space: nowrap;
  line-height: 1;
}
.upcoming-remove {
  border: 0;
  background: transparent;
  padding: 0;
  margin: 0;
  color: #c45656;
  font-size: 0.78rem;
  line-height: 1;
  cursor: pointer;
  white-space: nowrap;
}
.upcoming-remove:hover:not(:disabled) {
  text-decoration: underline;
}
.upcoming-remove:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}
.steps {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.55rem;
}
.step {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.7rem 0.85rem;
  background: #fafcfc;
}
.step.active {
  border-color: var(--brand-soft);
  background: var(--brand-mist);
}
.step.done {
  opacity: 0.78;
}
.step.fail {
  border-color: #e8a0a0;
  background: #fff5f5;
}
.step.run {
  border-color: #a0cfff;
}
.step-top {
  display: flex;
  gap: 0.45rem;
  align-items: center;
  flex-wrap: wrap;
}
.badge {
  font-size: 0.72rem;
  padding: 0.1rem 0.4rem;
  border-radius: 999px;
  background: #eef2f6;
  color: #3d4f5f;
}
.gate {
  font-size: 0.72rem;
  color: #b88230;
}
.step-desc {
  margin: 0.35rem 0 0;
  font-size: 0.85rem;
  color: var(--ink-muted);
  line-height: 1.45;
}
.err {
  margin: 0.35rem 0 0;
  color: #c45656;
  font-size: 0.82rem;
}
.remedy {
  margin-top: 0.55rem;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.45rem;
}
.remedy .hint {
  margin: 0;
  flex: 1 1 12rem;
}
.done-card {
  margin-top: 1rem;
  padding: 1rem 1.1rem;
  border-radius: 12px;
  border: 1px solid rgba(61, 155, 143, 0.35);
  background: linear-gradient(135deg, rgba(232, 244, 242, 0.9), rgba(255, 255, 255, 0.7));
}
.done-card h4 {
  margin: 0;
  font-size: 1.05rem;
  color: var(--brand-deep);
}
.done-text {
  margin: 0.45rem 0 0;
  color: var(--ink-muted);
  line-height: 1.5;
  font-size: 0.9rem;
}
.done-meta {
  margin: 0.35rem 0 0;
  font-size: 0.85rem;
  color: var(--brand-soft);
  font-weight: 600;
}
.done-actions {
  margin-top: 0.75rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}
.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 1rem;
}
.last-msg {
  margin: 0.75rem 0 0;
  font-size: 0.85rem;
  color: var(--brand-deep);
}
.review-tip {
  margin-top: 0.5rem;
}
@media (max-width: 900px) {
  .layout {
    grid-template-columns: 1fr;
  }
}
</style>
