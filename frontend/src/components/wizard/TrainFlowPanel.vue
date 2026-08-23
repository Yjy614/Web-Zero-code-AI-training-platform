<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import { useAppStore } from '@/stores/app'
import { useAuthStore } from '@/stores/auth'
import { fetchDevices, type DeviceItem } from '@/api/system'
import {
  cancelJob,
  createTask,
  downloadTaskFile,
  fetchAiAdvice,
  getJob,
  getTask,
  listTasks,
  listTaskJobs,
  listWeights,
  patchTask,
  splitTask,
  startEval,
  startExport,
  startTrain,
  type JobItem,
  type TaskItem,
  type TrainConfig,
  type WeightItem,
} from '@/api/tasks'
import SplitRatioBar from '@/components/wizard/SplitRatioBar.vue'

const props = defineProps<{
  step: number
  datasetId: number
  datasetName: string
  /** 当前数据集图片数，用于划分预览 */
  imageCount?: number
}>()

const emit = defineEmits<{
  'update:step': [step: number]
}>()

const appStore = useAppStore()
const authStore = useAuthStore()
/** 演示相关文案仅管理员可见 */
const showDemoHint = computed(() => appStore.demoMode && authStore.isAdmin)

const loading = ref(false)
const task = ref<TaskItem | null>(null)
const taskName = ref('')
const weights = ref<WeightItem[]>([])
const deviceOptions = ref<DeviceItem[]>([{ id: 'cpu', label: 'CPU' }])
const config = reactive<TrainConfig>({
  train_ratio: 0.7,
  val_ratio: 0.2,
  test_ratio: 0.1,
  epochs: 50,
  batch: 8,
  imgsz: 640,
  train_strategy: 'stable',
  device: 'cpu',
  pretrained_weight: '',
  augment: true,
})
/** batch / imgsz / 训练策略默认值；未勾选自定义时强制使用 */
const DEFAULT_BATCH = 8
const DEFAULT_IMGSZ = 640
const DEFAULT_STRATEGY = 'stable'
const STRATEGY_OPTIONS = [
  {
    id: 'stable',
    label: '稳健（默认）',
    tip: '常用学习率与线性衰减，适合大多数工业检测首训。',
  },
  {
    id: 'fast',
    label: '快速收敛',
    tip: '余弦学习率 + 较短预热，前期下降更快；抖动大时改回稳健。',
  },
  {
    id: 'finetune',
    label: '细调（小学习率）',
    tip: '更小学习率，适合继续微调或小样本，降低过拟合风险。',
  },
] as const
const customBatchImgsz = ref(false)
/** 勾选自定义时记住用户上次改过的值 */
const customBatchDraft = ref(DEFAULT_BATCH)
const customImgszDraft = ref(DEFAULT_IMGSZ)
const customStrategyDraft = ref(DEFAULT_STRATEGY)
let syncingHyperUi = false

/** 可选超参：默认 / 自定义（绑定单选） */
const hyperMode = computed({
  get: () => (customBatchImgsz.value ? 'custom' : 'default'),
  set: (v: string) => {
    customBatchImgsz.value = v === 'custom'
  },
})

const strategyLabel = computed(() => {
  const hit = STRATEGY_OPTIONS.find((s) => s.id === config.train_strategy)
  return hit?.label || '稳健（默认）'
})

const strategyTip = computed(() => {
  const hit = STRATEGY_OPTIONS.find((s) => s.id === config.train_strategy)
  return hit?.tip || ''
})
const splitInfo = ref('')
const job = ref<JobItem | null>(null)
const pollTimer = ref<number | null>(null)
/** 正在点击「开始训练」时，禁止 resume 把上一轮结果写回来 */
let resumePaused = false
const chartRef = ref<HTMLDivElement | null>(null)
/** 本次向导流程内的训练曲线（整条流程结束前保留，仅重新开训时清空） */
const trainHistory = ref<Array<{ epoch: number; loss: number; map50: number }>>([])
const aiAdviceLoading = ref(false)
const aiAdviceText = ref('')
const aiAdviceSource = ref('')

/** 展示前再清一次 Markdown 残留（兼容旧缓存建议） */
const aiAdviceDisplay = computed(() => {
  let t = (aiAdviceText.value || '').replace(/\r\n/g, '\n')
  t = t.replace(/\*\*/g, '').replace(/__/g, '')
  t = t.replace(/(?<!\*)\*(?!\*)/g, '')
  t = t.replace(/^#{1,6}\s*/gm, '')
  t = t.replace(/^\s*[-•]\s+/gm, '')
  return t.trim()
})
let chart: echarts.ECharts | null = null
let syncingSplit = false

/** 仅列出权重仓库中真实已上传文件 */
const weightOptions = computed(() => weights.value.map((w) => w.name))

/** 清除历史占位/乱码权重名，只保留合法文件名 */
function sanitizeWeightName(name: string) {
  const raw = (name || '').trim()
  if (!raw) return ''
  if (raw.includes('占位') || raw.includes('未上传')) return ''
  const m = raw.match(/[\w.-]+\.(?:pt|pth|onnx)/i)
  if (!m) return ''
  const fname = m[0]
  // 带括号说明、或文件名后还有多余字符（含历史乱码）→ 视为占位，丢弃
  if (raw !== fname) return ''
  return fname
}

function normalizeSplit() {
  if (syncingSplit) return
  syncingSplit = true
  try {
    let tr = Math.min(0.9, Math.max(0.5, Number(config.train_ratio) || 0.7))
    let vr = Math.min(0.4, Math.max(0.05, Number(config.val_ratio) || 0.2))
    tr = Math.round(tr * 100) / 100
    vr = Math.round(vr * 100) / 100
    if (tr + vr > 0.95) {
      vr = Math.round((0.95 - tr) * 100) / 100
      if (vr < 0.05) {
        vr = 0.05
        tr = 0.9
      }
    }
    const te = Math.round((1 - tr - vr) * 100) / 100
    config.train_ratio = tr
    config.val_ratio = vr
    config.test_ratio = Math.max(0, te)
  } finally {
    syncingSplit = false
  }
}

const history = computed(() => trainHistory.value)

const showTrainChart = computed(() => trainHistory.value.length > 0)

const trainJob = computed(() => (job.value?.type === 'train' ? job.value : null))
const lastTrainStatus = ref<{ message: string; status: string; progress: number } | null>(null)

function syncTrainHistoryFromJob(j: JobItem | null) {
  if (!j || j.type !== 'train') return
  const h = j.result?.history as Array<{ epoch: number; loss: number; map50: number }> | undefined
  // 必须整表替换：新 Job 尚无 history 时也要清空，避免残留上一次曲线
  trainHistory.value = Array.isArray(h) ? h : []
  lastTrainStatus.value = {
    message: j.message,
    status: j.status,
    progress: j.progress,
  }
}

const evalMetrics = computed(() => {
  const m = (job.value?.type === 'eval' ? job.value.result?.metrics : null) || task.value?.metrics || {}
  return m as Record<string, unknown>
})

const exportFiles = computed(() => {
  const files = job.value?.result?.files
  return Array.isArray(files) ? (files as Array<{ name: string; format: string; demo?: boolean }>) : []
})

async function ensureTask() {
  if (task.value && task.value.dataset_id === props.datasetId) return task.value
  const { data: list } = await listTasks()
  const found = list.find((t) => t.dataset_id === props.datasetId)
  if (found) {
    task.value = found
    applyConfig(found)
    return found
  }
  if (!taskName.value.trim()) {
    taskName.value = `${props.datasetName}_train`
  }
  const { data } = await createTask(taskName.value.trim(), props.datasetId)
  task.value = data
  applyConfig(data)
  return data
}

function applyConfig(t: TaskItem) {
  taskName.value = t.name
  const c = t.config || {}
  config.train_ratio = Number(c.train_ratio ?? 0.7)
  config.val_ratio = Number(c.val_ratio ?? 0.2)
  config.test_ratio = Number(c.test_ratio ?? Math.max(0, 1 - config.train_ratio - config.val_ratio))
  config.epochs = Number(c.epochs ?? 50)
  config.batch = Number(c.batch ?? DEFAULT_BATCH)
  config.imgsz = Number(c.imgsz ?? DEFAULT_IMGSZ)
  const strategyRaw = String(c.train_strategy ?? DEFAULT_STRATEGY)
  config.train_strategy = STRATEGY_OPTIONS.some((s) => s.id === strategyRaw)
    ? strategyRaw
    : DEFAULT_STRATEGY
  config.device = String(c.device ?? 'cpu')
  config.pretrained_weight = sanitizeWeightName(String(c.pretrained_weight ?? ''))
  config.augment = Boolean(c.augment ?? true)
  normalizeSplit()
  // 与默认值不同则视为已开启自定义
  syncingHyperUi = true
  const isCustom =
    config.batch !== DEFAULT_BATCH ||
    config.imgsz !== DEFAULT_IMGSZ ||
    config.train_strategy !== DEFAULT_STRATEGY
  customBatchImgsz.value = isCustom
  customBatchDraft.value = config.batch
  customImgszDraft.value = config.imgsz
  customStrategyDraft.value = config.train_strategy
  if (!isCustom) {
    config.batch = DEFAULT_BATCH
    config.imgsz = DEFAULT_IMGSZ
    config.train_strategy = DEFAULT_STRATEGY
  }
  syncingHyperUi = false
}

watch(
  () => [config.train_ratio, config.val_ratio],
  () => normalizeSplit(),
)

watch(customBatchImgsz, (on) => {
  if (syncingHyperUi) return
  if (on) {
    config.batch = customBatchDraft.value
    config.imgsz = customImgszDraft.value
    config.train_strategy = customStrategyDraft.value
  } else {
    customBatchDraft.value = config.batch
    customImgszDraft.value = config.imgsz
    customStrategyDraft.value = config.train_strategy
    config.batch = DEFAULT_BATCH
    config.imgsz = DEFAULT_IMGSZ
    config.train_strategy = DEFAULT_STRATEGY
  }
})

watch(
  () => [config.batch, config.imgsz, config.train_strategy],
  () => {
    if (!customBatchImgsz.value || syncingHyperUi) return
    customBatchDraft.value = config.batch
    customImgszDraft.value = config.imgsz
    customStrategyDraft.value = config.train_strategy
  },
)

/** 将当前表单写入任务配置（不含 loading / 提示） */
async function persistConfig() {
  normalizeSplit()
  if (!customBatchImgsz.value) {
    config.batch = DEFAULT_BATCH
    config.imgsz = DEFAULT_IMGSZ
    config.train_strategy = DEFAULT_STRATEGY
  }
  config.pretrained_weight = sanitizeWeightName(config.pretrained_weight)
  const t = await ensureTask()
  const { data } = await patchTask(t.id, { config: { ...config }, step: 4 })
  task.value = data
  return data
}

/** 下一步：自动保存配置并划分数据集，再进入训练步 */
async function goNextToTrain() {
  loading.value = true
  try {
    await persistConfig()
    const t = await ensureTask()
    const { data } = await splitTask(t.id)
    splitInfo.value = `训练集 ${data.train} · 验证集 ${data.val} · 测试集 ${data.test ?? 0}`
    const refreshed = await getTask(t.id)
    task.value = refreshed.data
    emit('update:step', 4)
  } catch {
    // 错误由 http 拦截器提示；停留在配置步
  } finally {
    loading.value = false
  }
}

function stopPoll() {
  if (pollTimer.value) {
    window.clearInterval(pollTimer.value)
    pollTimer.value = null
  }
}

async function pollJob(jobId: number) {
  stopPoll()
  // 立刻拉一次，避免等第一个 interval 才看到进度
  try {
    const { data } = await getJob(jobId)
    job.value = data
    syncTrainHistoryFromJob(data)
    if (props.step === 4) void scheduleRenderChart()
    if (['completed', 'failed', 'cancelled'].includes(data.status)) {
      if (task.value) {
        const refreshed = await getTask(task.value.id)
        task.value = refreshed.data
      }
      return
    }
  } catch {
    // 首次拉取失败仍启动轮询
  }
  pollTimer.value = window.setInterval(async () => {
    try {
      const { data } = await getJob(jobId)
      job.value = data
      syncTrainHistoryFromJob(data)
      if (props.step === 4) void scheduleRenderChart()
      if (['completed', 'failed', 'cancelled'].includes(data.status)) {
        stopPoll()
        if (task.value) {
          const refreshed = await getTask(task.value.id)
          task.value = refreshed.data
        }
        if (data.status === 'completed') ElMessage.success(data.message || '完成')
        if (data.status === 'failed') ElMessage.error(data.message || '失败')
        if (data.status === 'cancelled') ElMessage.warning(data.message || '训练已停止')
      }
    } catch {
      // 轮询失败不打断
    }
  }, 1200)
}

/**
 * 仅恢复「进行中」的训练 Job（刷新 / 从标注返回后轮询会丢）。
 * 已完成的上一轮结果不自动带回，避免新开向导进入训练步仍显示旧曲线。
 * 本会话内步骤 4~7 之间切换时，曲线仍由内存中的 trainHistory 保留。
 */
async function resumeActiveTrainJob() {
  if (resumePaused) return
  try {
    const t = await ensureTask()
    const { data: jobs } = await listTaskJobs(t.id, 'train')
    const active = jobs.find((j) => j.status === 'running' || j.status === 'pending')
    if (!active) return
    // 已在轮询同一 Job 则不重复开
    if (pollTimer.value && job.value?.id === active.id) {
      syncTrainHistoryFromJob(active)
      return
    }
    job.value = active
    syncTrainHistoryFromJob(active)
    await pollJob(active.id)
    if (props.step === 4) await scheduleRenderChart()
  } catch {
    // 恢复失败不影响继续操作
  }
}

async function onTrain() {
  loading.value = true
  // 立刻清空上一轮展示，并阻止异步 resume 抢写回来
  resumePaused = true
  stopPoll()
  trainHistory.value = []
  lastTrainStatus.value = null
  job.value = null
  chart?.dispose()
  chart = null
  try {
    await persistConfig()
    const t = await ensureTask()
    if (!(await getTask(t.id)).data.status || true) {
      // 确保已划分
      try {
        await splitTask(t.id)
      } catch {
        // 可能已划分
      }
    }
    const { data } = await startTrain(t.id)
    job.value = data
    syncTrainHistoryFromJob(data)
    await scheduleRenderChart()
    await pollJob(data.id)
  } catch {
    // 开训失败：保持清空，避免又显示上一轮完成态
    trainHistory.value = []
    lastTrainStatus.value = null
    job.value = null
  } finally {
    resumePaused = false
    loading.value = false
  }
}

async function onCancel() {
  if (!job.value) return
  try {
    await cancelJob(job.value.id)
    // 本地立即反馈：真实训练需等当前 epoch 结束才会停下
    if (job.value.status === 'running') {
      job.value = {
        ...job.value,
        message: '正在停止：将在当前 epoch 结束后停止，请稍候…',
      }
      lastTrainStatus.value = {
        message: '正在停止：将在当前 epoch 结束后停止，请稍候…',
        status: 'running',
        progress: job.value.progress,
      }
    }
    ElMessage.info('已请求停止。训练不会瞬间中断，将在当前 epoch 结束后停止。')
  } catch {
    // 错误由拦截器提示
  }
}

async function onEval() {
  if (!task.value) await ensureTask()
  loading.value = true
  try {
    const { data } = await startEval(task.value!.id)
    job.value = data
    await pollJob(data.id)
  } finally {
    loading.value = false
  }
}

/** 调用设置中的大模型，针对数据量与可调参数给出建议 */
async function onAiAdvice() {
  if (!task.value) await ensureTask()
  aiAdviceLoading.value = true
  try {
    const { data } = await fetchAiAdvice(task.value!.id)
    aiAdviceText.value = data.advice || ''
    aiAdviceSource.value = data.source || ''
    // 后端已写入报告；刷新任务 metrics
    const refreshed = await getTask(task.value!.id)
    task.value = refreshed.data
    if (data.source === 'llm') {
      ElMessage.success('已生成 AI 优化建议，下载报告将包含该建议')
    } else {
      ElMessage.warning('已使用本地规则建议（下载报告也会写入）；请检查设置中的大模型配置')
    }
  } finally {
    aiAdviceLoading.value = false
  }
}

function hydrateAiAdviceFromTask() {
  const m = task.value?.metrics || {}
  const advice = m.ai_advice
  if (typeof advice === 'string' && advice.trim()) {
    aiAdviceText.value = advice
    aiAdviceSource.value = typeof m.ai_advice_source === 'string' ? m.ai_advice_source : ''
  }
}

async function onExport() {
  if (!task.value) await ensureTask()
  loading.value = true
  try {
    const { data } = await startExport(task.value!.id)
    job.value = data
    await pollJob(data.id)
  } finally {
    loading.value = false
  }
}

/** 完成向导：停止轮询、清空本流程状态，回到导入步以便再次开训 */
function onFinishWizard() {
  stopPoll()
  trainHistory.value = []
  lastTrainStatus.value = null
  aiAdviceText.value = ''
  aiAdviceSource.value = ''
  job.value = null
  chart?.dispose()
  chart = null
  ElMessage.success('本轮流程已完成，可重新导入或选择数据集开始训练')
  emit('update:step', 0)
}

/** 等 DOM 可见且有宽度后再画图，避免 v-show/切换步骤导致宽度为 0 压成竖线 */
async function scheduleRenderChart() {
  await nextTick()
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()))
  renderChart()
}

function renderChart() {
  if (!showTrainChart.value || !chartRef.value) return

  const el = chartRef.value
  // 容器尚无宽度时推迟（常见于刚从 display:none / v-if 挂载）
  if (el.clientWidth < 80) {
    window.setTimeout(() => renderChart(), 60)
    return
  }

  if (chart && chart.getDom() !== el) {
    chart.dispose()
    chart = null
  }
  if (!chart) chart = echarts.init(el)

  const hs = history.value
  const labelInterval = hs.length <= 12 ? 0 : Math.max(0, Math.ceil(hs.length / 10) - 1)
  chart.setOption(
    {
      color: ['#1a5f7a', '#3d9b8f'],
      tooltip: { trigger: 'axis' },
      legend: { top: 8, left: 'center', data: ['loss', 'mAP50'] },
      grid: { left: 52, right: 28, top: 48, bottom: 36, containLabel: true },
      xAxis: {
        type: 'category',
        data: hs.map((x) => String(x.epoch)),
        boundaryGap: false,
        axisLabel: { interval: labelInterval, hideOverlap: true },
        name: 'epoch',
        nameLocation: 'middle',
        nameGap: 28,
      },
      yAxis: { type: 'value', scale: true },
      series: [
        {
          name: 'loss',
          type: 'line',
          smooth: true,
          showSymbol: hs.length <= 20,
          data: hs.map((x) => x.loss),
        },
        {
          name: 'mAP50',
          type: 'line',
          smooth: true,
          showSymbol: hs.length <= 20,
          data: hs.map((x) => x.map50),
        },
      ],
    },
    { notMerge: true },
  )
  chart.resize()
}

async function loadWeights() {
  const { data } = await listWeights('detect')
  weights.value = data
  const current = sanitizeWeightName(config.pretrained_weight)
  if (!current || !data.some((w) => w.name === current)) {
    config.pretrained_weight = data[0]?.name || ''
  } else {
    config.pretrained_weight = current
  }
}

async function loadDevices() {
  try {
    const { data } = await fetchDevices()
    deviceOptions.value = data.devices?.length ? data.devices : [{ id: 'cpu', label: 'CPU' }]
    if (!deviceOptions.value.some((d) => d.id === config.device)) {
      config.device = deviceOptions.value[0]?.id || 'cpu'
    }
  } catch {
    deviceOptions.value = [{ id: 'cpu', label: 'CPU' }]
    config.device = 'cpu'
  }
}

watch(
  () => props.step,
  async (s) => {
    if (s >= 3) {
      await ensureTask()
      await Promise.all([loadWeights(), loadDevices()])
      await resumeActiveTrainJob()
    }
    if (s === 5) {
      hydrateAiAdviceFromTask()
    }
    if (s !== 4) {
      // 离开训练步只释放图表实例，保留本次流程的 trainHistory；轮询继续以便返回后立刻有数据
      chart?.dispose()
      chart = null
      return
    }
    await scheduleRenderChart()
  },
)

watch(showTrainChart, async (visible) => {
  if (visible && props.step === 4) await scheduleRenderChart()
})

onMounted(async () => {
  await loadWeights()
  try {
    await ensureTask()
    await resumeActiveTrainJob()
  } catch {
    // 首次进入可稍后创建
  }
  window.addEventListener('resize', () => chart?.resize())
})

onUnmounted(() => {
  stopPoll()
  chart?.dispose()
  chart = null
})
</script>

<template>
  <div class="m3" v-loading="loading">
    <!-- 步骤4 配置 -->
    <div v-if="step === 3" class="step-body">
      <h3>步骤 4 · 配置</h3>
      <p class="muted">
        设置划分比例与训练超参。点击「下一步」将自动保存配置并划分数据集。
      </p>
      <div class="config-layout">
        <section class="config-card">
          <h4 class="card-title card-title-solo">基础配置</h4>
          <div class="form-grid">
            <label>训练任务名</label>
            <el-input v-model="taskName" :disabled="Boolean(task)" placeholder="如 demo_detect_train" />

            <label>数据划分</label>
            <SplitRatioBar
              class="split-field"
              v-model:train-ratio="config.train_ratio"
              v-model:val-ratio="config.val_ratio"
              :image-count="imageCount"
            />

            <label>预训练权重</label>
            <div class="weight-field">
              <el-select
                v-model="config.pretrained_weight"
                filterable
                clearable
                style="width: 100%"
                placeholder="请选择已上传的权重"
              >
                <el-option v-for="w in weightOptions" :key="w" :label="w" :value="w" />
              </el-select>
              <p v-if="!weightOptions.length" class="field-warn">
                暂无可用权重，请管理员先在「权重仓库」上传。
              </p>
            </div>

            <label>epochs</label>
            <div class="epochs-row">
              <el-slider v-model="config.epochs" :min="1" :max="500" :step="1" />
              <el-input-number v-model="config.epochs" :min="1" :max="500" controls-position="right" />
            </div>

            <label>设备</label>
            <el-select v-model="config.device" style="width: 100%" placeholder="根据服务器探测">
              <el-option
                v-for="d in deviceOptions"
                :key="d.id"
                :label="d.label"
                :value="d.id"
              />
            </el-select>
            <label>数据增强</label>
            <el-switch v-model="config.augment" />
          </div>
        </section>

        <aside class="config-card hyper-card">
          <div class="card-head">
            <h4 class="card-title">可选超参</h4>
            <el-radio-group v-model="hyperMode" class="hyper-mode">
              <el-radio value="default">默认</el-radio>
              <el-radio value="custom">自定义</el-radio>
            </el-radio-group>
          </div>
          <div class="hyper-stack">
            <div class="hyper-row">
              <label>batch</label>
              <div class="field-with-tip">
                <el-input-number
                  v-model="config.batch"
                  :min="1"
                  :max="128"
                  controls-position="right"
                  :disabled="hyperMode === 'default'"
                />
                <p class="field-tip">每步同时送入模型的图片数。越大通常越稳、越吃显存；样本少时可适当减小。</p>
              </div>
            </div>
            <div class="hyper-row">
              <label>imgsz</label>
              <div class="field-with-tip">
                <el-input-number
                  v-model="config.imgsz"
                  :min="320"
                  :max="1280"
                  :step="32"
                  controls-position="right"
                  :disabled="hyperMode === 'default'"
                />
                <p class="field-tip">训练时缩放后的边长（像素）。目标较小可适当增大；显存不足则减小。</p>
              </div>
            </div>
            <div class="hyper-row">
              <label>训练策略</label>
              <div class="field-with-tip">
                <el-radio-group
                  v-model="config.train_strategy"
                  class="strategy-group"
                  :disabled="hyperMode === 'default'"
                >
                  <el-radio v-for="s in STRATEGY_OPTIONS" :key="s.id" :value="s.id">
                    {{ s.label }}
                  </el-radio>
                </el-radio-group>
                <p class="field-tip">{{ strategyTip }}</p>
              </div>
            </div>
          </div>
          <p v-if="hyperMode === 'default'" class="hyper-hint">
            当前使用默认值：batch={{ DEFAULT_BATCH }}，imgsz={{ DEFAULT_IMGSZ }}，策略={{ strategyLabel }}
          </p>
        </aside>
      </div>
      <p v-if="splitInfo" class="muted">{{ splitInfo }}</p>
      <p v-if="showDemoHint" class="muted">
        当前为演示模式：训练将走 Mock。关闭演示模式并上传真实 .pt 后可本机真实训练。
      </p>
      <div class="actions">
        <el-button @click="emit('update:step', 2)">上一步</el-button>
        <el-button type="primary" @click="goNextToTrain">下一步：训练</el-button>
      </div>
    </div>

    <!-- 步骤5 训练 -->
    <div v-else-if="step === 4" class="step-body">
      <h3>步骤 5 · 训练</h3>
      <p class="muted">
        {{
          showDemoHint
            ? '演示模式按配置的 epochs 生成假曲线。点击「开始训练」后显示曲线；本流程内切换步骤会保留曲线。'
            : '按配置 epochs 运行训练，曲线随进度更新。本流程内切换步骤会保留曲线，仅再次开训时清空。'
        }}
      </p>
      <div class="actions">
        <el-button type="primary" :disabled="job?.status === 'running'" @click="onTrain">开始训练</el-button>
        <el-button :disabled="trainJob?.status !== 'running'" @click="onCancel">停止</el-button>
        <el-button @click="emit('update:step', 3)">上一步</el-button>
        <el-button type="primary" plain @click="emit('update:step', 5)">下一步：评估</el-button>
      </div>
      <div v-if="lastTrainStatus" class="job-box">
        <p>{{ lastTrainStatus.message }}（{{ lastTrainStatus.status }}）</p>
        <el-progress :percentage="Math.round(lastTrainStatus.progress)" />
      </div>
      <!-- 不用 v-show 隐藏画布，避免 ECharts 在宽度为 0 时初始化成竖线 -->
      <div v-if="showTrainChart" ref="chartRef" class="chart" />
      <p v-else class="muted chart-placeholder">尚未开始训练，暂无曲线。</p>
    </div>

    <!-- 步骤6 评估 -->
    <div v-else-if="step === 5" class="step-body">
      <h3>步骤 6 · 评估</h3>
      <p class="muted">生成指标与报告；生成 AI 建议后，下载的 HTML 报告会包含该建议。</p>
      <div class="actions">
        <el-button type="primary" @click="onEval">开始评估</el-button>
        <el-button
          type="success"
          plain
          :loading="aiAdviceLoading"
          :disabled="!task"
          @click="onAiAdvice"
        >
          AI 优化建议
        </el-button>
        <el-button
          v-if="task"
          @click="downloadTaskFile(task.id, 'report', 'eval_report.html')"
        >
          下载 HTML 报告
        </el-button>
        <el-button @click="emit('update:step', 4)">上一步</el-button>
        <el-button type="primary" plain @click="emit('update:step', 6)">下一步：导出</el-button>
      </div>
      <div v-if="job?.type === 'eval'" class="job-box">
        <p>{{ job.message }}</p>
        <el-progress :percentage="Math.round(job.progress)" />
      </div>
      <div class="metric-grid">
        <div class="metric"><span>mAP@0.5</span><strong>{{ evalMetrics.map50 ?? '—' }}</strong></div>
        <div class="metric"><span>mAP@0.5:0.95</span><strong>{{ evalMetrics.map50_95 ?? '—' }}</strong></div>
        <div class="metric"><span>Precision</span><strong>{{ evalMetrics.precision ?? '—' }}</strong></div>
        <div class="metric"><span>Recall</span><strong>{{ evalMetrics.recall ?? '—' }}</strong></div>
      </div>
      <p v-if="evalMetrics.suggestion" class="suggest">{{ evalMetrics.suggestion }}</p>
      <div v-if="aiAdviceText" class="ai-advice">
        <div class="ai-advice-head">
          <strong>AI 优化建议</strong>
          <span class="muted">来源：{{ aiAdviceSource === 'llm' ? '大模型' : '本地规则' }}</span>
        </div>
        <pre class="ai-advice-body">{{ aiAdviceDisplay }}</pre>
      </div>
    </div>

    <!-- 步骤7 导出 -->
    <div v-else-if="step === 6" class="step-body">
      <h3>步骤 7 · 导出</h3>
      <p class="muted">
        {{ showDemoHint ? '演示模式生成 PT/ONNX 占位文件。' : '导出 PT，并尝试导出 ONNX。' }}
        完成后可返回导入步，准备下一轮训练。
      </p>
      <div class="actions">
        <el-button type="primary" @click="onExport">导出 PT / ONNX</el-button>
        <el-button @click="emit('update:step', 5)">上一步</el-button>
        <el-button type="success" @click="onFinishWizard">完成</el-button>
      </div>
      <div v-if="job?.type === 'export'" class="job-box">
        <p>{{ job.message }}</p>
        <el-progress :percentage="Math.round(job.progress)" />
      </div>
      <ul class="export-list">
        <li v-for="f in exportFiles" :key="f.name">
          <span>{{ f.name }}{{ f.demo ? '（演示文件）' : '' }}</span>
          <el-button
            v-if="task"
            link
            type="primary"
            @click="downloadTaskFile(task.id, 'export', f.name)"
          >
            下载
          </el-button>
        </li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.step-body h3 {
  margin: 0 0 0.85rem;
  font-family: var(--font-display);
}
.muted {
  color: var(--ink-muted);
  font-size: 0.9rem;
  line-height: 1.6;
}
.form-grid {
  display: grid;
  grid-template-columns: 120px minmax(0, 1fr);
  gap: 0.75rem 0.85rem;
  align-items: center;
  margin: 0;
  max-width: none;
  min-width: 0;
}
.form-grid > * {
  min-width: 0;
}
.form-grid :deep(.el-select),
.form-grid :deep(.el-input) {
  width: 100%;
}
.form-grid > label:has(+ .split-field),
.form-grid > label:has(+ .field-with-tip) {
  align-self: start;
  padding-top: 0.55rem;
}
.config-layout {
  display: grid;
  grid-template-columns: minmax(0, 55fr) minmax(0, 43fr);
  column-gap: 2%;
  align-items: stretch;
  margin: 1rem 0;
}
.config-card {
  min-width: 0;
  padding: 1.1rem 1.2rem 1.25rem;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: #fff;
  box-shadow: 0 1px 2px rgba(15, 61, 79, 0.04);
}
.hyper-card {
  display: flex;
  flex-direction: column;
  padding: 1.2rem 1.35rem 1.4rem;
  overflow: visible;
  min-width: 0;
}
.hyper-stack {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: space-evenly;
  gap: 1.35rem;
  min-height: 12rem;
  min-width: 0;
}
.hyper-row {
  display: grid;
  grid-template-columns: 88px minmax(0, 1fr);
  column-gap: 1rem;
  align-items: start;
  min-width: 0;
}
.hyper-row > label {
  padding-top: 0.45rem;
  font-size: 0.9rem;
  color: var(--ink);
  line-height: 1.4;
}
.card-title {
  margin: 0;
  font-size: 0.92rem;
  font-weight: 700;
  color: var(--brand-deep);
  letter-spacing: 0.02em;
}
.card-title-solo {
  margin-bottom: 0.95rem;
}
.card-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.55rem 1rem;
  margin-bottom: 0.35rem;
  flex-shrink: 0;
}
.hyper-mode {
  flex-shrink: 0;
}
.hyper-mode :deep(.el-radio) {
  margin-right: 0.85rem;
}
.hyper-mode :deep(.el-radio:last-child) {
  margin-right: 0;
}
.hyper-hint {
  margin: 1rem 0 0;
  padding-top: 0.85rem;
  border-top: 1px solid var(--line);
  font-size: 0.78rem;
  color: var(--ink-faint);
  line-height: 1.5;
  flex-shrink: 0;
}
.strategy-group.el-radio-group {
  /* 固定等间距；窄屏自动换行，避免被裁切 */
  display: flex !important;
  flex-wrap: wrap;
  justify-content: flex-start;
  align-items: center;
  gap: 0.65rem 1.5rem;
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
}
.strategy-group :deep(.el-radio) {
  margin: 0 !important;
  margin-right: 0 !important;
  height: auto;
  white-space: nowrap;
  flex: 0 0 auto;
  max-width: 100%;
}
.strategy-group :deep(.el-radio__label) {
  padding-left: 6px;
  padding-right: 0;
}
.hyper-card .field-tip {
  overflow-wrap: anywhere;
  word-break: break-word;
}
.field-with-tip {
  display: grid;
  gap: 0.5rem;
  min-width: 0;
}
.field-tip {
  margin: 0;
  font-size: 0.78rem;
  color: var(--ink-faint);
  line-height: 1.55;
}
.hyper-card :deep(.el-input-number) {
  width: 100%;
  max-width: 410px;
}
.hyper-card :deep(.el-input-number .el-input) {
  width: 100%;
}
.hyper-card :deep(.el-input-number .el-input__wrapper) {
  width: 100%;
}
.epochs-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 118px;
  gap: 0.65rem;
  align-items: center;
  width: 100%;
  min-width: 0;
}
.epochs-row :deep(.el-slider) {
  min-width: 0;
  margin: 0 8px;
}
.epochs-row :deep(.el-input-number) {
  width: 118px;
}
.weight-field {
  display: grid;
  gap: 0.35rem;
}
.field-warn {
  margin: 0;
  font-size: 0.82rem;
  color: var(--warning);
}
.inline {
  display: grid;
  gap: 0.35rem;
}
.actions {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
  margin: 0.9rem 0;
  align-items: center;
}
.job-box {
  margin: 0.8rem 0;
  padding: 0.85rem 1rem;
  background: var(--brand-mist);
  border-radius: 10px;
}
.chart {
  width: 100%;
  min-width: 280px;
  height: 320px;
  margin-top: 0.5rem;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
  box-sizing: border-box;
}
.step-body {
  min-width: 0;
}
.chart-placeholder {
  margin-top: 1rem;
  padding: 2rem 1rem;
  text-align: center;
  border: 1px dashed var(--line);
  border-radius: 10px;
  background: #fafbfc;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
  margin-top: 1rem;
}
.metric {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.9rem;
  background: #fff;
}
.metric span {
  display: block;
  color: var(--ink-faint);
  font-size: 0.8rem;
}
.metric strong {
  font-size: 1.4rem;
  font-family: var(--font-display);
  color: var(--brand-deep);
}
.suggest {
  margin-top: 1rem;
  padding: 0.85rem 1rem;
  background: #f7f3e8;
  border-radius: 10px;
  color: #5a4a2a;
  line-height: 1.6;
}
.ai-advice {
  margin-top: 1rem;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
  overflow: hidden;
}
.ai-advice-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.75rem;
  padding: 0.7rem 1rem;
  background: var(--brand-mist);
  border-bottom: 1px solid var(--line);
}
.ai-advice-body {
  margin: 0;
  padding: 1rem;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: var(--font-body, inherit);
  font-size: 0.92rem;
  line-height: 1.65;
  color: var(--ink);
}
.export-list {
  list-style: none;
  margin: 1rem 0 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(0, 1fr));
  gap: 0.75rem;
}
.export-list li {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.75rem;
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.55rem 0.75rem;
}
.export-list li span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
@media (max-width: 900px) {
  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .config-layout {
    grid-template-columns: 1fr;
  }
  .form-grid {
    grid-template-columns: 1fr;
  }
  .epochs-row {
    grid-template-columns: 1fr;
  }
  .export-list {
    grid-template-columns: 1fr;
  }
}
</style>
