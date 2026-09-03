<script setup lang="ts">
/**
 * 对话式 AI Agent：多轮对话 + 工具调用（查数据集/标注/权重，不确定时 ask_user）。
 */
defineOptions({ name: 'AgentChat' })
import { computed, nextTick, onActivated, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ChatDotRound, Delete, Plus, Promotion } from '@element-plus/icons-vue'
import {
  cancelAgentChatTurn,
  createAgentChatSession,
  deleteAgentChatSession,
  getAgentChatSession,
  listAgentChatSessions,
  renameAgentChatSession,
  streamAgentChatMessage,
  streamContinueAgentChatJob,
  type AgentChatMessage,
  type AgentChatSession,
  type AgentChatSessionSummary,
  type AgentChatStreamEvent,
} from '@/api/agent'
import { getDataset, createDataset, listDatasets, uploadImages, uploadZip, type DatasetItem } from '@/api/datasets'
import { cancelJob, getJob, listModels, predictModel, type ModelItem } from '@/api/tasks'
import AnnotateStepPanel from '@/components/annotate/AnnotateStepPanel.vue'
import { renderMarkdown } from '@/utils/markdown'
import type { WizardTaskType } from '@/composables/useWizardSteps'

const router = useRouter()
const route = useRoute()

const sessions = ref<AgentChatSessionSummary[]>([])
const session = ref<AgentChatSession | null>(null)
const loadingList = ref(false)
const loadingSession = ref(false)
const sending = ref(false)
const draft = ref('')
const listEl = ref<HTMLElement | null>(null)
/** 轮询得到的 Job 实时状态（用于刷新对话里的 Job 卡片，不单独做顶部进度条） */
const liveJobs = ref<Record<number, { status: string; progress: number; message: string }>>({})
const renamingId = ref<string | null>(null)
const renameDraft = ref('')
const uploading = ref(false)
const uploadPercent = ref(0)
const fileInputRef = ref<HTMLInputElement | null>(null)
/** 流式状态文案：先思考，再按需显示工具进度 */
const liveStatus = ref('')
const liveThinking = ref('')
/** 回答正文流式缓冲（尚未写入会话消息前） */
const liveAnswer = ref('')
/** 流式思考区：滚到最新一行 */
const liveThinkBodyRef = ref<HTMLElement | null>(null)
/** 最近一次发送失败，供重试 */
const sendError = ref<{ message: string; retryText: string } | null>(null)
/** Job 续跑失败，供手动重试 */
const continueError = ref<{ message: string; jobId: number } | null>(null)
const stopping = ref(false)
const cancellingJob = ref(false)
let streamAbort: AbortController | null = null
/** 右侧标注抽屉 */
const annotateVisible = ref(false)
const annotateDatasetId = ref<number | null>(null)
const annotateTaskType = ref<WizardTaskType>('detect')
const annotatePanelRef = ref<{ flushSave?: () => Promise<void> } | null>(null)
/** 工具按钮指定的目标数据集；普通 + 号不预设 */
const preferredUploadDatasetId = ref<number | null>(null)
/** 工具已指定数据集时，选完文件跳过用途弹窗 */
const skipUploadIntent = ref(false)
const pendingFiles = ref<File[]>([])
const intentVisible = ref(false)
const intentConfirming = ref(false)
const intentMode = ref<'create' | 'existing' | 'infer'>('create')
const newDsName = ref('')
const newDsType = ref('detect')
const existingOptions = ref<DatasetItem[]>([])
const existingId = ref<number | null>(null)
const inferModels = ref<ModelItem[]>([])
const inferModelId = ref<number | null>(null)
let jobTimer: number | null = null
let continuingJob = false
/** 已处理完的 Job，防止取消后重复 continue-job */
const finishedJobIds = new Set<number>()

const pendingFileNames = computed(() => pendingFiles.value.map((f) => f.name))

const displayMessages = computed(() => {
  const msgs = session.value?.messages || []
  return msgs.filter((m) => {
    if (m.role === 'user') {
      // 隐藏系统回写的内部提示
      if (String(m.content || '').startsWith('[系统]')) return false
      return true
    }
    if (m.role === 'assistant') {
      const hasText = Boolean((m.content || '').trim())
      const hasReason = Boolean(m.ui?.reasoning)
      // 不因「计划调用工具」单独占一条气泡，减少过程噪音
      return hasText || hasReason || m.ui?.kind === 'ask_user'
    }
    if (m.role === 'tool') {
      // ask_user 只是内部停顿工具：问题已在助手气泡/快捷选项展示，不必再出工具卡片
      const toolName = String(m.ui?.tool || m.name || '')
      if (toolName === 'ask_user') return false
      // 闸门中断时补的占位 tool，对用户无意义
      if (m.ui?.hidden || m.ui?.skipped) return false
      return true
    }
    return false
  })
})

const pendingOptions = computed(() => {
  const opts = session.value?.pending_ask?.options
  return Array.isArray(opts) ? opts.filter((x) => String(x).trim()) : []
})

const activeJobId = computed(() => {
  const id = session.value?.pending_job?.job_id
  return typeof id === 'number' ? id : Number(id || 0) || null
})

function stopJobPoll() {
  if (jobTimer) {
    window.clearInterval(jobTimer)
    jobTimer = null
  }
}

function jobCardText(m: AgentChatMessage) {
  const id = Number(m.ui?.job_id || 0)
  const live = id ? liveJobs.value[id] : null
  const status = live?.status || String(m.ui?.status || '')
  const progress = live ? live.progress : Number(m.ui?.progress || 0)
  const msg = live?.message || String(m.ui?.summary || '')
  const pct = Math.min(100, Math.round(Number(progress) || 0))
  if (status === 'pending' && pct <= 0) {
    return `Job #${id || '-'} · 已提交，训练启动中…`
  }
  const tail = msg ? ` · ${msg}` : ''
  return `Job #${id || '-'} · ${status || 'running'} · ${pct}%${tail}`
}

async function onJobFinished() {
  if (!session.value || continuingJob) return
  const pendingId = Number(session.value.pending_job?.job_id || 0)
  if (!pendingId) {
    stopJobPoll()
    return
  }
  // 同一 Job 只续跑一次，避免取消后轮询打爆 continue-job
  if (finishedJobIds.has(pendingId)) {
    session.value.pending_job = null
    stopJobPoll()
    return
  }
  finishedJobIds.add(pendingId)
  continuingJob = true
  sending.value = true
  continueError.value = null
  liveStatus.value = '任务已结束，正在总结…'
  liveThinking.value = ''
  liveAnswer.value = ''
  streamAbort?.abort()
  streamAbort = new AbortController()
  try {
    const turn = await streamContinueAgentChatJob(
      session.value.id,
      (ev) => applyStreamEvent(ev),
      streamAbort.signal,
    )
    if (turn?.session) {
      session.value = turn.session
      if (turn.session.pending_job?.job_id) startJobPoll()
    }
    await refreshSessions()
    await scrollBottom()
  } catch (e) {
    finishedJobIds.delete(pendingId)
    const msg = e instanceof Error ? e.message : '续跑失败'
    continueError.value = { message: msg, jobId: pendingId }
    // 服务端可能已无 pending_job：清本地状态，停止轮询，避免 400 刷屏
    if (session.value) {
      session.value = {
        ...session.value,
        pending_job: null,
        status: session.value.status === 'waiting_job' ? 'idle' : session.value.status,
      }
    }
    stopJobPoll()
  } finally {
    continuingJob = false
    sending.value = false
    stopping.value = false
    liveStatus.value = ''
    liveThinking.value = ''
    liveAnswer.value = ''
    streamAbort = null
    stopJobPoll()
    if (session.value?.pending_job?.job_id) startJobPoll()
  }
}

function startJobPoll() {
  stopJobPoll()
  const jobId = activeJobId.value
  if (!jobId) return
  const tick = async () => {
    // 会话已不再等待该 Job
    if (Number(session.value?.pending_job?.job_id || 0) !== jobId) {
      stopJobPoll()
      return
    }
    try {
      const { data: job } = await getJob(jobId)
      liveJobs.value = {
        ...liveJobs.value,
        [jobId]: {
          status: job.status,
          progress: Number(job.progress || 0),
          message: job.message || job.status,
        },
      }
      if (session.value?.messages?.length) {
        for (const m of session.value.messages) {
          if (m.ui?.kind === 'job_progress' && Number(m.ui.job_id) === jobId) {
            m.ui = {
              ...m.ui,
              status: job.status,
              progress: Number(job.progress || 0),
              summary: job.message || job.status,
            }
          }
        }
      }
      if (!['pending', 'running'].includes(job.status)) {
        stopJobPoll()
        await onJobFinished()
      }
    } catch {
      /* 轮询失败不打断 */
    }
  }
  void tick()
  jobTimer = window.setInterval(() => {
    void tick()
  }, 2500)
}

watch(
  () => session.value?.pending_job?.job_id,
  (id) => {
    if (id) startJobPoll()
    else stopJobPoll()
  },
)

async function refreshSessions() {
  loadingList.value = true
  try {
    const { data } = await listAgentChatSessions()
    sessions.value = data
  } catch {
    sessions.value = []
  } finally {
    loadingList.value = false
  }
}

async function openSession(id: string) {
  loadingSession.value = true
  stopJobPoll()
  try {
    const { data } = await getAgentChatSession(id)
    session.value = data
    if (data.pending_job?.job_id) startJobPoll()
  } catch {
    session.value = null
  } finally {
    loadingSession.value = false
    // loading 结束后再滚到底，避免回到页面时停在顶部
    await scrollBottom(true)
  }
}

async function onNewChat() {
  stopJobPoll()
  try {
    const { data } = await createAgentChatSession()
    session.value = data
    await refreshSessions()
    await scrollBottom(true)
  } catch {
    // 拦截器
  }
}

async function onDeleteSession(id: string, ev?: Event) {
  ev?.stopPropagation()
  try {
    await ElMessageBox.confirm('删除该对话？', '确认', { type: 'warning' })
  } catch {
    return
  }
  try {
    await deleteAgentChatSession(id)
    if (session.value?.id === id) {
      session.value = null
      stopJobPoll()
    }
    await refreshSessions()
    ElMessage.success('已删除')
  } catch {
    // 拦截器
  }
}

async function startRename(s: AgentChatSessionSummary, ev?: Event) {
  ev?.stopPropagation()
  renamingId.value = s.id
  renameDraft.value = s.title || ''
  await nextTick()
  const el = document.querySelector('.hist-rename') as HTMLInputElement | null
  el?.focus()
  el?.select()
}

function cancelRename() {
  renamingId.value = null
  renameDraft.value = ''
}

async function commitRename(id: string) {
  if (renamingId.value !== id) return
  const title = renameDraft.value.trim()
  const prev = sessions.value.find((x) => x.id === id)?.title || ''
  renamingId.value = null
  renameDraft.value = ''
  if (!title) {
    ElMessage.warning('名称不能为空')
    return
  }
  if (title === prev) return
  try {
    const { data } = await renameAgentChatSession(id, title)
    if (session.value?.id === id) {
      session.value = { ...session.value, title: data.title }
    }
    await refreshSessions()
  } catch {
    // 拦截器
  }
}

/** 滚到对话最底部；force 时重试几次，避免 KeepAlive 恢复时高度未就绪 */
async function scrollBottom(force = false) {
  await nextTick()
  const el = listEl.value
  if (!el) return
  const apply = () => {
    el.scrollTop = el.scrollHeight
  }
  apply()
  if (!force) return
  const delays = [0, 50, 120, 280]
  for (const ms of delays) {
    await new Promise<void>((resolve) => {
      if (ms === 0) requestAnimationFrame(() => resolve())
      else setTimeout(resolve, ms)
    })
    apply()
  }
}

/** 思考内容区滚到最新生成的一行 */
async function scrollLiveThinking() {
  await nextTick()
  const el = liveThinkBodyRef.value
  if (!el) return
  el.scrollTop = el.scrollHeight
}

let streamScrollTick = 0

/** 把后端状态文案收成更口语的进度提示 */
function friendlyLiveStatus(text: string, phase: string, label: string) {
  const t = (text || '').trim()
  const lab = (label || '').trim()
  if (phase === 'thinking' || t.includes('思考')) return '正在思考…'
  if (t.includes('已确认') || t.includes('正在执行') || t.includes('正在启动')) {
    if (lab) return `正在${lab}…`
    return t.replace('已确认，', '').replace(/[「」]/g, '') || '正在处理…'
  }
  if (t.startsWith('准备调用') || t.startsWith('正在调用')) {
    if (lab) return `正在${lab}…`
    const m = t.match(/[：:]\s*(.+)$/)
    if (m?.[1]) {
      const first = m[1].split('、')[0]?.trim()
      if (first) return `正在${first}…`
    }
    return '正在处理…'
  }
  return t || '处理中…'
}

/** 统一处理 SSE 事件（发消息 / Job 续跑共用） */
function applyStreamEvent(ev: AgentChatStreamEvent) {
  if (ev.type === 'status') {
    liveStatus.value = friendlyLiveStatus(String(ev.text || ''), String(ev.phase || ''), String(ev.label || ev.tool || ''))
  } else if (ev.type === 'thinking_delta' && ev.delta) {
    liveThinking.value += String(ev.delta)
    liveStatus.value = '正在思考…'
    streamScrollTick += 1
    void scrollLiveThinking()
    if (streamScrollTick % 8 === 0) void scrollBottom()
  } else if (ev.type === 'answer_delta' && ev.delta) {
    liveAnswer.value += String(ev.delta)
    liveStatus.value = '正在回复…'
    streamScrollTick += 1
    if (streamScrollTick % 8 === 0) void scrollBottom()
  } else if (ev.type === 'thinking' && ev.text) {
    if (!liveThinking.value) liveThinking.value = String(ev.text)
    liveStatus.value = '正在思考…'
    void scrollLiveThinking()
  } else if (ev.type === 'session' && ev.session) {
    session.value = ev.session as AgentChatSession
    liveThinking.value = ''
    liveAnswer.value = ''
    void scrollBottom()
  } else if (ev.type === 'cancelled' && ev.session) {
    session.value = ev.session
  } else if (ev.type === 'done' && ev.turn?.session) {
    session.value = ev.turn.session
  } else if (ev.type === 'error') {
    if (ev.session) session.value = ev.session
    if (ev.message) ElMessage.error(ev.message)
  }
}

const QUICK_PROMPTS = [
  { label: '看看现有数据集', text: '帮我看看现在有哪些数据集，哪些已经可以训练？' },
  { label: '从现场图片开始', text: '我有一些现场图片，想做视觉检测，请告诉我该怎么开始。' },
]

const showWelcome = computed(() => {
  if (sending.value) return false
  if (!session.value) return true
  return !(displayMessages.value || []).length
})

const liveAnswerHtml = computed(() => renderMarkdown(liveAnswer.value))

async function retryFailedSend() {
  if (!sendError.value || sending.value) return
  const text = sendError.value.retryText
  sendError.value = null
  await sendText(text)
}

async function retryContinueJob() {
  if (!continueError.value || sending.value || continuingJob) return
  const jobId = continueError.value.jobId
  continueError.value = null
  finishedJobIds.delete(jobId)
  // 恢复 pending 以便走续跑逻辑
  if (session.value) {
    session.value = {
      ...session.value,
      pending_job: {
        ...(session.value.pending_job || {}),
        job_id: jobId,
      },
      status: 'waiting_job',
    }
  }
  await onJobFinished()
}

async function sendText(text: string) {
  const content = text.trim()
  if (!content || sending.value) return
  if (!session.value) {
    const { data } = await createAgentChatSession()
    session.value = data
    await refreshSessions()
  }
  const sid = session.value!.id
  sendError.value = null
  // 乐观更新：立刻显示用户气泡（主流对话体验）
  const tempId = `local-${Date.now()}`
  session.value.messages = [
    ...(session.value.messages || []),
    {
      id: tempId,
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    },
  ]
  draft.value = ''
  sending.value = true
  liveStatus.value = '正在思考…'
  liveThinking.value = ''
  liveAnswer.value = ''
  streamAbort?.abort()
  streamAbort = new AbortController()
  await scrollBottom()
  try {
    let sawError = false
    streamScrollTick = 0
    const turn = await streamAgentChatMessage(
      sid,
      content,
      (ev) => {
        if (ev.type === 'error') sawError = true
        applyStreamEvent(ev)
      },
      streamAbort.signal,
    )
    if (turn?.session) {
      session.value = turn.session
      if (turn.session.pending_job?.job_id) startJobPoll()
    }
    if (session.value?.messages?.some((m) => m.id === tempId)) {
      session.value.messages = session.value.messages.filter((m) => m.id !== tempId)
    }
    if (!turn && !sawError && !streamAbort?.signal.aborted) {
      sendError.value = { message: '未收到完整回复', retryText: content }
    }
    await refreshSessions()
    await scrollBottom()
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') {
      // 用户主动停止
    } else {
      const already = Boolean((e as { alreadyNotified?: boolean })?.alreadyNotified)
      const msg = e instanceof Error ? e.message : '发送失败'
      sendError.value = { message: msg, retryText: content }
      if (!already) {
        ElMessage.error(msg)
      }
      if (session.value?.messages) {
        session.value.messages = session.value.messages.filter((m) => m.id !== tempId)
      }
    }
  } finally {
    const wasAborted = Boolean(streamAbort?.signal.aborted)
    sending.value = false
    stopping.value = false
    liveStatus.value = ''
    liveThinking.value = ''
    liveAnswer.value = ''
    streamAbort = null
    if (wasAborted) {
      try {
        const { data } = await getAgentChatSession(sid)
        session.value = data
      } catch {
        /* ignore */
      }
    }
    await refreshSessions()
    await scrollBottom()
  }
}

/** 终止当前思考/工具循环 */
async function stopGenerating() {
  if (!sending.value || stopping.value) return
  const sid = session.value?.id
  if (!sid) return
  stopping.value = true
  liveStatus.value = '正在停止…'
  try {
    await cancelAgentChatTurn(sid)
  } catch {
    /* 仍尝试中断前端流 */
  }
  streamAbort?.abort()
}

/** 停止当前对话关联的训练/评估等 Job */
async function stopActiveJob(jobId?: number | null) {
  const id = Number(jobId || activeJobId.value || 0)
  if (!id || cancellingJob.value) return
  cancellingJob.value = true
  try {
    const { data } = await cancelJob(id)
    ElMessage.success(String((data as { message?: string })?.message || '已请求停止'))
    liveJobs.value = {
      ...liveJobs.value,
      [id]: {
        status: 'running',
        progress: liveJobs.value[id]?.progress || 0,
        message: '正在停止…',
      },
    }
    // 确保轮询仍在，结束后走一次收尾（不会重复 continue）
    if (session.value?.pending_job?.job_id) {
      startJobPoll()
    }
  } catch {
    // 拦截器
  } finally {
    cancellingJob.value = false
  }
}

async function onSend() {
  if (sending.value) {
    await stopGenerating()
    return
  }
  await sendText(draft.value)
}

function onPickOption(opt: string) {
  void sendText(opt)
}

function normalizeTaskType(raw?: string | null): WizardTaskType {
  const t = String(raw || 'detect')
  if (t === 'segment' || t === 'pose') return t
  return 'detect'
}

/** 在对话页右侧打开标注面板（不跳转向导） */
function openAnnotateDrawer(opts?: { datasetId?: number | null; taskType?: string | null }) {
  const ctx = session.value?.context || {}
  const dsId = Number(opts?.datasetId || ctx.dataset_id || 0) || null
  if (!dsId) {
    ElMessage.warning('请先创建或指定数据集，再去标注')
    return
  }
  annotateDatasetId.value = dsId
  annotateTaskType.value = normalizeTaskType(opts?.taskType || String(ctx.task_type || 'detect'))
  annotateVisible.value = true
}

function goAnnotate() {
  openAnnotateDrawer()
}

async function onAnnotateDrawerClose() {
  try {
    await annotatePanelRef.value?.flushSave?.()
  } catch {
    /* ignore */
  }
}

/** 标注完成并回写 Agent，推动检查标注 */
async function finishAnnotateAndNotify() {
  try {
    await annotatePanelRef.value?.flushSave?.()
  } catch {
    /* ignore */
  }
  const dsId = annotateDatasetId.value
  const dsName = String(session.value?.context?.dataset_name || (dsId ? `#${dsId}` : ''))
  annotateVisible.value = false
  const tip = dsName
    ? `我已完成数据集「${dsName}」的标注，请检查标注情况并建议下一步。`
    : '我已经完成标注，请检查数据集标注情况并建议下一步。'
  await sendText(tip)
}

function onQuickPrompt(text: string) {
  void sendText(text)
}

function isWizardAnnotateNav(action: {
  type: string
  path?: string
  query?: Record<string, string>
}) {
  if (action.type !== 'navigate' || !action.path) return false
  const step = String(action.query?.step || '')
  return step === '2' && /\/(detect|segment|pose)\/wizard/.test(action.path)
}

function bubbleClass(m: AgentChatMessage) {
  if (m.role === 'user') return 'bubble user'
  if (m.role === 'tool') return 'bubble tool'
  if (m.ui?.kind === 'error') return 'bubble assistant error'
  return 'bubble assistant'
}

function toolLabel(m: AgentChatMessage) {
  return String(m.ui?.label || m.name || '工具')
}

function toolSummary(m: AgentChatMessage) {
  return String(m.ui?.summary || m.content || '')
}

function toolOk(m: AgentChatMessage) {
  return m.ui?.ok !== false
}

function toolActions(m: AgentChatMessage): Array<{
  type: string
  label: string
  path?: string
  query?: Record<string, string>
  dataset_id?: number
  task_type?: string
}> {
  const raw = m.ui?.actions
  if (!Array.isArray(raw)) return []
  return raw
    .filter((a) => a && typeof a === 'object')
    .map((a) => {
      const item = a as {
        type?: string
        label?: string
        path?: string
        query?: Record<string, string>
        dataset_id?: number
        task_type?: string
      }
      return {
        type: String(item.type || 'navigate'),
        label: String(item.label || '打开'),
        path: item.path ? String(item.path) : undefined,
        query: item.query || {},
        dataset_id: item.dataset_id != null ? Number(item.dataset_id) : undefined,
        task_type: item.task_type ? String(item.task_type) : undefined,
      }
    })
    .filter((a) => a.type === 'navigate' || a.type === 'upload' || a.type === 'annotate')
}

/** 助手回复下方展示：自身 actions + 邻近工具返回的跳转/上传按钮 */
function actionsForAssistant(msg: AgentChatMessage) {
  const msgs = session.value?.messages || []
  const idx = msgs.findIndex((m) => m.id === msg.id)
  const seen = new Set<string>()
  const out: ReturnType<typeof toolActions> = []
  const pushActs = (list: ReturnType<typeof toolActions>) => {
    for (const act of list) {
      const key = `${act.type}|${act.path || ''}|${act.dataset_id || ''}|${act.label}`
      if (seen.has(key)) continue
      seen.add(key)
      out.push(act)
    }
  }
  pushActs(toolActions(msg))
  if (idx < 0) return out
  for (let i = idx - 1; i >= 0 && i >= idx - 8; i -= 1) {
    const m = msgs[i]
    if (m.role === 'user') break
    if (m.role !== 'tool') continue
    pushActs(toolActions(m))
  }
  return out
}

function onToolAction(action: {
  type: string
  label: string
  path?: string
  query?: Record<string, string>
  dataset_id?: number
  task_type?: string
}) {
  if (action.type === 'upload') {
    // 明确指定数据集时，选完文件直接导入，不再追问用途
    openFilePicker({ datasetId: action.dataset_id ?? null, skipIntent: true })
    return
  }
  if (action.type === 'annotate' || isWizardAnnotateNav(action)) {
    const dsFromQuery = action.query?.datasetId ? Number(action.query.datasetId) : null
    openAnnotateDrawer({
      datasetId: action.dataset_id ?? dsFromQuery,
      taskType: action.task_type || undefined,
    })
    return
  }
  if (action.path) {
    router.push({ path: action.path, query: action.query || {} })
  }
}

/** 打开文件选择：任意时刻都可点，不要求已有数据集上下文 */
function openFilePicker(opts?: { datasetId?: number | null; skipIntent?: boolean }) {
  preferredUploadDatasetId.value = opts?.datasetId ?? null
  skipUploadIntent.value = Boolean(opts?.skipIntent && opts?.datasetId)
  fileInputRef.value?.click()
}

function clearPendingFiles() {
  pendingFiles.value = []
  intentVisible.value = false
  preferredUploadDatasetId.value = null
  skipUploadIntent.value = false
}

/** 关闭用途弹窗时清理未确认的附件 */
function onIntentDialogClosed() {
  if (uploading.value || intentConfirming.value) return
  pendingFiles.value = []
  preferredUploadDatasetId.value = null
  skipUploadIntent.value = false
}

async function loadExistingDatasets() {
  const types = ['detect', 'segment', 'pose']
  const all: DatasetItem[] = []
  for (const tt of types) {
    try {
      const { data } = await listDatasets(tt)
      all.push(...data)
    } catch {
      /* ignore */
    }
  }
  existingOptions.value = all
  if (!existingId.value && all.length) existingId.value = all[0].id
}

async function loadInferModels() {
  try {
    const { data } = await listModels()
    inferModels.value = data
    if (!inferModelId.value && data.length) inferModelId.value = data[0].id
  } catch {
    inferModels.value = []
  }
}

async function onFilesSelected(ev: Event) {
  const input = ev.target as HTMLInputElement
  const files = Array.from(input.files || [])
  input.value = ''
  if (!files.length) return

  pendingFiles.value = files
  const skipIntent = skipUploadIntent.value
  skipUploadIntent.value = false

  // 工具明确指定数据集 → 直接上传
  if (skipIntent && preferredUploadDatasetId.value) {
    const dsId = preferredUploadDatasetId.value
    clearPendingFiles()
    await uploadFilesToDataset(dsId, files)
    return
  }

  // 有会话上下文数据集时，默认选「导入到已有」，但仍可改用途
  const ctxId = Number(session.value?.context?.dataset_id || 0) || null
  if (ctxId) {
    intentMode.value = 'existing'
    existingId.value = ctxId
  } else {
    intentMode.value = 'create'
    newDsName.value = `upload_${new Date().toISOString().slice(0, 10).replace(/-/g, '')}`
    newDsType.value = 'detect'
  }
  await Promise.all([loadExistingDatasets(), loadInferModels()])
  intentVisible.value = true
}

async function confirmAttachIntent() {
  const files = [...pendingFiles.value]
  if (!files.length) {
    clearPendingFiles()
    return
  }

  if (intentMode.value === 'create') {
    const name = newDsName.value.trim()
    if (!name) {
      ElMessage.warning('请输入数据集名称')
      return
    }
    intentConfirming.value = true
    try {
      const { data: ds } = await createDataset(name, newDsType.value)
      if (session.value) {
        session.value.context = {
          ...(session.value.context || {}),
          dataset_id: ds.id,
          dataset_name: ds.name,
          task_type: ds.task_type,
        }
      } else {
        const { data: sess } = await createAgentChatSession()
        session.value = sess
        session.value.context = {
          dataset_id: ds.id,
          dataset_name: ds.name,
          task_type: ds.task_type,
        }
        await refreshSessions()
      }
      clearPendingFiles()
      await uploadFilesToDataset(ds.id, files)
    } catch {
      // 拦截器
    } finally {
      intentConfirming.value = false
    }
    return
  }

  if (intentMode.value === 'existing') {
    if (!existingId.value) {
      ElMessage.warning('请选择数据集')
      return
    }
    const dsId = existingId.value
    clearPendingFiles()
    await uploadFilesToDataset(dsId, files)
    return
  }

  // 推理试用
  if (!inferModelId.value) {
    ElMessage.warning('请选择模型库中的模型')
    return
  }
  const image = files.find((f) => !/\.zip$/i.test(f.name))
  if (!image) {
    ElMessage.warning('推理试用请选择图片（暂不支持仅 zip）')
    return
  }
  const modelId = inferModelId.value
  clearPendingFiles()
  await runInferWithFile(modelId, image, files)
}

async function uploadFilesToDataset(datasetId: number, files: File[]) {
  if (!session.value) {
    const { data } = await createAgentChatSession()
    session.value = data
    await refreshSessions()
  }

  uploading.value = true
  uploadPercent.value = 0
  try {
    const zips = files.filter((f) => /\.zip$/i.test(f.name))
    const images = files.filter((f) => !/\.zip$/i.test(f.name))
    let saved = 0
    const names: string[] = []

    if (images.length) {
      const { data } = await uploadImages(datasetId, images, (p) => {
        uploadPercent.value = p
      })
      saved += Number(data?.saved || images.length)
      names.push(...images.map((f) => f.name))
    }
    for (const z of zips) {
      const { data } = await uploadZip(datasetId, z, (p) => {
        uploadPercent.value = p
      })
      saved += Number(data?.saved || 0)
      names.push(z.name)
    }

    let imageCount = 0
    let dsName = String(session.value?.context?.dataset_name || `#${datasetId}`)
    try {
      const { data: ds } = await getDataset(datasetId)
      imageCount = Number(ds.image_count || 0)
      dsName = ds.name || dsName
      if (session.value) {
        session.value.context = {
          ...(session.value.context || {}),
          dataset_id: ds.id,
          dataset_name: ds.name,
          task_type: ds.task_type,
        }
      }
    } catch {
      /* ignore */
    }

    const preview = names.slice(0, 3).join('、') + (names.length > 3 ? ` 等 ${names.length} 个文件` : '')
    ElMessage.success(`上传完成，约 ${saved} 张`)
    await sendText(
      `我刚往数据集「${dsName}」(id=${datasetId}) 上传了：${preview}。写入约 ${saved} 张，当前共 ${imageCount} 张。请检查标注情况，并告诉我下一步（标注或训练）。`,
    )
  } catch {
    // 拦截器
  } finally {
    uploading.value = false
    uploadPercent.value = 0
  }
}

async function runInferWithFile(modelId: number, image: File, allFiles: File[]) {
  if (!session.value) {
    const { data } = await createAgentChatSession()
    session.value = data
    await refreshSessions()
  }
  uploading.value = true
  try {
    const { data } = await predictModel(modelId, image)
    const model = inferModels.value.find((m) => m.id === modelId)
    const count = Number(data.count ?? data.detections?.length ?? 0)
    const b64 = data.image_base64 || ''
    const note = `我上传了图片「${image.name}」做推理试用（模型：${model?.name || '#' + modelId}），检出 ${count} 个目标。请帮我解读结果是否可用。`
    // 先写入带预览的本地助手卡片，再让 Agent 文字解读
    session.value.messages = [
      ...(session.value.messages || []),
      {
        id: `local-user-infer-${Date.now()}`,
        role: 'user',
        content: `上传图片推理：${allFiles.map((f) => f.name).join('、')}`,
        created_at: new Date().toISOString(),
      },
      {
        id: `local-infer-${Date.now()}`,
        role: 'tool',
        name: 'run_sample_infer',
        content: JSON.stringify({ ok: true, detection_count: count }),
        created_at: new Date().toISOString(),
        ui: {
          kind: 'infer_preview',
          label: '推理试用',
          summary: `${model?.name || '模型'} · 检出 ${count}`,
          preview_base64: b64,
          sample_image: image.name,
          detection_count: count,
        },
      },
    ]
    await scrollBottom()
    await sendText(note)
  } catch {
    // 拦截器
  } finally {
    uploading.value = false
  }
}
function isJobCard(m: AgentChatMessage) {
  return m.ui?.kind === 'job_progress'
}

function isJobRunning(m: AgentChatMessage) {
  if (!isJobCard(m)) return false
  const id = Number(m.ui?.job_id || 0)
  const live = id ? liveJobs.value[id] : null
  const status = String(live?.status || m.ui?.status || '')
  return ['pending', 'running'].includes(status)
}

function isInferCard(m: AgentChatMessage) {
  return m.ui?.kind === 'infer_preview' && Boolean(m.ui?.preview_base64)
}

function previewSrc(m: AgentChatMessage) {
  const b64 = String(m.ui?.preview_base64 || '')
  if (!b64) return ''
  return b64.startsWith('data:') ? b64 : `data:image/jpeg;base64,${b64}`
}

function assistantHtml(m: AgentChatMessage) {
  const raw = String(m.content || '')
  if (m.ui?.kind === 'error') {
    return renderMarkdown(friendlyErrorText(raw))
  }
  return renderMarkdown(raw)
}

function reasoningText(m: AgentChatMessage) {
  return String(m.ui?.reasoning || '').trim()
}

/** 仅本轮进行中时展开最新一条思考；结束后默认收起（可点击再看） */
function isLiveReasoning(m: AgentChatMessage) {
  if (!sending.value) return false
  const list = (session.value?.messages || []).filter(
    (x) => x.role === 'assistant' && String(x.ui?.reasoning || '').trim(),
  )
  const last = list[list.length - 1]
  return Boolean(last && last.id === m.id)
}

function reasoningSummary(m: AgentChatMessage) {
  return isLiveReasoning(m) ? '思考中…' : '思考过程'
}

/** 把接口原始错误收成用户可读短句 */
function friendlyErrorText(raw: string) {
  const text = String(raw || '').trim()
  if (!text) return '出了点问题，请重试。'
  const lower = text.toLowerCase()
  if (lower.includes('tool_calls') || lower.includes("role 'tool'")) {
    return '对话状态异常，请点重试；若仍失败，开一个新对话再试。'
  }
  if (lower.includes('http 401') || lower.includes('unauthorized')) {
    return '大模型接口未授权，请到系统设置检查 API Key。'
  }
  if (lower.includes('http 429') || lower.includes('rate')) {
    return '请求太频繁，请稍后再试。'
  }
  if (lower.includes('timeout') || lower.includes('timed out')) {
    return '响应超时，请重试。'
  }
  if (lower.includes('http 400')) {
    return '请求未被模型接受，请重试；若反复出现，建议新开对话。'
  }
  if (lower.includes('http 5') || lower.includes('502') || lower.includes('503')) {
    return '大模型服务暂时不可用，请稍后再试。'
  }
  // 截断过长 JSON / 堆栈
  const oneLine = text.replace(/\s+/g, ' ')
  if (oneLine.length > 120) {
    return `出了点问题：${oneLine.slice(0, 100)}… 请重试。`
  }
  if (oneLine.startsWith('发送失败：') || oneLine.startsWith('调用失败：') || oneLine.startsWith('任务续跑失败：')) {
    return oneLine
  }
  return `出了点问题：${oneLine}`
}

watch(
  () => session.value?.messages?.length,
  () => {
    void scrollBottom(true)
  },
)

watch(loadingSession, (v) => {
  if (!v && session.value) void scrollBottom(true)
})

onMounted(async () => {
  await refreshSessions()
  await applyEntryQuery()
})

/** KeepAlive 下离开再回来不会走 onMounted，需在激活时滚到底 */
onActivated(() => {
  void scrollBottom(true)
  // 从首页等再次带 query 进入时处理；首挂载由 onMounted 负责
  if (route.query.fresh || route.query.q) void applyEntryQuery()
})

/** 处理首页等入口带入的 query：fresh / q（不再自动代发引导话术） */
let entryBusy = false
async function applyEntryQuery() {
  const fresh = String(route.query.fresh || '') === '1'
  const q = String(route.query.q || '').trim()
  if (!fresh && !q) {
    if (!session.value && sessions.value[0]) await openSession(sessions.value[0].id)
    return
  }
  if (entryBusy) return
  entryBusy = true
  try {
    // 先清 query，避免 KeepAlive 下 onMounted + onActivated 双触发
    await router.replace({ path: '/app/agent/chat', query: {} })
    await onNewChat()
    // 仅当显式带 q 时才代发；空对话留给用户自己提问
    if (q) await sendText(q)
  } finally {
    entryBusy = false
  }
}

onUnmounted(() => stopJobPoll())
</script>

<template>
  <section class="chat-page">
    <aside class="history">
      <div class="history-head">
        <strong>对话</strong>
        <el-button type="primary" size="small" :icon="Plus" @click="onNewChat">新对话</el-button>
      </div>
      <div v-loading="loadingList" class="history-list">
        <button
          v-for="s in sessions"
          :key="s.id"
          type="button"
          class="hist-item"
          :class="{ active: session?.id === s.id }"
          @click="renamingId === s.id ? undefined : openSession(s.id)"
        >
          <el-icon><ChatDotRound /></el-icon>
          <input
            v-if="renamingId === s.id"
            v-model="renameDraft"
            class="hist-rename"
            maxlength="80"
            @click.stop
            @keydown.enter.prevent="commitRename(s.id)"
            @keydown.esc.prevent="cancelRename"
            @blur="commitRename(s.id)"
          />
          <span
            v-else
            class="hist-title"
            title="双击重命名"
            @dblclick="startRename(s, $event)"
          >
            {{ s.title }}
          </span>
          <el-button
            class="hist-del"
            link
            type="danger"
            :icon="Delete"
            @click="onDeleteSession(s.id, $event)"
          />
        </button>
        <p v-if="!sessions.length" class="hist-empty">暂无会话，点「新对话」开始</p>
      </div>
    </aside>

    <div class="main-chat" v-loading="loadingSession">
      <header class="chat-head">
        <div>
          <h2>{{ session?.title || 'AI Agent' }}</h2>
        </div>
        <div v-if="session?.context?.dataset_id" class="ctx">
          <span>上下文：{{ session.context.dataset_name || ('#' + session.context.dataset_id) }}</span>
          <el-button size="small" @click="goAnnotate">去标注</el-button>
        </div>
      </header>

      <div ref="listEl" class="messages">
        <div v-if="showWelcome" class="welcome">
          <h3>你好，我是训练平台 Agent</h3>
          <p>直接说你的现场需求或问题就行，我会一步步跟你确认后再动手。</p>
          <div class="welcome-chips">
            <button
              v-for="item in QUICK_PROMPTS"
              :key="item.label"
              type="button"
              class="welcome-chip"
              :disabled="sending"
              @click="onQuickPrompt(item.text)"
            >
              {{ item.label }}
            </button>
          </div>
          <el-button type="primary" @click="onNewChat">开始新对话</el-button>
        </div>

        <template v-else-if="session">
          <div v-for="m in displayMessages" :key="m.id" :class="bubbleClass(m)">
            <template v-if="m.role === 'tool'">
              <div v-if="isInferCard(m)" class="infer-card">
                <div class="tool-chip" :class="{ fail: !toolOk(m) }">
                  <span class="tool-name">{{ toolLabel(m) }}</span>
                  <span class="tool-sum">{{ toolSummary(m) }}</span>
                </div>
                <img class="infer-img" :src="previewSrc(m)" :alt="String(m.ui?.sample_image || 'preview')" />
                <div v-if="toolActions(m).length" class="tool-actions">
                  <button
                    v-for="(act, ai) in toolActions(m)"
                    :key="ai"
                    type="button"
                    class="tool-act-btn"
                    @click="onToolAction(act)"
                  >
                    {{ act.label }}
                  </button>
                </div>
              </div>
              <div v-else class="tool-block">
                <div :class="['tool-chip', isJobCard(m) ? 'job' : '', !toolOk(m) ? 'fail' : '']">
                  <span class="tool-name">{{ toolLabel(m) }}</span>
                  <span class="tool-sum">{{ isJobCard(m) ? jobCardText(m) : toolSummary(m) }}</span>
                </div>
                <div v-if="isJobCard(m) && isJobRunning(m)" class="tool-actions">
                  <button
                    type="button"
                    class="tool-act-btn danger"
                    :disabled="cancellingJob"
                    @click="stopActiveJob(Number(m.ui?.job_id || 0))"
                  >
                    {{ cancellingJob ? '停止中…' : '停止训练' }}
                  </button>
                </div>
                <div v-else-if="toolActions(m).length" class="tool-actions">
                  <button
                    v-for="(act, ai) in toolActions(m)"
                    :key="ai"
                    type="button"
                    class="tool-act-btn"
                    @click="onToolAction(act)"
                  >
                    {{ act.label }}
                  </button>
                </div>
              </div>
            </template>
            <template v-else-if="m.role === 'user'">
              <div class="text">{{ m.content }}</div>
            </template>
            <template v-else>
              <details
                v-if="reasoningText(m)"
                class="think-block"
                v-bind="isLiveReasoning(m) ? { open: true } : {}"
              >
                <summary>{{ reasoningSummary(m) }}</summary>
                <pre class="think-body">{{ reasoningText(m) }}</pre>
              </details>
              <div v-if="(m.content || '').trim()" class="md" v-html="assistantHtml(m)" />
              <div v-if="actionsForAssistant(m).length" class="tool-actions under-assistant">
                <button
                  v-for="(act, ai) in actionsForAssistant(m)"
                  :key="ai"
                  type="button"
                  class="tool-act-btn"
                  @click="onToolAction(act)"
                >
                  {{ act.label }}
                </button>
              </div>
            </template>
          </div>
          <div v-if="sending" class="bubble assistant thinking">
            <div class="think-live-row">
              <div class="think-live-status">{{ liveStatus || '正在思考…' }}</div>
              <button type="button" class="stop-inline" :disabled="stopping" @click="stopGenerating">
                {{ stopping ? '停止中…' : '停止' }}
              </button>
            </div>
            <details v-if="liveThinking" class="think-block live" open>
              <summary>思考中…</summary>
              <pre ref="liveThinkBodyRef" class="think-body">{{ liveThinking }}</pre>
            </details>
            <div v-if="liveAnswer" class="live-answer md" v-html="liveAnswerHtml" />
          </div>
        </template>
      </div>

      <div v-if="sendError" class="fail-banner">
        <span>{{ friendlyErrorText(sendError.message) }}</span>
        <button type="button" class="fail-retry" :disabled="sending" @click="retryFailedSend">重试</button>
      </div>
      <div v-else-if="continueError" class="fail-banner">
        <span>{{ friendlyErrorText(continueError.message) }}</span>
        <button type="button" class="fail-retry" :disabled="sending" @click="retryContinueJob">重试</button>
      </div>

      <div v-if="activeJobId && !sending" class="job-banner">
        <span>后台任务 Job #{{ activeJobId }} 进行中</span>
        <button type="button" class="stop-inline" :disabled="cancellingJob" @click="stopActiveJob()">
          {{ cancellingJob ? '停止中…' : '停止训练' }}
        </button>
      </div>

      <div v-if="pendingOptions.length && !sending" class="quick-opts">
        <button
          v-for="opt in pendingOptions"
          :key="opt"
          type="button"
          class="quick-chip"
          :disabled="sending"
          @click="onPickOption(opt)"
        >
          {{ opt }}
        </button>
      </div>

      <footer class="composer-wrap">
        <input
          ref="fileInputRef"
          type="file"
          class="hidden-file"
          multiple
          accept="image/*,.zip,application/zip"
          @change="onFilesSelected"
        />
        <div class="composer-card" :class="{ busy: uploading }">
          <el-input
            v-model="draft"
            class="composer-input"
            type="textarea"
            :autosize="{ minRows: 2, maxRows: 6 }"
            resize="none"
            placeholder="发消息，或点 + 添加图片/zip（建数据集、导入、推理试用均可）…"
            :disabled="sending || uploading"
            @keydown.enter.exact.prevent="onSend"
          />
          <div v-if="uploading" class="upload-line">
            <el-progress :percentage="uploadPercent" :stroke-width="8" />
            <span>正在上传…</span>
          </div>
          <div class="composer-bar">
            <div class="composer-left">
              <button
                type="button"
                class="composer-plus"
                title="添加图片或 zip"
                :disabled="sending || uploading"
                @click="openFilePicker()"
              >
                <el-icon :size="18"><Plus /></el-icon>
              </button>
              <div class="composer-hints">
                <span>Enter 发送</span>
                <span class="dot">·</span>
                <span>+ 添加附件</span>
              </div>
            </div>
            <button
              type="button"
              class="composer-send"
              :disabled="uploading || (!sending && !draft.trim())"
              :class="{ ready: Boolean(draft.trim()) || sending, stop: sending }"
              :title="sending ? '停止生成' : '发送'"
              @click="onSend"
            >
              <span v-if="sending" class="send-stop-icon" />
              <el-icon v-else-if="!sending" :size="18"><Promotion /></el-icon>
            </button>
          </div>
        </div>
      </footer>
    </div>

    <el-dialog
      v-model="intentVisible"
      title="这些文件用来做什么？"
      width="480px"
      :close-on-click-modal="false"
      @closed="onIntentDialogClosed"
    >
      <p class="intent-files">
        已选 {{ pendingFileNames.length }} 个文件：
        {{ pendingFileNames.slice(0, 4).join('、') }}{{ pendingFileNames.length > 4 ? '…' : '' }}
      </p>
      <el-radio-group v-model="intentMode" class="intent-modes">
        <el-radio value="create">新建数据集并导入</el-radio>
        <el-radio value="existing">导入到已有数据集</el-radio>
        <el-radio value="infer">用于推理试用</el-radio>
      </el-radio-group>

      <div v-if="intentMode === 'create'" class="intent-form">
        <el-form label-width="88px" size="default">
          <el-form-item label="名称">
            <el-input v-model="newDsName" placeholder="数据集名称" maxlength="64" />
          </el-form-item>
          <el-form-item label="任务类型">
            <el-select v-model="newDsType" style="width: 100%">
              <el-option label="目标检测 detect" value="detect" />
              <el-option label="实例分割 segment" value="segment" />
              <el-option label="姿态估计 pose" value="pose" />
            </el-select>
          </el-form-item>
        </el-form>
      </div>
      <div v-else-if="intentMode === 'existing'" class="intent-form">
        <el-select
          v-model="existingId"
          filterable
          placeholder="选择数据集"
          style="width: 100%"
          :disabled="!existingOptions.length"
        >
          <el-option
            v-for="d in existingOptions"
            :key="d.id"
            :label="`${d.name} (${d.task_type} · ${d.image_count || 0} 张)`"
            :value="d.id"
          />
        </el-select>
        <p v-if="!existingOptions.length" class="intent-hint">暂无数据集，可改选「新建数据集并导入」</p>
      </div>
      <div v-else class="intent-form">
        <el-select
          v-model="inferModelId"
          filterable
          placeholder="选择模型库中的模型"
          style="width: 100%"
          :disabled="!inferModels.length"
        >
          <el-option
            v-for="m in inferModels"
            :key="m.id"
            :label="`${m.name} (${m.task_type || '-'})`"
            :value="m.id"
          />
        </el-select>
        <p class="intent-hint">将用第一张图片做一次推理试用；zip 请改选数据集导入。</p>
      </div>

      <template #footer>
        <el-button @click="clearPendingFiles">取消</el-button>
        <el-button type="primary" :loading="intentConfirming || uploading" @click="confirmAttachIntent">
          确认
        </el-button>
      </template>
    </el-dialog>

    <el-drawer
      v-model="annotateVisible"
      title="标注"
      direction="rtl"
      size="86%"
      class="annotate-drawer"
      destroy-on-close
      :close-on-click-modal="false"
      @close="onAnnotateDrawerClose"
    >
      <div v-if="annotateDatasetId" class="annotate-drawer-body">
        <AnnotateStepPanel
          ref="annotatePanelRef"
          embedded
          :dataset-id="annotateDatasetId"
          :task-type="annotateTaskType"
        />
      </div>
      <template #footer>
        <div class="annotate-footer">
          <el-button @click="annotateVisible = false">关闭</el-button>
          <el-button type="primary" :disabled="sending" @click="finishAnnotateAndNotify">
            标完了，继续对话
          </el-button>
        </div>
      </template>
    </el-drawer>
  </section>
</template>

<style scoped>
.chat-page {
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr);
  gap: 0.85rem;
  height: calc(100vh - 140px);
  min-height: 480px;
}
.history {
  display: flex;
  flex-direction: column;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #f7fafb;
  overflow: hidden;
}
.history-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.7rem 0.75rem;
  border-bottom: 1px solid var(--line);
}
.history-list {
  flex: 1;
  overflow: auto;
  padding: 0.45rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}
.hist-item {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  width: 100%;
  border: 0;
  background: transparent;
  border-radius: 8px;
  padding: 0.45rem 0.5rem;
  cursor: pointer;
  text-align: left;
  color: var(--ink);
}
.hist-item:hover,
.hist-item.active {
  background: #e8f1f4;
}
.hist-title {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 0.86rem;
}
.hist-rename {
  flex: 1;
  min-width: 0;
  height: 28px;
  border: 1px solid var(--brand-soft, #9ec3d6);
  border-radius: 6px;
  padding: 0 0.4rem;
  font-size: 0.86rem;
  outline: none;
  background: #fff;
  color: var(--ink);
}
.hist-del {
  opacity: 0;
}
.hist-item:hover .hist-del {
  opacity: 1;
}
.hist-empty {
  margin: 0.75rem 0.4rem;
  font-size: 0.82rem;
  color: var(--ink-muted);
}
.main-chat {
  display: flex;
  flex-direction: column;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #f5f7f9;
  overflow: hidden;
  min-width: 0;
}
.chat-head {
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  align-items: flex-start;
  padding: 0.85rem 1rem;
  border-bottom: 1px solid #e8eef2;
  background: #fff;
}
.chat-head h2 {
  margin: 0;
  font-size: 1.05rem;
  color: var(--brand-deep);
}
.ctx {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.8rem;
  color: var(--ink-muted);
  white-space: nowrap;
}
.messages {
  flex: 1;
  overflow: auto;
  padding: 1rem 1rem 0.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
  background: transparent;
}
.welcome {
  margin: auto;
  text-align: center;
  color: var(--ink-muted);
  max-width: 460px;
}
.welcome h3 {
  margin: 0 0 0.4rem;
  color: var(--brand-deep);
}
.welcome-chips {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.45rem;
  margin: 0.9rem 0 1rem;
}
.welcome-chip {
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 999px;
  padding: 0.35rem 0.75rem;
  font-size: 0.82rem;
  cursor: pointer;
  transition: border-color 0.15s ease, background 0.15s ease;
}
.welcome-chip:hover:not(:disabled) {
  border-color: rgba(61, 155, 143, 0.45);
  background: #f7fbfa;
}
.welcome-chip:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}
.annotate-footer {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
}
.bubble {
  max-width: min(720px, 92%);
  padding: 0.65rem 0.85rem;
  border-radius: 12px;
  font-size: 0.92rem;
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
}
.bubble.user {
  align-self: flex-end;
  background: var(--brand-deep);
  color: #fff;
  border-bottom-right-radius: 4px;
}
.bubble.assistant {
  align-self: flex-start;
  background: #eef3f6;
  color: var(--ink);
  border-bottom-left-radius: 4px;
}
.bubble.assistant .md {
  white-space: normal;
}
.bubble.assistant .md :deep(p) {
  margin: 0 0 0.55rem;
}
.bubble.assistant .md :deep(p:last-child) {
  margin-bottom: 0;
}
.bubble.assistant .md :deep(ul),
.bubble.assistant .md :deep(ol) {
  margin: 0.35rem 0 0.55rem;
  padding-left: 1.25rem;
}
.bubble.assistant .md :deep(li) {
  margin: 0.15rem 0;
}
.bubble.assistant .md :deep(strong) {
  font-weight: 650;
  color: var(--brand-deep);
}
.bubble.assistant .md :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 0.45rem 0 0.65rem;
  font-size: 0.86rem;
  background: #fff;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #d7e0e6;
}
.bubble.assistant .md :deep(th),
.bubble.assistant .md :deep(td) {
  border: 1px solid #d7e0e6;
  padding: 0.4rem 0.55rem;
  text-align: left;
  vertical-align: top;
}
.bubble.assistant .md :deep(th) {
  background: #e8f0f4;
  font-weight: 600;
  color: var(--brand-deep);
}
.bubble.assistant .md :deep(code) {
  font-family: ui-monospace, Consolas, monospace;
  font-size: 0.85em;
  padding: 0.05rem 0.3rem;
  border-radius: 4px;
  background: #e4ecef;
}
.bubble.assistant .md :deep(pre) {
  margin: 0.4rem 0;
  padding: 0.55rem 0.7rem;
  border-radius: 8px;
  background: #1e2a32;
  color: #e8eef2;
  overflow: auto;
}
.bubble.assistant .md :deep(pre code) {
  background: transparent;
  padding: 0;
  color: inherit;
}
.bubble.assistant.error {
  background: #fff1f0;
  color: #a94442;
}
.bubble.tool {
  align-self: flex-start;
  padding: 0;
  background: transparent;
  max-width: min(720px, 92%);
}
.bubble.thinking {
  opacity: 0.92;
  font-style: normal;
  color: #5a6b78;
}
.live-answer {
  margin-top: 0.55rem;
  word-break: break-word;
  color: var(--ink, #1f2d3d);
  font-style: normal;
  line-height: 1.55;
}
.live-answer.md {
  white-space: normal;
}
.live-answer.md :deep(p) {
  margin: 0 0 0.45rem;
}
.live-answer.md :deep(p:last-child) {
  margin-bottom: 0;
}
.fail-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin: 0 1rem 0.35rem;
  padding: 0.45rem 0.75rem;
  border-radius: 10px;
  border: 1px solid #f0c2bd;
  background: #fff5f4;
  color: #a94442;
  font-size: 0.86rem;
}
.fail-retry {
  flex-shrink: 0;
  border: 1px solid #e8a59e;
  background: #fff;
  color: #b42318;
  font-size: 0.78rem;
  padding: 0.2rem 0.65rem;
  border-radius: 999px;
  cursor: pointer;
}
.fail-retry:hover:not(:disabled) {
  background: #fff1f0;
}
.fail-retry:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}
.think-live-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
}
.think-live-status {
  font-style: italic;
  margin-bottom: 0;
}
.stop-inline {
  flex-shrink: 0;
  border: 1px solid #d7e0e6;
  background: #fff;
  color: #b42318;
  font-size: 0.78rem;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  cursor: pointer;
}
.stop-inline:hover:not(:disabled) {
  border-color: #f0b4ae;
  background: #fff5f4;
}
.stop-inline:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.job-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin: 0 1rem 0.35rem;
  padding: 0.45rem 0.75rem;
  border-radius: 10px;
  border: 1px solid #e6ebef;
  background: #f7fafb;
  color: #5a6b78;
  font-size: 0.85rem;
}
.tool-act-btn.danger {
  color: #b42318;
  border-color: #f0b4ae;
}
.tool-act-btn.danger:hover {
  background: #fff5f4;
}
.composer-send.stop {
  background: #b42318;
  color: #fff;
}
.composer-send.stop:hover {
  transform: none;
  filter: brightness(1.05);
}
.send-stop-icon {
  width: 12px;
  height: 12px;
  border-radius: 2px;
  background: #fff;
}
.think-block {
  margin: 0 0 0.55rem;
  border: 1px solid #e4ebf0;
  border-radius: 10px;
  background: #f7fafb;
  overflow: hidden;
}
.think-block > summary {
  cursor: pointer;
  list-style: none;
  padding: 0.4rem 0.65rem;
  font-size: 0.82rem;
  color: #6a7c88;
  user-select: none;
}
.think-block > summary::-webkit-details-marker {
  display: none;
}
.think-block > summary::before {
  content: '▸ ';
  color: #9aabb6;
}
.think-block[open] > summary::before {
  content: '▾ ';
}
.think-body {
  margin: 0;
  padding: 0 0.7rem 0.65rem;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.78rem;
  line-height: 1.45;
  color: #4a5c68;
  max-height: 220px;
  overflow: auto;
}
.tool-chip {
  display: inline-flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.35rem 0.55rem;
  padding: 0.35rem 0.6rem;
  border-radius: 8px;
  border: 1px dashed #c5d3db;
  background: #f4f8fa;
  font-size: 0.8rem;
  color: #4a5c6a;
}
.tool-chip.job {
  border-color: #9ec3d6;
  background: #e8f4fa;
}
.tool-chip.fail {
  border-color: #e8a59e;
  border-style: solid;
  background: #fff5f4;
  color: #a94442;
}
.tool-block {
  display: grid;
  gap: 0.4rem;
}
.tool-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}
.tool-act-btn {
  border: 1px solid #c5d5de;
  background: #fff;
  color: var(--brand-deep);
  font-size: 0.8rem;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  cursor: pointer;
}
.tool-act-btn:hover {
  background: #e8f1f4;
}
.under-assistant {
  margin-top: 0.55rem;
}
.infer-card {
  display: grid;
  gap: 0.45rem;
  max-width: min(520px, 92vw);
}
.infer-img {
  width: 100%;
  border-radius: 8px;
  border: 1px solid #d7e0e6;
  background: #111;
}
.tool-name {
  font-weight: 600;
  color: var(--brand-deep);
}
.quick-opts {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  padding: 0 1.1rem 0.45rem;
}
.quick-chip {
  border: 1px solid #e6ebef;
  background: #fff;
  color: #5a6b78;
  font-size: 0.8rem;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  cursor: pointer;
}
.quick-chip:hover:not(:disabled) {
  border-color: #c5d5de;
  color: var(--brand-deep);
}
.quick-chip:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}
.composer-wrap {
  padding: 0.35rem 1rem 1rem;
  background: transparent;
}
.composer-card {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  padding: 0.85rem 0.95rem 0.7rem;
  border-radius: 22px;
  background: #fff;
  border: 1px solid #e8eef2;
  box-shadow:
    0 1px 2px rgba(20, 40, 60, 0.04),
    0 8px 24px rgba(20, 40, 60, 0.06);
}
.composer-input :deep(.el-textarea__inner) {
  border: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
  padding: 0.15rem 0.2rem 0.35rem;
  min-height: 56px !important;
  line-height: 1.55;
  font-size: 0.95rem;
  color: #1f2a32;
  resize: none;
}
.composer-input :deep(.el-textarea__inner::placeholder) {
  color: #9aa8b3;
}
.composer-input :deep(.el-textarea__inner:focus) {
  box-shadow: none !important;
}
.composer-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  min-height: 40px;
  padding-top: 0.15rem;
}
.composer-left {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  min-width: 0;
}
.composer-plus {
  width: 36px;
  height: 36px;
  border: 0;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #eef2f5;
  color: #5a6b78;
  cursor: pointer;
}
.composer-plus:hover:not(:disabled) {
  background: #e2eaf0;
  color: var(--brand-deep);
}
.composer-plus:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.hidden-file {
  display: none;
}
.upload-line {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  font-size: 0.78rem;
  color: var(--ink-muted);
  padding: 0 0.2rem 0.25rem;
}
.upload-line .el-progress {
  flex: 1;
}
.composer-hints {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.75rem;
  color: #9aa8b3;
  user-select: none;
}
.composer-hints .dot {
  opacity: 0.7;
}
.composer-send {
  width: 40px;
  height: 40px;
  border: 0;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #e8eef2;
  color: #7a8b98;
  cursor: not-allowed;
  transition:
    background 0.15s ease,
    color 0.15s ease,
    transform 0.12s ease;
}
.composer-send.ready {
  background: var(--brand-deep, #1f4f63);
  color: #fff;
  cursor: pointer;
}
.composer-send.ready:hover {
  transform: translateY(-1px);
}
.composer-send:disabled {
  cursor: not-allowed;
}
.send-loading {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.35);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
.intent-files {
  margin: 0 0 0.85rem;
  color: #5a6b78;
  font-size: 0.9rem;
  line-height: 1.45;
  word-break: break-all;
}
.intent-modes {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.45rem;
  margin-bottom: 0.9rem;
}
.intent-form {
  margin-top: 0.25rem;
}
.intent-hint {
  margin: 0.55rem 0 0;
  color: #8a9aa6;
  font-size: 0.82rem;
  line-height: 1.4;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
@media (max-width: 900px) {
  .chat-page {
    grid-template-columns: 1fr;
    height: auto;
  }
  .history {
    max-height: 200px;
  }
}
.annotate-drawer-body {
  height: calc(100vh - 120px);
  max-height: calc(100vh - 120px);
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  padding: 0;
}
</style>

<style>
/* el-drawer 传送到 body，需非 scoped */
.annotate-drawer.el-drawer .el-drawer__body {
  overflow: hidden;
  padding: 12px 16px 16px;
  box-sizing: border-box;
}
</style>
