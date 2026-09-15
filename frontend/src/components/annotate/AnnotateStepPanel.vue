<script setup lang="ts">
/**
 * 标注步骤面板：检测矩形框 / 分割多边形 / 姿态框+关键点。
 */
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import BBoxAnnotatorCanvas from '@/components/annotate/BBoxAnnotatorCanvas.vue'
import PolygonAnnotatorCanvas from '@/components/annotate/PolygonAnnotatorCanvas.vue'
import PoseAnnotatorCanvas from '@/components/annotate/PoseAnnotatorCanvas.vue'
import { cancelJob, getJob } from '@/api/tasks'
import {
  fetchImageObjectUrl,
  getAnnotation,
  getClasses,
  getPoseSkeleton,
  getPrelabelStatus,
  listImages,
  putAnnotation,
  putClasses,
  putPoseSkeleton,
  revertPrelabel,
  samAssist,
  startPrelabel,
  type BBox,
  type ImageItem,
  type PolygonInstance,
  type PoseInstance,
  type PoseSkeleton,
  type PrelabelStatus,
} from '@/api/datasets'
import { classAccent } from '@/utils/classColors'
import type { WizardTaskType } from '@/composables/useWizardSteps'

const props = defineProps<{
  datasetId: number
  taskType?: WizardTaskType
  /** 嵌入侧栏/抽屉时隐藏向导「下一步/返回清洗」与顶部说明 */
  embedded?: boolean
  /** 仅展示这些文件名（如主动学习难例/简单例），顺序与传入一致 */
  onlyNames?: string[]
  /** 隐藏 AI 预标注（主动学习复核场景） */
  hidePrelabel?: boolean
  /** 嵌入模式下返回按钮文案 */
  backLabel?: string
}>()

const emit = defineEmits<{
  back: []
  next: []
}>()

const isSegment = computed(() => props.taskType === 'segment')
const isPose = computed(() => props.taskType === 'pose')
const isDetect = computed(() => !isSegment.value && !isPose.value)
/** 一期姿态不做预标注；主动学习复核也不展示预标注 */
const showPrelabel = computed(() => !isPose.value && !props.hidePrelabel)

/** 分割标注模式：点选连点 / SAM2 单击 */
const segMode = ref<'point' | 'sam'>('point')
const samBusy = ref(false)

const loading = ref(false)
const images = ref<ImageItem[]>([])
const classes = ref<string[]>([])
const newClass = ref('')
const currentClassId = ref(0)
const imageIndex = ref(0)
const boxes = ref<BBox[]>([])
const polygons = ref<PolygonInstance[]>([])
const poses = ref<PoseInstance[]>([])
const poseSkeleton = ref<PoseSkeleton | null>(null)
const poseTemplate = ref<'coco17' | 'custom'>('coco17')
const customKptCount = ref(5)
const customKptNamesText = ref('')
const imageUrl = ref('')
let objectUrlToRevoke: string | null = null
let navigating = false

/** AI 预标注（仅检测） */
const prelabelInfo = ref<PrelabelStatus | null>(null)
const prelabelRunning = ref(false)
const prelabelProgress = ref(0)
const prelabelMessage = ref('')
const prelabelJobId = ref<number | null>(null)
let prelabelPollTimer: number | null = null

const activeImages = computed(() => {
  const all = images.value.filter((i) => i.status === 'active')
  if (!props.onlyNames?.length) return all
  const byName = new Map(all.map((i) => [i.name, i]))
  return props.onlyNames.map((n) => byName.get(n)).filter((x): x is ImageItem => Boolean(x))
})
const currentImage = computed(() => activeImages.value[imageIndex.value] || null)
const canPrelabel = computed(
  () => Boolean(prelabelInfo.value?.can_prelabel) && !prelabelRunning.value,
)
const canRevertPrelabel = computed(
  () => Boolean(prelabelInfo.value?.last_written?.length) && !prelabelRunning.value,
)
const prelabelHint = computed(() => {
  if (prelabelRunning.value) return '预标注进行中，请稍候…'
  if (prelabelInfo.value?.block_reason) return prelabelInfo.value.block_reason
  if (isPose.value) return '姿态估计一期请人工标注框与关键点'
  if (isSegment.value) {
    return segMode.value === 'sam'
      ? 'SAM 模式：在目标上单击即可生成轮廓'
      : '点选模式：单击加点，双击或 Enter 闭合'
  }
  return ''
})

const poseKptNames = computed(() => poseSkeleton.value?.kpt_names || [])
const poseEdges = computed(() => poseSkeleton.value?.skeleton || [])

async function loadPoseSkeleton() {
  if (!isPose.value) {
    poseSkeleton.value = null
    return
  }
  try {
    const { data } = await getPoseSkeleton(props.datasetId)
    poseSkeleton.value = data
    poseTemplate.value = data.template === 'custom' ? 'custom' : 'coco17'
    customKptCount.value = data.kpt_shape?.[0] || data.kpt_names?.length || 5
    customKptNamesText.value = (data.kpt_names || []).join(', ')
  } catch {
    poseSkeleton.value = null
  }
}

async function applyPoseSkeleton() {
  if (!isPose.value) return
  try {
    const body =
      poseTemplate.value === 'custom'
        ? {
            template: 'custom',
            kpt_count: customKptCount.value,
            kpt_names: customKptNamesText.value
              .split(/[,，\s]+/)
              .map((s) => s.trim())
              .filter(Boolean),
          }
        : { template: 'coco17' }
    const { data } = await putPoseSkeleton(props.datasetId, body)
    poseSkeleton.value = data
    ElMessage.success(`已应用骨架：${data.kpt_names.length} 个关键点`)
  } catch (e) {
    const msg = e instanceof Error ? e.message : '保存骨架失败'
    ElMessage.error(msg)
  }
}

function toggleSegMode() {
  if (samBusy.value) return
  segMode.value = segMode.value === 'point' ? 'sam' : 'point'
}

async function onSamClick(pt: { x: number; y: number }) {
  if (!isSegment.value || segMode.value !== 'sam') return
  if (!currentImage.value) return
  if (!classes.value.length) {
    ElMessage.warning('请先添加类别')
    return
  }
  if (samBusy.value) return
  samBusy.value = true
  try {
    const { data } = await samAssist(props.datasetId, {
      image: currentImage.value.name,
      x: pt.x,
      y: pt.y,
      positive: true,
    })
    const points = (data.points || []).map((p) => ({ x: p.x, y: p.y }))
    if (points.length < 3) {
      ElMessage.warning('SAM 未得到有效轮廓，请换位置再点')
      return
    }
    polygons.value = [
      ...polygons.value,
      { class_id: currentClassId.value, points },
    ]
    ElMessage.success('已添加 SAM 轮廓')
  } catch {
    // 拦截器提示
  } finally {
    samBusy.value = false
  }
}

function revokeUrl() {
  if (objectUrlToRevoke) {
    URL.revokeObjectURL(objectUrlToRevoke)
    objectUrlToRevoke = null
  }
}

function stopPrelabelPoll() {
  if (prelabelPollTimer) {
    window.clearInterval(prelabelPollTimer)
    prelabelPollTimer = null
  }
}

async function refreshPrelabelStatus() {
  const { data } = await getPrelabelStatus(props.datasetId)
  prelabelInfo.value = data
  if (data.running_job_id && !prelabelRunning.value) {
    // 页面刷新后恢复锁屏与轮询
    prelabelRunning.value = true
    prelabelJobId.value = data.running_job_id
    await pollPrelabelJob(data.running_job_id)
  }
}

async function refreshImages() {
  const { data } = await listImages(props.datasetId)
  images.value = data
}

async function loadClasses() {
  const { data } = await getClasses(props.datasetId)
  classes.value = data.classes || []
  if (currentClassId.value >= classes.value.length) {
    currentClassId.value = Math.max(0, classes.value.length - 1)
  }
  // 清理历史遗留：类别已删但标注文件里仍残留越界 class_id（如「类5」）
  try {
    await putClasses(props.datasetId, classes.value, [])
  } catch {
    // 清理失败不阻断标注页
  }
}

async function loadCurrentAnnotation() {
  boxes.value = []
  polygons.value = []
  poses.value = []
  if (!currentImage.value) {
    revokeUrl()
    imageUrl.value = ''
    return
  }
  const name = currentImage.value.name
  let url = ''
  try {
    url = await fetchImageObjectUrl(props.datasetId, name)
  } catch (e) {
    const msg = e instanceof Error ? e.message : `图片加载失败：${name}`
    ElMessage.error(msg)
    return
  }
  // 先切换到新 URL，再释放旧 URL，避免画布加载中途 blob 被吊销
  const prev = objectUrlToRevoke
  objectUrlToRevoke = url
  imageUrl.value = url
  if (prev && prev !== url) URL.revokeObjectURL(prev)

  try {
    const { data } = await getAnnotation(props.datasetId, name)
    boxes.value = data.boxes || []
    polygons.value = data.polygons || []
    poses.value = data.poses || []
  } catch {
    boxes.value = []
    polygons.value = []
    poses.value = []
  }
}

async function saveAnnotation(showToast = true) {
  if (!currentImage.value || prelabelRunning.value) return
  if (isSegment.value) {
    await putAnnotation(props.datasetId, currentImage.value.name, {
      polygons: polygons.value,
    })
  } else if (isPose.value) {
    await putAnnotation(props.datasetId, currentImage.value.name, { poses: poses.value })
  } else {
    await putAnnotation(props.datasetId, currentImage.value.name, { boxes: boxes.value })
  }
  if (showToast) ElMessage.success('标注已保存')
  await refreshImages()
  await refreshPrelabelStatus()
}

async function flushSave() {
  if (prelabelRunning.value) return
  await saveAnnotation(false)
}

async function prevImage() {
  if (prelabelRunning.value || navigating || imageIndex.value <= 0) return
  navigating = true
  try {
    await saveAnnotation(false)
    imageIndex.value -= 1
    await loadCurrentAnnotation()
  } finally {
    navigating = false
  }
}

async function nextImage() {
  if (prelabelRunning.value || navigating || imageIndex.value >= activeImages.value.length - 1) return
  navigating = true
  try {
    await saveAnnotation(false)
    imageIndex.value += 1
    await loadCurrentAnnotation()
  } finally {
    navigating = false
  }
}

/** 活跃图中未标注图片的下标列表（按当前列表顺序） */
const unlabeledIndices = computed(() =>
  activeImages.value
    .map((img, idx) => (img.has_label ? -1 : idx))
    .filter((idx) => idx >= 0),
)

/**
 * 跳转到下一张未标注图片；多次点击按未标注顺序逐一切换（到末尾后回到第一张）。
 */
async function jumpToNextUnlabeled() {
  if (prelabelRunning.value || navigating) return
  const indices = unlabeledIndices.value
  if (!indices.length) {
    ElMessage.info('当前没有未标注图片')
    return
  }
  const cur = imageIndex.value
  // 找当前之后的下一张未标注；若没有则从头循环
  const nextIdx = indices.find((i) => i > cur) ?? indices[0]
  if (nextIdx === cur) {
    ElMessage.info('仅剩当前这一张未标注图片')
    return
  }
  navigating = true
  try {
    await saveAnnotation(false)
    imageIndex.value = nextIdx
    await loadCurrentAnnotation()
  } finally {
    navigating = false
  }
}

/** 序号输入框展示值（1-based） */
const indexInput = ref('1')

watch(imageIndex, (idx) => {
  indexInput.value = String(idx + 1)
})

/**
 * 按输入序号跳转（1-based）；回车或失焦时生效。
 */
async function jumpToIndexInput() {
  if (prelabelRunning.value || navigating) {
    indexInput.value = String(imageIndex.value + 1)
    return
  }
  const total = activeImages.value.length
  if (!total) return
  const raw = Number.parseInt(String(indexInput.value).trim(), 10)
  if (!Number.isFinite(raw)) {
    indexInput.value = String(imageIndex.value + 1)
    ElMessage.warning('请输入有效的图片序号')
    return
  }
  const target = Math.min(total, Math.max(1, raw))
  indexInput.value = String(target)
  const nextIdx = target - 1
  if (nextIdx === imageIndex.value) return
  navigating = true
  try {
    await saveAnnotation(false)
    imageIndex.value = nextIdx
    await loadCurrentAnnotation()
  } finally {
    navigating = false
  }
}

async function onNavigate(dir: 'prev' | 'next') {
  if (prelabelRunning.value) return
  if (dir === 'prev') await prevImage()
  else await nextImage()
}

async function addClass() {
  if (prelabelRunning.value) return
  const name = newClass.value.trim()
  if (!name) return
  if (classes.value.includes(name)) {
    ElMessage.warning('类别已存在')
    return
  }
  classes.value = [...classes.value, name]
  newClass.value = ''
  await putClasses(props.datasetId, classes.value)
  await refreshPrelabelStatus()
}

async function removeClass(idx: number) {
  if (prelabelRunning.value) return
  const name = classes.value[idx] || `类${idx}`
  try {
    await ElMessageBox.confirm(
      `删除类别「${name}」后，将同时删除所有图片中该类别的标注，且更大编号的类别会自动前移。确认删除？`,
      '删除类别',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  const next = classes.value.filter((_, i) => i !== idx)
  const { data } = await putClasses(props.datasetId, next, [idx])
  classes.value = data.classes || next
  if (currentClassId.value === idx) {
    currentClassId.value = Math.max(0, classes.value.length - 1)
  } else if (currentClassId.value > idx) {
    currentClassId.value -= 1
  } else {
    currentClassId.value = Math.min(currentClassId.value, Math.max(0, classes.value.length - 1))
  }
  await loadCurrentAnnotation()
  await refreshImages()
  await refreshPrelabelStatus()
  const purged = data.purge?.annotations_removed ?? 0
  if (purged > 0) {
    ElMessage.success(`类别已删除，并清除 ${purged} 条对应标注`)
  } else {
    ElMessage.success('类别已删除')
  }
}

async function pollPrelabelJob(jobId: number) {
  stopPrelabelPoll()
  prelabelPollTimer = window.setInterval(async () => {
    try {
      const { data } = await getJob(jobId)
      prelabelProgress.value = Number(data.progress) || 0
      prelabelMessage.value = data.message || ''
      if (['completed', 'failed', 'cancelled'].includes(data.status)) {
        stopPrelabelPoll()
        prelabelRunning.value = false
        prelabelJobId.value = null
        await refreshImages()
        await refreshPrelabelStatus()
        await loadCurrentAnnotation()
        if (data.status === 'completed') {
          ElMessage.success(data.message || '预标注完成')
        } else if (data.status === 'failed') {
          ElMessage.error(data.message || '预标注失败')
        } else {
          ElMessage.warning(data.message || '预标注已取消')
        }
      }
    } catch {
      // 轮询失败不打断，下一次再试
    }
  }, 1200)
}

async function onPrelabel() {
  if (isPose.value) {
    ElMessage.info('姿态估计预标注将在后续版本开放')
    return
  }
  if (isSegment.value) {
    ElMessage.info('实例分割预标注将在二期开放')
    return
  }
  if (!canPrelabel.value) {
    ElMessage.warning(prelabelHint.value || '当前无法预标注')
    return
  }
  try {
    await ElMessageBox.confirm(
      `将使用已标注图片快速训练，再自动标注剩余未标注图片。\n进行期间不能切换或编辑图片，请耐心等待进度。\n\n已标注 ${prelabelInfo.value?.labeled_count ?? 0} 张 · 未标注 ${prelabelInfo.value?.unlabeled_count ?? 0} 张`,
      '开始 AI 预标注',
      { type: 'info', confirmButtonText: '开始', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  // 确认后立刻锁屏，避免保存/排队期间用户以为没开始而重复点击
  prelabelRunning.value = true
  prelabelJobId.value = null
  prelabelProgress.value = 0
  prelabelMessage.value = '正在准备预标注，请稍候…'
  try {
    // 此时已锁屏，需绕过 flushSave 的 running 守卫强制落盘
    if (currentImage.value) {
      if (isSegment.value) {
        await putAnnotation(props.datasetId, currentImage.value.name, { polygons: polygons.value })
      } else if (isPose.value) {
        await putAnnotation(props.datasetId, currentImage.value.name, { poses: poses.value })
      } else {
        await putAnnotation(props.datasetId, currentImage.value.name, { boxes: boxes.value })
      }
    }
    const { data } = await startPrelabel(props.datasetId)
    prelabelJobId.value = data.id
    prelabelProgress.value = data.progress || 0
    prelabelMessage.value = data.message || '排队中'
    await pollPrelabelJob(data.id)
  } catch {
    stopPrelabelPoll()
    prelabelRunning.value = false
    prelabelJobId.value = null
    prelabelProgress.value = 0
    prelabelMessage.value = ''
  }
}

async function onCancelPrelabel() {
  if (!prelabelJobId.value) return
  try {
    await cancelJob(prelabelJobId.value)
    prelabelMessage.value = '正在停止：将在当前 epoch 结束后停止，请稍候…'
    ElMessage.info('已请求停止。预标注不会瞬间中断，将在当前 epoch 结束后停止。')
  } catch {
    // 拦截器提示
  }
}

async function onRevertPrelabel() {
  if (!canRevertPrelabel.value) return
  const n = prelabelInfo.value?.last_written?.length || 0
  try {
    await ElMessageBox.confirm(
      `将删除本轮 AI 预标注写入的 ${n} 张图上的全部标注。若你已手动修改过这些图，修改也会一并清除。训练成功后预标注会自动确认为用户标注，届时无需再清除。是否继续？`,
      '清除本轮 AI 预标注',
      { type: 'warning', confirmButtonText: '清除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  const { data } = await revertPrelabel(props.datasetId)
  ElMessage.success(data.message)
  await refreshImages()
  await refreshPrelabelStatus()
  await loadCurrentAnnotation()
}

async function bootstrap() {
  loading.value = true
  stopPrelabelPoll()
  prelabelRunning.value = false
  try {
    await refreshImages()
    await loadClasses()
    await loadPoseSkeleton()
    imageIndex.value = 0
    await loadCurrentAnnotation()
    if (showPrelabel.value) await refreshPrelabelStatus()
  } finally {
    loading.value = false
  }
}

watch(
  () => props.datasetId,
  async () => {
    await bootstrap()
  },
)

watch(
  () => (props.onlyNames || []).join('\0'),
  async () => {
    imageIndex.value = 0
    await loadCurrentAnnotation()
  },
)

onMounted(() => {
  void bootstrap()
})

onUnmounted(() => {
  stopPrelabelPoll()
  void flushSave()
  revokeUrl()
})

defineExpose({ flushSave })
</script>

<template>
  <div class="step-body annotate" :class="{ embedded: embedded }" v-loading="loading">
    <template v-if="!embedded">
      <h3>步骤 3 · 标注</h3>
      <p class="muted intro">
        <template v-if="isPose">
          先配置骨架（COCO-17 或自定义点数），再画框并标注关键点。切换图片或进入下一步时会自动保存。
        </template>
        <template v-else-if="isSegment">
          可用点选或 SAM 标注部分图片，再用 AI 预标注补全其余图片（结果请人工复核）。切换图片或进入下一步时会自动保存。
        </template>
        <template v-else>
          可先手工标注部分图片，再使用 AI 预标注自动补全剩余图片（结果请人工复核）。切换图片或进入下一步时会自动保存标注。训练成功后，本轮预标注将自动视为用户标注，「清除本轮
          AI 预标注」将不再可用。
        </template>
      </p>
    </template>
    <div class="annotate-layout" :class="{ locked: prelabelRunning }">
      <aside class="side">
        <div v-if="isPose" class="class-box pose-box">
          <strong>骨架模板</strong>
          <el-radio-group v-model="poseTemplate" size="small" class="pose-tpl">
            <el-radio-button label="coco17">COCO-17</el-radio-button>
            <el-radio-button label="custom">自定义</el-radio-button>
          </el-radio-group>
          <div v-if="poseTemplate === 'custom'" class="pose-custom">
            <el-input-number v-model="customKptCount" :min="1" :max="64" size="small" />
            <el-input
              v-model="customKptNamesText"
              type="textarea"
              :rows="2"
              size="small"
              placeholder="可选：点名，逗号分隔；空则自动 kpt_0…"
            />
            <p class="muted prelabel-tip">
              自定义点数时，请使用匹配的 pose 权重或接受从头适配；COCO-17 可直接用官方 *-pose.pt
            </p>
          </div>
          <el-button size="small" type="primary" plain @click="applyPoseSkeleton">应用骨架</el-button>
          <p v-if="poseSkeleton" class="muted prelabel-tip">
            当前 {{ poseSkeleton.kpt_names.length }} 点 · {{ poseSkeleton.template }}
          </p>
        </div>
        <div class="class-box">
          <strong>类别</strong>
          <p v-if="!classes.length" class="muted class-hint">
            暂无类别，请先添加后再{{
              isSegment ? '画多边形' : isPose ? '画框并标关键点' : '画框'
            }}标注
          </p>
          <div class="class-list">
            <button
              v-for="(c, i) in classes"
              :key="c + i"
              type="button"
              class="class-item"
              :class="{ active: currentClassId === i }"
              :disabled="prelabelRunning"
              @click="currentClassId = i"
            >
              <span class="class-label">
                <i class="class-swatch" :style="{ background: classAccent(i) }" aria-hidden="true" />
                {{ c }}
              </span>
              <em @click.stop="removeClass(i)">删</em>
            </button>
          </div>
          <div class="form-row compact">
            <el-input
              v-model="newClass"
              size="small"
              placeholder="新类别"
              :disabled="prelabelRunning"
              @keyup.enter="addClass"
            />
            <el-button size="small" :disabled="prelabelRunning" @click="addClass">添加</el-button>
          </div>
        </div>
        <div class="nav-box">
          <p class="nav-title">
            <el-input
              v-model="indexInput"
              class="index-input"
              size="small"
              :disabled="prelabelRunning || !activeImages.length"
              title="输入序号后回车跳转"
              @keyup.enter="jumpToIndexInput"
              @blur="jumpToIndexInput"
            />
            <span class="index-total">/ {{ activeImages.length }}</span>
            <span v-if="currentImage" class="index-name">· {{ currentImage.name }}</span>
          </p>
          <div class="nav-row">
            <el-button size="small" :disabled="prelabelRunning || imageIndex <= 0" @click="prevImage">
              上一张 (A/←)
            </el-button>
            <el-button
              size="small"
              :disabled="prelabelRunning || imageIndex >= activeImages.length - 1"
              @click="nextImage"
            >
              下一张 (D/→)
            </el-button>
          </div>
          <div class="nav-stack">
            <div class="nav-row">
              <el-button
                type="success"
                plain
                :disabled="prelabelRunning || unlabeledIndices.length === 0"
                @click="jumpToNextUnlabeled"
              >
                未标注图片
                <template v-if="unlabeledIndices.length">（{{ unlabeledIndices.length }}）</template>
              </el-button>
              <el-button
                v-if="isSegment"
                type="primary"
                :plain="segMode !== 'sam'"
                :disabled="samBusy || prelabelRunning"
                :loading="samBusy"
                :title="segMode === 'point' ? '点击切换到 SAM 模式' : '点击切换到点选模式'"
                @click="toggleSegMode"
              >
                {{ segMode === 'point' ? '点选' : 'SAM' }}
              </el-button>
            </div>
            <el-button
              v-if="showPrelabel"
              type="warning"
              plain
              :disabled="!canPrelabel"
              :title="prelabelHint || 'AI 预标注'"
              @click="onPrelabel"
            >
              AI 预标注
            </el-button>
            <el-button v-if="showPrelabel" plain :disabled="!canRevertPrelabel" @click="onRevertPrelabel">
              清除本轮 AI 预标注
            </el-button>
            <p v-if="prelabelHint && !prelabelRunning" class="muted prelabel-tip">{{ prelabelHint }}</p>
            <p v-else-if="showPrelabel && prelabelInfo && !prelabelRunning" class="muted prelabel-tip">
              已标注 {{ prelabelInfo.labeled_count }} · 未标注 {{ prelabelInfo.unlabeled_count }}
              （需 ≥{{ prelabelInfo.min_labeled }} 张已标注）
            </p>
            <template v-if="!embedded">
              <el-button type="primary" plain :disabled="prelabelRunning || samBusy" @click="emit('next')">
                下一步：配置
              </el-button>
              <el-button :disabled="prelabelRunning || samBusy" @click="emit('back')">返回清洗</el-button>
            </template>
            <template v-else>
              <el-button type="primary" :disabled="prelabelRunning || samBusy" @click="emit('back')">
                {{ backLabel || '完成并返回' }}
              </el-button>
            </template>
          </div>
        </div>
      </aside>
      <div class="canvas-wrap">
        <div class="canvas-fill">
          <PolygonAnnotatorCanvas
            v-if="isSegment && imageUrl && !prelabelRunning"
            :key="`seg-${datasetId}-${currentImage?.name || ''}`"
            :image-url="imageUrl"
            :polygons="polygons"
            :class-id="currentClassId"
            :classes="classes"
            :mode="segMode"
            :sam-busy="samBusy"
            @update:polygons="polygons = $event"
            @navigate="onNavigate"
            @sam-click="onSamClick"
          />
          <PoseAnnotatorCanvas
            v-else-if="isPose && imageUrl && !prelabelRunning"
            :key="`pose-${datasetId}-${currentImage?.name || ''}-${poseKptNames.length}`"
            :image-url="imageUrl"
            :poses="poses"
            :class-id="currentClassId"
            :classes="classes"
            :kpt-names="poseKptNames"
            :skeleton="poseEdges"
            @update:poses="poses = $event"
            @navigate="onNavigate"
          />
          <BBoxAnnotatorCanvas
            v-else-if="isDetect && imageUrl && !prelabelRunning"
            :key="`det-${datasetId}-${currentImage?.name || ''}`"
            :image-url="imageUrl"
            :boxes="boxes"
            :class-id="currentClassId"
            :classes="classes"
            @update:boxes="boxes = $event"
            @navigate="onNavigate"
          />
          <p v-else-if="!prelabelRunning" class="muted empty-hint">暂无图片可标注</p>

          <div v-if="prelabelRunning" class="prelabel-overlay">
            <div class="prelabel-card">
              <h4>AI 预标注进行中</h4>
              <p class="muted">请勿切换或编辑图片，完成后将自动解锁。</p>
              <p class="prelabel-msg">{{ prelabelMessage || '准备中…' }}</p>
              <el-progress :percentage="Math.min(100, Math.round(prelabelProgress))" />
              <el-button class="stop-btn" @click="onCancelPrelabel">停止</el-button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.intro {
  margin: 0 0 0.85rem;
  font-size: 0.9rem;
  line-height: 1.55;
}
.annotate-layout {
  display: flex;
  align-items: stretch;
  gap: 1rem;
  position: relative;
}
.annotate-layout.locked {
  pointer-events: none;
}
.annotate-layout.locked .prelabel-overlay {
  pointer-events: auto;
}
.side {
  width: 260px;
  flex-shrink: 0;
  display: grid;
  gap: 0.85rem;
}
.class-box,
.nav-box {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.85rem;
  background: #fff;
}
.class-hint {
  margin: 0.4rem 0 0.6rem;
  font-size: 0.85rem;
}
.class-list {
  display: grid;
  gap: 0.35rem;
  margin: 0.6rem 0;
  max-height: 240px;
  overflow: auto;
}
.class-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fafbfc;
  padding: 0.4rem 0.55rem;
  cursor: pointer;
  font: inherit;
  text-align: left;
}
.class-item:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.class-item.active {
  border-color: var(--brand);
  background: var(--brand-mist);
}
.class-label {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  min-width: 0;
}
.class-swatch {
  width: 0.7rem;
  height: 0.7rem;
  border-radius: 2px;
  flex-shrink: 0;
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.12);
}
.class-item em {
  font-style: normal;
  color: var(--ink-faint);
  font-size: 0.8rem;
}
.form-row.compact {
  display: flex;
  gap: 0.4rem;
}
.nav-title {
  margin: 0 0 0.55rem;
  font-size: 0.9rem;
  word-break: break-all;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem;
}
.index-input {
  width: 4.2rem;
}
.index-input :deep(.el-input__wrapper) {
  padding: 0 0.45rem;
}
.index-input :deep(.el-input__inner) {
  text-align: center;
  font-variant-numeric: tabular-nums;
}
.index-total {
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.index-name {
  color: var(--ink-muted);
  min-width: 0;
}
.nav-box > .nav-row {
  margin-bottom: 0.55rem;
}
.nav-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  width: 100%;
  margin-bottom: 0;
}
.nav-row > :only-child {
  grid-column: 1 / -1;
}
.nav-row :deep(.el-button) {
  width: 100%;
  margin: 0;
}
.nav-stack {
  display: grid;
  gap: 0.5rem;
  width: 100%;
}
.nav-stack :deep(.el-button) {
  width: 100%;
  margin: 0;
}
.prelabel-tip {
  margin: 0;
  font-size: 0.78rem;
  line-height: 1.45;
}
.canvas-wrap {
  /* 高度跟随左侧：自身不参与撑高，内容绝对填满拉伸后的高度 */
  flex: 1;
  min-width: 0;
  min-height: 420px;
  position: relative;
}
.canvas-fill {
  position: absolute;
  inset: 0;
  overflow: auto;
  display: flex;
  flex-direction: column;
  min-height: 420px;
  background: #f3f5f7;
}
.canvas-fill :deep(.annotator) {
  flex: 1;
  min-height: 0;
  height: 100%;
  width: 100%;
}
.empty-hint {
  margin: 1rem;
}
.prelabel-overlay {
  position: absolute;
  inset: 0;
  z-index: 5;
  display: grid;
  place-items: center;
  background: rgba(247, 250, 252, 0.92);
  border: 1px solid var(--line);
  border-radius: 12px;
}
.prelabel-card {
  width: min(100%, 420px);
  padding: 1.25rem 1.35rem;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(15, 61, 79, 0.08);
}
.prelabel-card h4 {
  margin: 0 0 0.45rem;
  font-size: 1.05rem;
  color: var(--brand-deep);
}
.prelabel-msg {
  margin: 0.85rem 0 0.65rem;
  font-size: 0.9rem;
  color: var(--ink);
  line-height: 1.5;
  min-height: 1.4em;
}
.stop-btn {
  margin-top: 0.85rem;
  width: 100%;
}
.muted {
  color: var(--ink-muted);
}
@media (max-width: 900px) {
  .annotate-layout {
    flex-direction: column;
  }
  .side {
    width: 100%;
  }
  .canvas-wrap {
    /* 窄屏改为独立高度，避免绝对定位高度为 0 */
    position: relative;
    min-height: 420px;
    height: auto;
  }
  .canvas-fill {
    position: relative;
    inset: auto;
    min-height: 420px;
  }
}

.pose-box {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.pose-tpl {
  width: 100%;
}
.pose-custom {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

/* Agent 侧栏嵌入：在抽屉可视区内铺满，避免外层滚轮 */
.annotate.embedded {
  height: 100%;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}
.annotate.embedded .annotate-layout {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  align-items: stretch;
}
.annotate.embedded .side {
  overflow: auto;
  max-height: 100%;
  align-content: start;
}
.annotate.embedded .canvas-wrap {
  /* 吃掉剩余宽度/高度，不再用 420px 最小高度把抽屉撑出滚动条 */
  flex: 1;
  min-width: 0;
  min-height: 0;
  height: auto;
  align-self: stretch;
}
.annotate.embedded .canvas-fill {
  min-height: 0;
}
</style>
