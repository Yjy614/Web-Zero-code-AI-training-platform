<script setup lang="ts">
/**
 * 姿态估计标注画布：矩形框 + 固定骨架关键点。
 * 操作：拖拽画框；选中实例后单击放置/移动关键点；右键切换可见性；Delete 删除实例。
 */
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import type { PoseInstance, PoseKeypoint } from '@/api/datasets'
import { classAccent } from '@/utils/classColors'

const props = defineProps<{
  imageUrl: string
  poses: PoseInstance[]
  classId: number
  classes: string[]
  kptNames: string[]
  skeleton: number[][]
}>()

const emit = defineEmits<{
  'update:poses': [poses: PoseInstance[]]
  navigate: [dir: 'prev' | 'next']
}>()

type Handle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w'
type Mode = 'none' | 'draw' | 'move' | 'resize' | 'kpt'

const HANDLE_SIZE = 8
const MIN_BOX = 6
const KPT_R = 5

const canvasRef = ref<HTMLCanvasElement | null>(null)
const wrapRef = ref<HTMLDivElement | null>(null)
const scale = ref(1)
const imgNatural = ref({ w: 0, h: 0 })
const imgEl = ref<HTMLImageElement | null>(null)
const selected = ref(-1)
const activeKpt = ref(0)
const hoverCursor = ref('crosshair')

let mode: Mode = 'none'
let resizeHandle: Handle | null = null
let dragKpt = -1
let startX = 0
let startY = 0
let curX = 0
let curY = 0
let originRect = { x: 0, y: 0, bw: 0, bh: 0 }
let suppressDrawUntil = 0

const kptN = computed(() => Math.max(1, props.kptNames.length || 17))

const displaySize = computed(() => ({
  w: Math.round(imgNatural.value.w * scale.value),
  h: Math.round(imgNatural.value.h * scale.value),
}))

function emptyKeypoints(): PoseKeypoint[] {
  return Array.from({ length: kptN.value }, () => ({ x: 0, y: 0, v: 0 }))
}

function padPose(p: PoseInstance): PoseInstance {
  const kpts = [...(p.keypoints || [])]
  while (kpts.length < kptN.value) kpts.push({ x: 0, y: 0, v: 0 })
  return { ...p, keypoints: kpts.slice(0, kptN.value) }
}

async function loadImage(url: string) {
  const img = new Image()
  img.src = url
  await img.decode()
  imgEl.value = img
  imgNatural.value = { w: img.naturalWidth, h: img.naturalHeight }
  fitScale()
  await nextTick()
  draw()
}

function fitScale() {
  const wrap = wrapRef.value
  if (!wrap || !imgNatural.value.w) return
  const maxW = wrap.clientWidth - 16
  const maxH = Math.max(360, wrap.clientHeight - 16)
  const s = Math.min(maxW / imgNatural.value.w, maxH / imgNatural.value.h, 1.5)
  scale.value = s > 0 ? s : 1
}

function yoloToPixel(p: PoseInstance) {
  const { w, h } = displaySize.value
  const bw = p.width * w
  const bh = p.height * h
  const x = p.x_center * w - bw / 2
  const y = p.y_center * h - bh / 2
  return { x, y, bw, bh }
}

function pixelToPose(x: number, y: number, bw: number, bh: number, classId: number, kpts: PoseKeypoint[]): PoseInstance {
  const { w, h } = displaySize.value
  return {
    class_id: classId,
    x_center: (x + bw / 2) / w,
    y_center: (y + bh / 2) / h,
    width: bw / w,
    height: bh / h,
    keypoints: kpts,
  }
}

function clampRect(x: number, y: number, bw: number, bh: number) {
  const { w, h } = displaySize.value
  let nx = x
  let ny = y
  let nbw = Math.max(MIN_BOX, bw)
  let nbh = Math.max(MIN_BOX, bh)
  if (nx < 0) nx = 0
  if (ny < 0) ny = 0
  if (nx + nbw > w) nbw = Math.max(MIN_BOX, w - nx)
  if (ny + nbh > h) nbh = Math.max(MIN_BOX, h - ny)
  if (nx + nbw > w) nx = Math.max(0, w - nbw)
  if (ny + nbh > h) ny = Math.max(0, h - nbh)
  return { x: nx, y: ny, bw: nbw, bh: nbh }
}

function handleCenters(x: number, y: number, bw: number, bh: number): Record<Handle, { x: number; y: number }> {
  return {
    nw: { x, y },
    n: { x: x + bw / 2, y },
    ne: { x: x + bw, y },
    e: { x: x + bw, y: y + bh / 2 },
    se: { x: x + bw, y: y + bh },
    s: { x: x + bw / 2, y: y + bh },
    sw: { x, y: y + bh },
    w: { x, y: y + bh / 2 },
  }
}

function drawHandles(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  bw: number,
  bh: number,
  color: string,
) {
  const hs = HANDLE_SIZE
  const half = hs / 2
  const points = handleCenters(x, y, bw, bh)
  ctx.fillStyle = '#ffffff'
  ctx.strokeStyle = color
  ctx.lineWidth = 1.5
  for (const p of Object.values(points)) {
    ctx.fillRect(p.x - half, p.y - half, hs, hs)
    ctx.strokeRect(p.x - half, p.y - half, hs, hs)
  }
}

function draw() {
  const canvas = canvasRef.value
  const img = imgEl.value
  if (!canvas || !img) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const { w, h } = displaySize.value
  canvas.width = w
  canvas.height = h
  ctx.clearRect(0, 0, w, h)
  ctx.drawImage(img, 0, 0, w, h)

  props.poses.forEach((raw, i) => {
    const p = padPose(raw)
    const { x, y, bw, bh } = yoloToPixel(p)
    const isSel = i === selected.value
    const accent = classAccent(p.class_id)
    ctx.strokeStyle = accent
    ctx.lineWidth = isSel ? 3 : 2
    ctx.strokeRect(x, y, bw, bh)
    if (isSel) {
      ctx.strokeStyle = '#ffffffcc'
      ctx.lineWidth = 1.5
      ctx.strokeRect(x - 1, y - 1, bw + 2, bh + 2)
    }

    const label = props.classes[p.class_id] || `类${p.class_id}`
    ctx.font = '12px sans-serif'
    const tw = ctx.measureText(label).width
    const boxW = tw + 8
    const boxH = 18
    let lx = x
    let ly = y - boxH
    if (ly < 0) ly = y
    ctx.fillStyle = accent
    ctx.fillRect(lx, ly, boxW, boxH)
    ctx.fillStyle = '#fff'
    ctx.textBaseline = 'top'
    ctx.fillText(label, lx + 4, ly + 2)

    // 骨架线
    ctx.lineWidth = 2
    for (const edge of props.skeleton || []) {
      if (!edge || edge.length < 2) continue
      const a = p.keypoints[edge[0]]
      const b = p.keypoints[edge[1]]
      if (!a || !b || a.v <= 0 || b.v <= 0) continue
      ctx.strokeStyle = accent
      ctx.beginPath()
      ctx.moveTo(a.x * w, a.y * h)
      ctx.lineTo(b.x * w, b.y * h)
      ctx.stroke()
    }

    // 关键点
    p.keypoints.forEach((k, ki) => {
      if (k.v <= 0) return
      const px = k.x * w
      const py = k.y * h
      ctx.beginPath()
      ctx.arc(px, py, isSel && ki === activeKpt.value ? KPT_R + 2 : KPT_R, 0, Math.PI * 2)
      ctx.fillStyle = k.v === 1 ? '#f59e0b' : accent
      ctx.fill()
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 1.5
      ctx.stroke()
      if (isSel) {
        ctx.fillStyle = '#111'
        ctx.font = '10px sans-serif'
        ctx.fillText(String(ki), px + 6, py - 6)
      }
    })

    if (isSel) drawHandles(ctx, x, y, bw, bh, accent)
  })

  if (mode === 'draw') {
    const x = Math.min(startX, curX)
    const y = Math.min(startY, curY)
    const bw = Math.abs(curX - startX)
    const bh = Math.abs(curY - startY)
    ctx.strokeStyle = classAccent(props.classId)
    ctx.setLineDash([6, 4])
    ctx.strokeRect(x, y, bw, bh)
    ctx.setLineDash([])
  }
}

function canvasPos(e: MouseEvent) {
  const canvas = canvasRef.value!
  const rect = canvas.getBoundingClientRect()
  return {
    x: ((e.clientX - rect.left) / rect.width) * canvas.width,
    y: ((e.clientY - rect.top) / rect.height) * canvas.height,
  }
}

function hitTestBox(x: number, y: number) {
  for (let i = props.poses.length - 1; i >= 0; i--) {
    const { x: bx, y: by, bw, bh } = yoloToPixel(padPose(props.poses[i]))
    if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) return i
  }
  return -1
}

function hitTestHandle(x: number, y: number, index: number): Handle | null {
  if (index < 0 || index >= props.poses.length) return null
  const { x: bx, y: by, bw, bh } = yoloToPixel(padPose(props.poses[index]))
  const centers = handleCenters(bx, by, bw, bh)
  const tol = HANDLE_SIZE
  for (const [name, p] of Object.entries(centers) as [Handle, { x: number; y: number }][]) {
    if (Math.abs(x - p.x) <= tol && Math.abs(y - p.y) <= tol) return name
  }
  return null
}

function hitTestKpt(x: number, y: number, index: number): number {
  if (index < 0 || index >= props.poses.length) return -1
  const { w, h } = displaySize.value
  const p = padPose(props.poses[index])
  for (let i = p.keypoints.length - 1; i >= 0; i--) {
    const k = p.keypoints[i]
    if (k.v <= 0) continue
    const dx = k.x * w - x
    const dy = k.y * h - y
    if (dx * dx + dy * dy <= (KPT_R + 4) ** 2) return i
  }
  return -1
}

function nextUnsetKpt(p: PoseInstance): number {
  const idx = p.keypoints.findIndex((k) => k.v <= 0)
  return idx >= 0 ? idx : activeKpt.value
}

function emitPoses(next: PoseInstance[]) {
  emit(
    'update:poses',
    next.map((p) => padPose(p)),
  )
}

function deleteAt(index: number) {
  if (index < 0) return
  const next = props.poses.filter((_, i) => i !== index)
  selected.value = -1
  emitPoses(next)
}

function commitBox(index: number, x: number, y: number, bw: number, bh: number) {
  const clamped = clampRect(x, y, bw, bh)
  const cur = padPose(props.poses[index])
  const box = pixelToPose(clamped.x, clamped.y, clamped.bw, clamped.bh, cur.class_id, cur.keypoints)
  box.x_center = Math.min(1, Math.max(0, box.x_center))
  box.y_center = Math.min(1, Math.max(0, box.y_center))
  box.width = Math.min(1, Math.max(0.001, box.width))
  box.height = Math.min(1, Math.max(0.001, box.height))
  const next = props.poses.slice()
  next[index] = box
  emitPoses(next)
}

function applyResize(handle: Handle, ox: number, oy: number, obw: number, obh: number, mx: number, my: number) {
  let x = ox
  let y = oy
  let bw = obw
  let bh = obh
  const right = ox + obw
  const bottom = oy + obh
  if (handle.includes('e')) bw = Math.max(MIN_BOX, mx - ox)
  if (handle.includes('s')) bh = Math.max(MIN_BOX, my - oy)
  if (handle.includes('w')) {
    const nx = Math.min(mx, right - MIN_BOX)
    bw = right - nx
    x = nx
  }
  if (handle.includes('n')) {
    const ny = Math.min(my, bottom - MIN_BOX)
    bh = bottom - ny
    y = ny
  }
  return clampRect(x, y, bw, bh)
}

function placeOrMoveKpt(index: number, kptIndex: number, x: number, y: number, v = 2) {
  const { w, h } = displaySize.value
  const next = props.poses.slice()
  const cur = padPose(next[index])
  const kpts = cur.keypoints.slice()
  kpts[kptIndex] = {
    x: Math.min(1, Math.max(0, x / w)),
    y: Math.min(1, Math.max(0, y / h)),
    v,
  }
  next[index] = { ...cur, keypoints: kpts }
  activeKpt.value = Math.min(kptN.value - 1, kptIndex + 1)
  emitPoses(next)
}

function cycleVisibility(index: number, kptIndex: number) {
  const next = props.poses.slice()
  const cur = padPose(next[index])
  const kpts = cur.keypoints.slice()
  const curV = kpts[kptIndex]?.v ?? 0
  const nv = curV === 0 ? 2 : curV === 2 ? 1 : 0
  const { w, h } = displaySize.value
  const { x, y, bw, bh } = yoloToPixel(cur)
  kpts[kptIndex] = {
    x: kpts[kptIndex]?.v ? kpts[kptIndex].x : (x + bw / 2) / w,
    y: kpts[kptIndex]?.v ? kpts[kptIndex].y : (y + bh / 2) / h,
    v: nv,
  }
  next[index] = { ...cur, keypoints: kpts }
  emitPoses(next)
}

function onDown(e: MouseEvent) {
  if (e.button === 2) return
  if (Date.now() < suppressDrawUntil) {
    mode = 'none'
    return
  }
  const { x, y } = canvasPos(e)

  if (selected.value >= 0) {
    const kh = hitTestKpt(x, y, selected.value)
    if (kh >= 0) {
      mode = 'kpt'
      dragKpt = kh
      activeKpt.value = kh
      return
    }
    const handle = hitTestHandle(x, y, selected.value)
    if (handle) {
      mode = 'resize'
      resizeHandle = handle
      originRect = yoloToPixel(padPose(props.poses[selected.value]))
      return
    }
  }

  const hit = hitTestBox(x, y)
  if (hit >= 0) {
    const cur = padPose(props.poses[hit])
    // 首次点选：仅选中；再次点击框内：放置下一个未标关键点
    if (hit !== selected.value) {
      selected.value = hit
      activeKpt.value = nextUnsetKpt(cur)
      mode = 'none'
      draw()
      return
    }
    selected.value = hit
    const allSet = cur.keypoints.every((k) => k.v > 0)
    if (e.shiftKey || !allSet) {
      const ki = nextUnsetKpt(cur)
      placeOrMoveKpt(hit, ki, x, y, 2)
      mode = 'none'
      draw()
      return
    }
    mode = 'move'
    startX = x
    startY = y
    originRect = yoloToPixel(cur)
    draw()
    return
  }

  mode = 'draw'
  startX = curX = x
  startY = curY = y
  selected.value = -1
  draw()
}

function onMove(e: MouseEvent) {
  const p = canvasPos(e)
  curX = p.x
  curY = p.y

  if (mode === 'none') {
    if (selected.value >= 0) {
      if (hitTestKpt(p.x, p.y, selected.value) >= 0) {
        hoverCursor.value = 'pointer'
        return
      }
      const handle = hitTestHandle(p.x, p.y, selected.value)
      if (handle) {
        hoverCursor.value = 'nwse-resize'
        return
      }
    }
    hoverCursor.value = hitTestBox(p.x, p.y) >= 0 ? 'move' : 'crosshair'
    return
  }

  if (mode === 'draw') {
    draw()
    return
  }
  if (mode === 'move' && selected.value >= 0) {
    const dx = p.x - startX
    const dy = p.y - startY
    commitBox(selected.value, originRect.x + dx, originRect.y + dy, originRect.bw, originRect.bh)
    return
  }
  if (mode === 'resize' && selected.value >= 0 && resizeHandle) {
    const r = applyResize(resizeHandle, originRect.x, originRect.y, originRect.bw, originRect.bh, p.x, p.y)
    commitBox(selected.value, r.x, r.y, r.bw, r.bh)
    return
  }
  if (mode === 'kpt' && selected.value >= 0 && dragKpt >= 0) {
    placeOrMoveKpt(selected.value, dragKpt, p.x, p.y, 2)
  }
}

function onUp(e: MouseEvent) {
  if (mode === 'draw') {
    const x = Math.min(startX, curX)
    const y = Math.min(startY, curY)
    const bw = Math.abs(curX - startX)
    const bh = Math.abs(curY - startY)
    if (bw >= MIN_BOX && bh >= MIN_BOX) {
      const clamped = clampRect(x, y, bw, bh)
      const pose = pixelToPose(
        clamped.x,
        clamped.y,
        clamped.bw,
        clamped.bh,
        props.classId,
        emptyKeypoints(),
      )
      const next = [...props.poses, pose]
      selected.value = next.length - 1
      activeKpt.value = 0
      emitPoses(next)
    }
  }
  mode = 'none'
  resizeHandle = null
  dragKpt = -1
  draw()
}

function onDblClick(e: MouseEvent) {
  const { x, y } = canvasPos(e)
  const hit = hitTestBox(x, y)
  if (hit >= 0) {
    suppressDrawUntil = Date.now() + 300
    deleteAt(hit)
  }
}

function onContextMenu(e: MouseEvent) {
  e.preventDefault()
  if (selected.value < 0) return
  const { x, y } = canvasPos(e)
  let ki = hitTestKpt(x, y, selected.value)
  if (ki < 0) ki = activeKpt.value
  cycleVisibility(selected.value, ki)
  draw()
}

function onKey(e: KeyboardEvent) {
  const tag = (e.target as HTMLElement)?.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA') return
  if (e.key === 'Delete' || e.key === 'Backspace') {
    if (selected.value >= 0) {
      e.preventDefault()
      deleteAt(selected.value)
    }
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault()
    emit('navigate', 'prev')
  } else if (e.key === 'ArrowRight') {
    e.preventDefault()
    emit('navigate', 'next')
  } else if (e.key === ']' || e.key === '[') {
    e.preventDefault()
    const d = e.key === ']' ? 1 : -1
    activeKpt.value = (activeKpt.value + d + kptN.value) % kptN.value
    draw()
  }
}

watch(
  () => props.imageUrl,
  (url) => {
    if (url) void loadImage(url)
  },
  { immediate: true },
)

watch(
  () => [props.poses, selected.value, activeKpt.value, props.skeleton, props.kptNames] as const,
  () => draw(),
  { deep: true },
)

onMounted(() => {
  window.addEventListener('keydown', onKey)
  window.addEventListener('resize', fitScale)
})
onUnmounted(() => {
  window.removeEventListener('keydown', onKey)
  window.removeEventListener('resize', fitScale)
})
</script>

<template>
  <div ref="wrapRef" class="pose-wrap">
    <canvas
      ref="canvasRef"
      class="pose-canvas"
      :style="{ cursor: hoverCursor }"
      @mousedown="onDown"
      @mousemove="onMove"
      @mouseup="onUp"
      @mouseleave="onUp"
      @dblclick="onDblClick"
      @contextmenu="onContextMenu"
    />
    <p class="hint">
      画框后选中实例，单击框内放置关键点（当前 #{{ activeKpt }}
      {{ kptNames[activeKpt] || '' }}）；拖点调整；右键切换可见性；[ ] 切换关键点序号
    </p>
  </div>
</template>

<style scoped>
.pose-wrap {
  width: 100%;
  height: 100%;
  min-height: 420px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}
.pose-canvas {
  max-width: 100%;
  background: #0f172a0a;
  border-radius: 8px;
}
.hint {
  margin: 0;
  font-size: 12px;
  color: #64748b;
  text-align: center;
  max-width: 720px;
  line-height: 1.4;
}
</style>
