<script setup lang="ts">
/**
 * 检测任务：矩形框标注画布（YOLO bbox）。
 * 支持：新建框、选中、拖动移动、八向缩放、双击/Delete 删除。
 */
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import type { BBox } from '@/api/datasets'
import { classAccent } from '@/utils/classColors'

const props = defineProps<{
  imageUrl: string
  boxes: BBox[]
  classId: number
  classes: string[]
}>()

const emit = defineEmits<{
  'update:boxes': [boxes: BBox[]]
  /** 请求切换图片：由外层处理保存与翻页 */
  navigate: [dir: 'prev' | 'next']
}>()

type Handle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w'
type Mode = 'none' | 'draw' | 'move' | 'resize'

const HANDLE_SIZE = 8
const MIN_BOX = 6

const canvasRef = ref<HTMLCanvasElement | null>(null)
const wrapRef = ref<HTMLDivElement | null>(null)
const scale = ref(1)
const imgNatural = ref({ w: 0, h: 0 })
const imgEl = ref<HTMLImageElement | null>(null)
const selected = ref(-1)
const hoverCursor = ref('crosshair')

let mode: Mode = 'none'
let resizeHandle: Handle | null = null
let startX = 0
let startY = 0
let curX = 0
let curY = 0
/** 开始拖动/缩放时的框像素快照 */
let originRect = { x: 0, y: 0, bw: 0, bh: 0 }
/** 抑制双击第二次 mousedown 后的 mouseup 误建框 */
let suppressDrawUntil = 0

const displaySize = computed(() => ({
  w: Math.round(imgNatural.value.w * scale.value),
  h: Math.round(imgNatural.value.h * scale.value),
}))

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

  props.boxes.forEach((b, i) => {
    const { x, y, bw, bh } = yoloToPixel(b)
    const isSel = i === selected.value
    const accent = classAccent(b.class_id)
    ctx.strokeStyle = accent
    ctx.lineWidth = isSel ? 3 : 2
    ctx.strokeRect(x, y, bw, bh)
    // 选中：白色外描边，便于在同类色块中辨认
    if (isSel) {
      ctx.strokeStyle = '#ffffffcc'
      ctx.lineWidth = 1.5
      ctx.strokeRect(x - 1, y - 1, bw + 2, bh + 2)
      ctx.strokeStyle = accent
      ctx.lineWidth = 3
      ctx.strokeRect(x, y, bw, bh)
    }

    // 类别标签：彩色底 + 白字
    const label = props.classes[b.class_id] || `类${b.class_id}`
    ctx.font = '12px sans-serif'
    const padX = 4
    const padY = 2
    const tw = ctx.measureText(label).width
    const th = 14
    const boxW = tw + padX * 2
    const boxH = th + padY * 2
    let lx = x
    let ly = y - boxH
    if (ly < 0) ly = y
    if (lx + boxW > w) lx = Math.max(0, w - boxW)
    ctx.fillStyle = accent
    ctx.fillRect(lx, ly, boxW, boxH)
    ctx.fillStyle = '#ffffff'
    ctx.textBaseline = 'top'
    ctx.fillText(label, lx + padX, ly + padY)

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

function yoloToPixel(b: BBox) {
  const { w, h } = displaySize.value
  const bw = b.width * w
  const bh = b.height * h
  const x = b.x_center * w - bw / 2
  const y = b.y_center * h - bh / 2
  return { x, y, bw, bh }
}

function pixelToYolo(x: number, y: number, bw: number, bh: number, classId: number): BBox {
  const { w, h } = displaySize.value
  return {
    class_id: classId,
    x_center: (x + bw / 2) / w,
    y_center: (y + bh / 2) / h,
    width: bw / w,
    height: bh / h,
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

function canvasPos(e: MouseEvent) {
  const canvas = canvasRef.value!
  const rect = canvas.getBoundingClientRect()
  return {
    x: ((e.clientX - rect.left) / rect.width) * canvas.width,
    y: ((e.clientY - rect.top) / rect.height) * canvas.height,
  }
}

function hitTestBox(x: number, y: number) {
  for (let i = props.boxes.length - 1; i >= 0; i--) {
    const { x: bx, y: by, bw, bh } = yoloToPixel(props.boxes[i])
    if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) return i
  }
  return -1
}

function hitTestHandle(x: number, y: number, index: number): Handle | null {
  if (index < 0 || index >= props.boxes.length) return null
  const { x: bx, y: by, bw, bh } = yoloToPixel(props.boxes[index])
  const centers = handleCenters(bx, by, bw, bh)
  const tol = HANDLE_SIZE
  for (const [name, p] of Object.entries(centers) as [Handle, { x: number; y: number }][]) {
    if (Math.abs(x - p.x) <= tol && Math.abs(y - p.y) <= tol) return name
  }
  return null
}

function cursorForHandle(h: Handle | null): string {
  if (!h) return 'move'
  const map: Record<Handle, string> = {
    nw: 'nwse-resize',
    se: 'nwse-resize',
    ne: 'nesw-resize',
    sw: 'nesw-resize',
    n: 'ns-resize',
    s: 'ns-resize',
    e: 'ew-resize',
    w: 'ew-resize',
  }
  return map[h]
}

function deleteBoxAt(index: number) {
  if (index < 0 || index >= props.boxes.length) return
  const next = props.boxes.filter((_, i) => i !== index)
  selected.value = -1
  emit('update:boxes', next)
}

function commitBoxAt(index: number, x: number, y: number, bw: number, bh: number) {
  const clamped = clampRect(x, y, bw, bh)
  const classId = props.boxes[index]?.class_id ?? props.classId
  const box = pixelToYolo(clamped.x, clamped.y, clamped.bw, clamped.bh, classId)
  box.x_center = Math.min(1, Math.max(0, box.x_center))
  box.y_center = Math.min(1, Math.max(0, box.y_center))
  box.width = Math.min(1, Math.max(0.001, box.width))
  box.height = Math.min(1, Math.max(0.001, box.height))
  const next = props.boxes.slice()
  next[index] = box
  emit('update:boxes', next)
}

function applyResize(handle: Handle, ox: number, oy: number, obw: number, obh: number, mx: number, my: number) {
  let x = ox
  let y = oy
  let bw = obw
  let bh = obh
  const right = ox + obw
  const bottom = oy + obh

  if (handle.includes('e')) {
    bw = Math.max(MIN_BOX, mx - ox)
  }
  if (handle.includes('s')) {
    bh = Math.max(MIN_BOX, my - oy)
  }
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

function onDown(e: MouseEvent) {
  if (Date.now() < suppressDrawUntil) {
    mode = 'none'
    return
  }
  const { x, y } = canvasPos(e)

  // 优先：已选中框的缩放手柄
  if (selected.value >= 0) {
    const handle = hitTestHandle(x, y, selected.value)
    if (handle) {
      mode = 'resize'
      resizeHandle = handle
      startX = x
      startY = y
      originRect = yoloToPixel(props.boxes[selected.value])
      return
    }
  }

  const hit = hitTestBox(x, y)
  if (hit >= 0) {
    selected.value = hit
    mode = 'move'
    resizeHandle = null
    startX = x
    startY = y
    originRect = yoloToPixel(props.boxes[hit])
    draw()
    return
  }

  // 空白处新建框
  mode = 'draw'
  resizeHandle = null
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
    // 悬停光标提示
    if (selected.value >= 0) {
      const handle = hitTestHandle(p.x, p.y, selected.value)
      if (handle) {
        hoverCursor.value = cursorForHandle(handle)
        return
      }
      const hit = hitTestBox(p.x, p.y)
      hoverCursor.value = hit === selected.value ? 'move' : hit >= 0 ? 'pointer' : 'crosshair'
      return
    }
    hoverCursor.value = hitTestBox(p.x, p.y) >= 0 ? 'pointer' : 'crosshair'
    return
  }

  if (mode === 'draw') {
    draw()
    return
  }

  if (mode === 'move' && selected.value >= 0) {
    const dx = curX - startX
    const dy = curY - startY
    const next = clampRect(originRect.x + dx, originRect.y + dy, originRect.bw, originRect.bh)
    commitBoxAt(selected.value, next.x, next.y, next.bw, next.bh)
    return
  }

  if (mode === 'resize' && selected.value >= 0 && resizeHandle) {
    const next = applyResize(
      resizeHandle,
      originRect.x,
      originRect.y,
      originRect.bw,
      originRect.bh,
      curX,
      curY,
    )
    commitBoxAt(selected.value, next.x, next.y, next.bw, next.bh)
  }
}

function onUp() {
  if (mode === 'draw') {
    if (Date.now() < suppressDrawUntil) {
      mode = 'none'
      draw()
      return
    }
    const x = Math.min(startX, curX)
    const y = Math.min(startY, curY)
    const bw = Math.abs(curX - startX)
    const bh = Math.abs(curY - startY)
    mode = 'none'
    if (bw > 4 && bh > 4) {
      const clamped = clampRect(x, y, bw, bh)
      const box = pixelToYolo(clamped.x, clamped.y, clamped.bw, clamped.bh, props.classId)
      box.x_center = Math.min(1, Math.max(0, box.x_center))
      box.y_center = Math.min(1, Math.max(0, box.y_center))
      box.width = Math.min(1, Math.max(0.001, box.width))
      box.height = Math.min(1, Math.max(0.001, box.height))
      emit('update:boxes', [...props.boxes, box])
      selected.value = props.boxes.length
    }
    draw()
    return
  }

  mode = 'none'
  resizeHandle = null
  draw()
}

/** 左键双击框体删除 */
function onDblClick(e: MouseEvent) {
  e.preventDefault()
  mode = 'none'
  suppressDrawUntil = Date.now() + 300
  const { x, y } = canvasPos(e)
  const hit = hitTestBox(x, y)
  if (hit >= 0) {
    deleteBoxAt(hit)
    draw()
  }
}

function onWheel(e: WheelEvent) {
  e.preventDefault()
  const delta = e.deltaY > 0 ? 0.9 : 1.1
  scale.value = Math.min(3, Math.max(0.2, scale.value * delta))
  nextTick(() => draw())
}

function isTypingTarget(el: EventTarget | null) {
  if (!(el instanceof HTMLElement)) return false
  const tag = el.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable
}

function onKey(e: KeyboardEvent) {
  if (isTypingTarget(e.target)) return

  if (e.key === 'Delete' || e.key === 'Backspace') {
    if (selected.value < 0) return
    e.preventDefault()
    deleteBoxAt(selected.value)
    return
  }

  if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
    e.preventDefault()
    emit('navigate', 'prev')
    return
  }
  if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
    e.preventDefault()
    emit('navigate', 'next')
  }
}

watch(
  () => props.imageUrl,
  async (url) => {
    selected.value = -1
    mode = 'none'
    if (url) await loadImage(url)
  },
)

watch(
  () => props.boxes,
  () => draw(),
  { deep: true },
)

onMounted(() => {
  window.addEventListener('keydown', onKey)
  window.addEventListener('resize', () => {
    fitScale()
    draw()
  })
  if (props.imageUrl) loadImage(props.imageUrl)
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKey)
})
</script>

<template>
  <div ref="wrapRef" class="annotator">
    <canvas
      ref="canvasRef"
      :style="{ cursor: hoverCursor }"
      @mousedown="onDown"
      @mousemove="onMove"
      @mouseup="onUp"
      @mouseleave="onUp"
      @dblclick.prevent="onDblClick"
      @wheel.prevent="onWheel"
    />
    <p class="tip">
      拖拽空白处画框 · 拖动框可移动 · 拖角点/边线可缩放 · 双击或 Delete 删除 · A/← 上一张 · D/→ 下一张 ·
      滚轮缩放 · 切换图片时自动保存
    </p>
  </div>
</template>

<style scoped>
.annotator {
  width: 100%;
  height: 100%;
  min-height: 0;
  box-sizing: border-box;
  background:
    linear-gradient(45deg, #eef2f5 25%, transparent 25%),
    linear-gradient(-45deg, #eef2f5 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #eef2f5 75%),
    linear-gradient(-45deg, transparent 75%, #eef2f5 75%);
  background-size: 20px 20px;
  background-position: 0 0, 0 10px, 10px -10px, -10px 0;
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  overflow: auto;
  display: grid;
  place-items: center;
  padding: 0.5rem;
  grid-template-rows: 1fr auto;
}

canvas {
  max-width: 100%;
  max-height: 100%;
  background: #111;
}

.tip {
  grid-column: 1;
  margin: 0.4rem 0 0;
  color: var(--ink-faint);
  font-size: 0.78rem;
  justify-self: start;
}
</style>
