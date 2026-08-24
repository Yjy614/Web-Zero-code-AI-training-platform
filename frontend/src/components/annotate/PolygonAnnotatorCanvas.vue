<script setup lang="ts">
/**
 * 实例分割：连点多边形标注。
 * 草稿/孤点与已保存多边形一律用归一化坐标，缩放时不会跑偏。
 * 闭合时靠近首点有高亮磁吸。图片加载方式与检测画布一致（Image + decode）。
 */
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import type { PolygonInstance } from '@/api/datasets'
import { classAccent, classFill } from '@/utils/classColors'

const props = defineProps<{
  imageUrl: string
  polygons: PolygonInstance[]
  classId: number
  classes: string[]
}>()

const emit = defineEmits<{
  'update:polygons': [polygons: PolygonInstance[]]
  navigate: [dir: 'prev' | 'next']
}>()

type Pt = { x: number; y: number }

const HIT_PT = 10
const HIT_EDGE = 8
/** 首点磁吸半径（像素，随缩放略放大） */
const SNAP_CLOSE = 22

const canvasRef = ref<HTMLCanvasElement | null>(null)
const wrapRef = ref<HTMLDivElement | null>(null)
const scale = ref(1)
const imgNatural = ref({ w: 0, h: 0 })
const imgEl = ref<HTMLImageElement | null>(null)
const selected = ref(-1)
/** 开链草稿：归一化坐标 */
const draft = ref<Pt[]>([])
/** 孤点：归一化坐标 */
const orphans = ref<Pt[]>([])
/** 靠近首点可闭合 */
const snapToClose = ref(false)
/** 连点时鼠标在画布上的像素位置（用于末点→光标虚线） */
const pointerPx = ref<Pt | null>(null)
const hoverCursor = ref('crosshair')
/** 拖动多边形顶点时的本地预览（避免等父组件回传才重绘） */
const previewPolys = ref<PolygonInstance[] | null>(null)

type DragTarget =
  | { kind: 'poly'; polyIdx: number; ptIdx: number }
  | { kind: 'draft'; ptIdx: number }
  | { kind: 'orphan'; ptIdx: number }
  | null

let dragging: DragTarget = null
let dragMoved = false
let suppressAddUntil = 0
let loadToken = 0
let resizeObserver: ResizeObserver | null = null

const displaySize = computed(() => ({
  w: Math.round(imgNatural.value.w * scale.value),
  h: Math.round(imgNatural.value.h * scale.value),
}))

async function loadImage(url: string) {
  const token = ++loadToken
  if (!url) {
    imgEl.value = null
    imgNatural.value = { w: 0, h: 0 }
    return
  }
  try {
    const img = new Image()
    img.src = url
    await img.decode()
    if (token !== loadToken) return
    if (!img.naturalWidth || !img.naturalHeight) {
      throw new Error('图片尺寸无效')
    }
    imgEl.value = img
    imgNatural.value = { w: img.naturalWidth, h: img.naturalHeight }
    fitScale()
    await nextTick()
    if (token !== loadToken) return
    draw()
    requestAnimationFrame(() => {
      if (token !== loadToken) return
      fitScale()
      draw()
    })
  } catch {
    if (token !== loadToken) return
    imgEl.value = null
    imgNatural.value = { w: 0, h: 0 }
    ElMessage.error('标注图片加载失败，请刷新后重试或检查图片文件')
  }
}

function fitScale() {
  const wrap = wrapRef.value
  if (!wrap || !imgNatural.value.w) return
  const maxW = Math.max(120, wrap.clientWidth - 16)
  const maxH = Math.max(240, wrap.clientHeight - 48)
  const s = Math.min(maxW / imgNatural.value.w, maxH / imgNatural.value.h, 1.5)
  scale.value = Number.isFinite(s) && s > 0 ? s : 1
}

function toNorm(px: number, py: number): Pt {
  const { w, h } = displaySize.value
  return {
    x: Math.min(1, Math.max(0, px / Math.max(w, 1))),
    y: Math.min(1, Math.max(0, py / Math.max(h, 1))),
  }
}

function toPixel(n: Pt): Pt {
  return { x: n.x * displaySize.value.w, y: n.y * displaySize.value.h }
}

/** 鼠标 → 画布内部像素（相对 canvas CSS 尺寸） */
function canvasPos(e: MouseEvent): Pt {
  const canvas = canvasRef.value
  if (!canvas) return { x: 0, y: 0 }
  const r = canvas.getBoundingClientRect()
  const { w, h } = displaySize.value
  const sx = w / Math.max(r.width, 1)
  const sy = h / Math.max(r.height, 1)
  return {
    x: (e.clientX - r.left) * sx,
    y: (e.clientY - r.top) * sy,
  }
}

function dist2(a: Pt, b: Pt) {
  const dx = a.x - b.x
  const dy = a.y - b.y
  return dx * dx + dy * dy
}

function distToSeg(p: Pt, a: Pt, b: Pt) {
  const abx = b.x - a.x
  const aby = b.y - a.y
  const len2 = abx * abx + aby * aby
  if (len2 < 1e-6) return Math.sqrt(dist2(p, a))
  let t = ((p.x - a.x) * abx + (p.y - a.y) * aby) / len2
  t = Math.min(1, Math.max(0, t))
  return Math.hypot(p.x - (a.x + t * abx), p.y - (a.y + t * aby))
}

function pointInPoly(px: number, py: number, pts: Pt[]) {
  let inside = false
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i].x
    const yi = pts[i].y
    const xj = pts[j].x
    const yj = pts[j].y
    const intersect = yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi + 1e-9) + xi
    if (intersect) inside = !inside
  }
  return inside
}

function polyPixels(i: number): Pt[] {
  return activePolys()[i].points.map(toPixel)
}

function activePolys(): PolygonInstance[] {
  return previewPolys.value || props.polygons
}

function draftPixels(): Pt[] {
  return draft.value.map(toPixel)
}

function snapRadius() {
  return Math.max(SNAP_CLOSE, 14 * scale.value)
}

function nearFirstPoint(p: Pt): boolean {
  if (draft.value.length < 3) return false
  return dist2(p, toPixel(draft.value[0])) <= snapRadius() ** 2
}

function emitPolys(next: PolygonInstance[]) {
  emit('update:polygons', next)
}

function clonePolys(): PolygonInstance[] {
  return activePolys().map((p) => ({
    class_id: p.class_id,
    points: p.points.map((q) => ({ x: q.x, y: q.y })),
  }))
}

function hitVertex(p: Pt): DragTarget {
  const r2 = HIT_PT * HIT_PT
  const polys = activePolys()
  for (let i = polys.length - 1; i >= 0; i--) {
    const pts = polyPixels(i)
    for (let j = 0; j < pts.length; j++) {
      if (dist2(p, pts[j]) <= r2) return { kind: 'poly', polyIdx: i, ptIdx: j }
    }
  }
  const d = draftPixels()
  // 草稿首点：可闭合时用更大阈值，便于点到
  for (let j = d.length - 1; j >= 0; j--) {
    const thr = j === 0 && d.length >= 3 ? snapRadius() ** 2 : r2
    if (dist2(p, d[j]) <= thr) return { kind: 'draft', ptIdx: j }
  }
  for (let j = orphans.value.length - 1; j >= 0; j--) {
    if (dist2(p, toPixel(orphans.value[j])) <= r2) return { kind: 'orphan', ptIdx: j }
  }
  return null
}

function hitEdge(p: Pt): { kind: 'poly' | 'draft'; idx: number; edgeIdx: number } | null {
  const polys = activePolys()
  for (let i = polys.length - 1; i >= 0; i--) {
    const pts = polyPixels(i)
    const n = pts.length
    if (n < 2) continue
    for (let j = 0; j < n; j++) {
      if (distToSeg(p, pts[j], pts[(j + 1) % n]) <= HIT_EDGE) {
        return { kind: 'poly', idx: i, edgeIdx: j }
      }
    }
  }
  const d = draftPixels()
  for (let j = 0; j < d.length - 1; j++) {
    if (distToSeg(p, d[j], d[j + 1]) <= HIT_EDGE) {
      return { kind: 'draft', idx: 0, edgeIdx: j }
    }
  }
  return null
}

function hitRegion(p: Pt): number {
  const polys = activePolys()
  for (let i = polys.length - 1; i >= 0; i--) {
    const pts = polyPixels(i)
    if (pts.length >= 3 && pointInPoly(p.x, p.y, pts)) return i
  }
  return -1
}

function closeDraft() {
  if (draft.value.length < 3) return false
  const points = draft.value.map((p) => ({ x: p.x, y: p.y }))
  const next = [...activePolys(), { class_id: props.classId, points }]
  emitPolys(next)
  selected.value = next.length - 1
  draft.value = []
  orphans.value = []
  snapToClose.value = false
  pointerPx.value = null
  draw()
  return true
}

function cancelDraft() {
  // Esc：取消当前连点，不留下孤点
  draft.value = []
  orphans.value = []
  snapToClose.value = false
  pointerPx.value = null
  draw()
}

function drawPolyPath(ctx: CanvasRenderingContext2D, pts: Pt[], closed: boolean) {
  if (!pts.length) return
  ctx.beginPath()
  ctx.moveTo(pts[0].x, pts[0].y)
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
  if (closed && pts.length >= 3) ctx.closePath()
}

function drawVertices(ctx: CanvasRenderingContext2D, pts: Pt[], accent: string, highlightIdx = -1) {
  pts.forEach((p, i) => {
    const hi = i === highlightIdx
    const r = hi ? 7 : 4
    if (hi) {
      // 首点磁吸高亮圈
      ctx.beginPath()
      ctx.arc(p.x, p.y, snapRadius(), 0, Math.PI * 2)
      ctx.strokeStyle = accent + '55'
      ctx.lineWidth = 2
      ctx.stroke()
      ctx.beginPath()
      ctx.arc(p.x, p.y, 11, 0, Math.PI * 2)
      ctx.fillStyle = accent + '33'
      ctx.fill()
    }
    ctx.beginPath()
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2)
    ctx.fillStyle = hi ? accent : '#fff'
    ctx.fill()
    ctx.strokeStyle = accent
    ctx.lineWidth = hi ? 2.5 : 1.5
    ctx.stroke()
  })
}

function draw() {
  const canvas = canvasRef.value
  const img = imgEl.value
  if (!canvas || !img) return
  // 不再保留孤点
  if (orphans.value.length) orphans.value = []
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const { w, h } = displaySize.value
  if (w <= 0 || h <= 0) return
  canvas.width = w
  canvas.height = h
  ctx.clearRect(0, 0, w, h)
  ctx.drawImage(img, 0, 0, w, h)

  activePolys().forEach((poly, i) => {
    const pts = poly.points.map(toPixel)
    const isSel = i === selected.value
    const accent = classAccent(poly.class_id)
    drawPolyPath(ctx, pts, true)
    if (pts.length >= 3) {
      ctx.fillStyle = classFill(poly.class_id, isSel ? '44' : '33')
      ctx.fill()
    }
    ctx.strokeStyle = accent
    ctx.lineWidth = isSel ? 3 : 2
    ctx.stroke()
    if (isSel) {
      ctx.strokeStyle = '#ffffffaa'
      ctx.lineWidth = 1.25
      ctx.stroke()
      ctx.strokeStyle = accent
      ctx.lineWidth = 3
      ctx.stroke()
    }
    drawVertices(ctx, pts, accent)
    const label = props.classes[poly.class_id] || `类${poly.class_id}`
    if (pts.length) {
      ctx.font = '12px sans-serif'
      const tw = ctx.measureText(label).width
      const lx = pts[0].x
      const ly = Math.max(0, pts[0].y - 18)
      ctx.fillStyle = accent
      ctx.fillRect(lx, ly, tw + 8, 16)
      ctx.fillStyle = '#fff'
      ctx.textBaseline = 'top'
      ctx.fillText(label, lx + 4, ly + 2)
    }
  })

  if (draft.value.length) {
    const accent = classAccent(props.classId)
    const pts = draftPixels()
    drawPolyPath(ctx, pts, false)
    ctx.strokeStyle = accent
    ctx.lineWidth = 2
    ctx.stroke()

    // 末点到鼠标的橡皮筋虚线；磁吸闭合时接到首点
    const last = pts[pts.length - 1]
    if (last && pointerPx.value && !dragging) {
      let end = pointerPx.value
      if (snapToClose.value && pts.length >= 3) {
        end = pts[0]
      }
      ctx.setLineDash([6, 4])
      ctx.beginPath()
      ctx.moveTo(last.x, last.y)
      ctx.lineTo(end.x, end.y)
      ctx.strokeStyle = snapToClose.value ? accent : accent + '99'
      ctx.lineWidth = snapToClose.value ? 2.5 : 1.5
      ctx.stroke()
      ctx.setLineDash([])
      // 闭合预览：半透明填充，便于看最终轮廓
      if (snapToClose.value && pts.length >= 3) {
        drawPolyPath(ctx, pts, true)
        ctx.fillStyle = classFill(props.classId, '22')
        ctx.fill()
      }
    }

    drawVertices(ctx, pts, accent, snapToClose.value ? 0 : -1)
  }
}

function bindDragListeners() {
  window.addEventListener('mousemove', onDragMove)
  window.addEventListener('mouseup', onDragEnd)
}

function unbindDragListeners() {
  window.removeEventListener('mousemove', onDragMove)
  window.removeEventListener('mouseup', onDragEnd)
}

function applyDragTo(p: Pt) {
  if (!dragging) return
  dragMoved = true
  const n = toNorm(p.x, p.y)
  if (dragging.kind === 'poly') {
    const next = clonePolys()
    next[dragging.polyIdx].points[dragging.ptIdx] = n
    previewPolys.value = next
    emitPolys(next)
    draw()
  } else if (dragging.kind === 'draft') {
    const d = [...draft.value]
    d[dragging.ptIdx] = n
    draft.value = d
    draw()
  } else if (dragging.kind === 'orphan') {
    const o = [...orphans.value]
    o[dragging.ptIdx] = n
    orphans.value = o
    draw()
  }
}

function onDragMove(e: MouseEvent) {
  if (!dragging) return
  e.preventDefault()
  applyDragTo(canvasPos(e))
}

function onDragEnd() {
  if (dragging && !dragMoved && dragging.kind === 'poly') {
    selected.value = dragging.polyIdx
  }
  dragging = null
  previewPolys.value = null
  hoverCursor.value = 'crosshair'
  unbindDragListeners()
  draw()
}

function onDown(e: MouseEvent) {
  if (e.button === 2) {
    e.preventDefault()
    cancelDraft()
    return
  }
  if (e.button !== 0) return
  // 双击第二次按下不加点，交给 onDblClick
  if (e.detail >= 2 || Date.now() < suppressAddUntil) return
  if (!imgEl.value) return

  const p = canvasPos(e)

  // 磁吸闭合：优先于拖动首点
  if (nearFirstPoint(p)) {
    closeDraft()
    return
  }

  const vHit = hitVertex(p)
  if (vHit) {
    if (vHit.kind === 'draft' && vHit.ptIdx === 0 && draft.value.length >= 3) {
      closeDraft()
      return
    }
    dragging = vHit
    dragMoved = false
    if (vHit.kind === 'poly') selected.value = vHit.polyIdx
    hoverCursor.value = 'grabbing'
    bindDragListeners()
    return
  }

  const eHit = hitEdge(p)
  if (eHit) {
    if (eHit.kind === 'poly') selected.value = eHit.idx
    draw()
    return
  }

  if (!draft.value.length) {
    const r = hitRegion(p)
    if (r >= 0) {
      selected.value = r
      draw()
      return
    }
    selected.value = -1
  }

  draft.value = [...draft.value, toNorm(p.x, p.y)]
  snapToClose.value = false
  pointerPx.value = p
  draw()
}

function onMove(e: MouseEvent) {
  // 拖动由 window 监听，避免移出顶点小圆或画布就断
  if (dragging) return
  if (!imgEl.value) return

  const p = canvasPos(e)
  pointerPx.value = p
  const canSnap = nearFirstPoint(p)
  snapToClose.value = canSnap
  // 连点过程中持续重绘，显示末点→鼠标虚线
  if (draft.value.length) draw()
  else if (canSnap) draw()

  if (canSnap) hoverCursor.value = 'crosshair'
  else if (hitVertex(p)) hoverCursor.value = 'grab'
  else if (hitEdge(p)) hoverCursor.value = 'pointer'
  else if (!draft.value.length && hitRegion(p) >= 0) hoverCursor.value = 'pointer'
  else hoverCursor.value = 'crosshair'
}

function onPointerLeave() {
  pointerPx.value = null
  snapToClose.value = false
  if (draft.value.length) draw()
}

function deletePoly(i: number) {
  emitPolys(activePolys().filter((_, idx) => idx !== i))
  if (selected.value === i) selected.value = -1
  else if (selected.value > i) selected.value -= 1
}

function deleteVertexAt(target: NonNullable<DragTarget>) {
  if (target.kind === 'orphan') {
    orphans.value = orphans.value.filter((_, i) => i !== target.ptIdx)
    draw()
    return
  }
  if (target.kind === 'draft') {
    const d = [...draft.value]
    d.splice(target.ptIdx, 1)
    draft.value = d
    snapToClose.value = false
    draw()
    return
  }
  const { polyIdx, ptIdx } = target
  const list = activePolys()
  const pts = list[polyIdx].points.map((q) => ({ x: q.x, y: q.y }))
  const cls = list[polyIdx].class_id
  pts.splice(ptIdx, 1)
  if (pts.length >= 3) {
    const next = clonePolys()
    next[polyIdx] = { class_id: cls, points: pts }
    emitPolys(next)
  } else if (pts.length === 2) {
    deletePoly(polyIdx)
    // 剩余两点变为新草稿；旧草稿丢弃（不产生孤点）
    draft.value = pts
  } else if (pts.length === 1) {
    // 单点即孤点：自动删除
    deletePoly(polyIdx)
  } else {
    deletePoly(polyIdx)
  }
  draw()
}

function deleteEdgeAt(edge: { kind: 'poly' | 'draft'; idx: number; edgeIdx: number }) {
  if (edge.kind === 'draft') {
    const d = draft.value
    const j = edge.edgeIdx
    const left = d.slice(0, j + 1)
    const right = d.slice(j + 1)
    const parts = [left, right].filter((x) => x.length >= 2)
    // 长度 < 2 的片段视为孤点，自动丢弃；多段时保留最长的一段作草稿
    parts.sort((a, b) => b.length - a.length)
    draft.value = parts[0] ? [...parts[0]] : []
    snapToClose.value = false
    draw()
    return
  }

  const list = activePolys()
  const pts = list[edge.idx].points.map((q) => ({ x: q.x, y: q.y }))
  const n = pts.length
  const ei = edge.edgeIdx
  const chain: Pt[] = []
  for (let k = 1; k <= n; k++) {
    chain.push(pts[(ei + k) % n])
  }
  deletePoly(edge.idx)
  // 断开后不足 2 点则整段丢弃（孤点自动删除）
  draft.value = chain.length >= 2 ? chain : []
  snapToClose.value = false
  draw()
}

function onDblClick(e: MouseEvent) {
  e.preventDefault()
  suppressAddUntil = Date.now() + 280
  dragging = null
  snapToClose.value = false

  const p = canvasPos(e)
  const vHit = hitVertex(p)
  if (vHit) {
    // 双击首点且可闭合：视为闭合，而不是删点
    if (vHit.kind === 'draft' && vHit.ptIdx === 0 && draft.value.length >= 3) {
      closeDraft()
      return
    }
    deleteVertexAt(vHit)
    return
  }
  const eHit = hitEdge(p)
  if (eHit) {
    deleteEdgeAt(eHit)
    return
  }
  const r = hitRegion(p)
  if (r >= 0) {
    deletePoly(r)
    draw()
  }
}

function onKey(e: KeyboardEvent) {
  const tag = (e.target as HTMLElement)?.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA') return
  if (e.key === 'Enter') {
    e.preventDefault()
    closeDraft()
    return
  }
  if (e.key === 'Escape') {
    e.preventDefault()
    cancelDraft()
    return
  }
  if (e.key === 'Delete' || e.key === 'Backspace') {
    e.preventDefault()
    if (draft.value.length) {
      draft.value = draft.value.slice(0, -1)
      snapToClose.value = false
      draw()
      return
    }
    if (orphans.value.length && selected.value < 0) {
      orphans.value = orphans.value.slice(0, -1)
      draw()
      return
    }
    if (selected.value >= 0) {
      deletePoly(selected.value)
      draw()
    }
    return
  }
  if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') {
    e.preventDefault()
    emit('navigate', 'prev')
  } else if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') {
    e.preventDefault()
    emit('navigate', 'next')
  }
}

function onWheel(e: WheelEvent) {
  const factor = e.deltaY > 0 ? 0.9 : 1.1
  scale.value = Math.min(3, Math.max(0.2, scale.value * factor))
  // 归一化坐标无需换算，直接重绘即可贴合图片
  draw()
}

watch(
  () => props.imageUrl,
  async (url) => {
    selected.value = -1
    draft.value = []
    orphans.value = []
    snapToClose.value = false
    pointerPx.value = null
    previewPolys.value = null
    unbindDragListeners()
    dragging = null
    if (url) await loadImage(url)
  },
)

watch(
  () => props.polygons,
  () => {
    if (!dragging) draw()
  },
  { deep: true },
)

watch(displaySize, () => {
  if (imgEl.value) draw()
})

onMounted(() => {
  window.addEventListener('keydown', onKey)
  window.addEventListener('resize', () => {
    fitScale()
    draw()
  })
  nextTick(() => {
    if (typeof ResizeObserver !== 'undefined' && wrapRef.value) {
      resizeObserver = new ResizeObserver(() => {
        if (!imgEl.value) return
        fitScale()
        draw()
      })
      resizeObserver.observe(wrapRef.value)
    }
  })
  if (props.imageUrl) loadImage(props.imageUrl)
})

onUnmounted(() => {
  loadToken += 1
  window.removeEventListener('keydown', onKey)
  unbindDragListeners()
  resizeObserver?.disconnect()
  resizeObserver = null
})
</script>

<template>
  <div ref="wrapRef" class="annotator" @contextmenu.prevent>
    <canvas
      ref="canvasRef"
      :style="{ cursor: hoverCursor }"
      @mousedown="onDown"
      @mousemove="onMove"
      @mouseleave="onPointerLeave"
      @dblclick.prevent="onDblClick"
      @wheel.prevent="onWheel"
    />
    <p class="tip">
      单击连点（虚线跟随鼠标）· 拖动顶点 · 靠近首点磁吸闭合 · 双击区域删整块 · 双击边/点删除 ·
      无连线的单点会自动清除 · Enter 闭合 · Esc 取消 · 滚轮缩放
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
  background-position:
    0 0,
    0 10px,
    10px -10px,
    -10px 0;
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
  font-size: 0.78rem;
  color: var(--muted);
  text-align: center;
}
</style>
