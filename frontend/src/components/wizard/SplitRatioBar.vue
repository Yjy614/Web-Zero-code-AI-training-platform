<script setup lang="ts">
/**
 * 训练 / 验证 / 测试 双柄分段比例条。
 * 拖动两个分界点调整比例，三者始终合计 100%。
 */
import { computed, onUnmounted, ref } from 'vue'

const props = defineProps<{
  trainRatio: number
  valRatio: number
  /** 数据集当前可用图片总数，用于预估各集张数 */
  imageCount?: number
}>()

const emit = defineEmits<{
  'update:trainRatio': [value: number]
  'update:valRatio': [value: number]
}>()

const STEP = 1
const TRAIN_MIN = 50
const TRAIN_MAX = 90
const VAL_MIN = 5
const VAL_MAX = 40
const TEST_MIN = 5

const trackRef = ref<HTMLElement | null>(null)
/** 当前拖动的手柄：train 为训练|验证分界，val 为验证|测试分界 */
const dragging = ref<'train' | 'val' | null>(null)

const trainPct = computed(() => Math.round(Number(props.trainRatio) * 100))
const valPct = computed(() => Math.round(Number(props.valRatio) * 100))
const testPct = computed(() => Math.max(0, 100 - trainPct.value - valPct.value))

/**
 * 按当前比例预估各集张数；保证 train+val+test === 总张数。
 */
const counts = computed(() => {
  const n = Math.max(0, Math.floor(Number(props.imageCount) || 0))
  if (n <= 0) return { train: 0, val: 0, test: 0 }

  let tr = Math.max(0, Number(props.trainRatio) || 0)
  let vr = Math.max(0, Number(props.valRatio) || 0)
  let te = Math.max(0, 1 - tr - vr)
  const sumR = tr + vr + te
  if (sumR <= 0) return { train: n, val: 0, test: 0 }
  tr /= sumR
  vr /= sumR
  te /= sumR

  // 先 floor，再用最大余数法把差额补齐，确保三者和为 n
  const floors = [
    { key: 'train' as const, r: tr, n: Math.floor(n * tr), frac: n * tr - Math.floor(n * tr) },
    { key: 'val' as const, r: vr, n: Math.floor(n * vr), frac: n * vr - Math.floor(n * vr) },
    { key: 'test' as const, r: te, n: Math.floor(n * te), frac: n * te - Math.floor(n * te) },
  ]
  let left = n - floors.reduce((s, x) => s + x.n, 0)
  floors
    .slice()
    .sort((a, b) => b.frac - a.frac)
    .forEach((x) => {
      if (left <= 0) return
      x.n += 1
      left -= 1
    })

  let nTrain = floors.find((x) => x.key === 'train')!.n
  let nVal = floors.find((x) => x.key === 'val')!.n
  let nTest = floors.find((x) => x.key === 'test')!.n

  // 图片足够时，尽量保证有比例的集合至少 1 张（从训练挪）
  if (te > 0 && n >= 3 && nTest < 1 && nTrain > 1) {
    nTrain -= 1
    nTest += 1
  }
  if (vr > 0 && n >= 2 && nVal < 1 && nTrain > 1) {
    nTrain -= 1
    nVal += 1
  }

  // 最终兜底：强制三者和等于 n
  nTest = n - nTrain - nVal
  if (nTest < 0) {
    nVal = Math.max(0, nVal + nTest)
    nTest = n - nTrain - nVal
    if (nTest < 0) {
      nTrain = Math.max(0, nTrain + nTest)
      nTest = n - nTrain - nVal
    }
  }
  return { train: nTrain, val: nVal, test: nTest }
})

function formatLegend(label: string, count: number) {
  if ((props.imageCount ?? 0) > 0) return `${label} ${count} 张`
  return label
}

/** 左分界（训练结束）、右分界（验证结束）位置 % */
const leftPos = computed(() => trainPct.value)
const rightPos = computed(() => trainPct.value + valPct.value)

function snap(pct: number) {
  return Math.round(pct / STEP) * STEP
}

function applyPercents(train: number, val: number) {
  let t = Math.min(TRAIN_MAX, Math.max(TRAIN_MIN, snap(train)))
  let v = Math.min(VAL_MAX, Math.max(VAL_MIN, snap(val)))
  if (t + v > 100 - TEST_MIN) {
    v = Math.max(VAL_MIN, 100 - TEST_MIN - t)
    v = snap(v)
  }
  if (t + v > 100 - TEST_MIN) {
    t = Math.max(TRAIN_MIN, 100 - TEST_MIN - v)
    t = snap(t)
  }
  emit('update:trainRatio', Math.round(t) / 100)
  emit('update:valRatio', Math.round(v) / 100)
}

function clientToPercent(clientX: number) {
  const el = trackRef.value
  if (!el) return 0
  const rect = el.getBoundingClientRect()
  if (rect.width <= 0) return 0
  const raw = ((clientX - rect.left) / rect.width) * 100
  return Math.min(100, Math.max(0, raw))
}

function onPointerDown(which: 'train' | 'val', e: PointerEvent) {
  e.preventDefault()
  e.stopPropagation()
  dragging.value = which
  ;(e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId)
  window.addEventListener('pointermove', onPointerMove)
  window.addEventListener('pointerup', onPointerUp)
  window.addEventListener('pointercancel', onPointerUp)
}

function onTrackPointerDown(e: PointerEvent) {
  // 点击轨道：靠近哪个分界就拖哪个
  const pct = clientToPercent(e.clientX)
  const dLeft = Math.abs(pct - leftPos.value)
  const dRight = Math.abs(pct - rightPos.value)
  dragging.value = dLeft <= dRight ? 'train' : 'val'
  moveTo(pct)
  window.addEventListener('pointermove', onPointerMove)
  window.addEventListener('pointerup', onPointerUp)
  window.addEventListener('pointercancel', onPointerUp)
}

function onPointerMove(e: PointerEvent) {
  if (!dragging.value) return
  moveTo(clientToPercent(e.clientX))
}

function onPointerUp() {
  dragging.value = null
  window.removeEventListener('pointermove', onPointerMove)
  window.removeEventListener('pointerup', onPointerUp)
  window.removeEventListener('pointercancel', onPointerUp)
}

function moveTo(pct: number) {
  const p = snap(pct)
  if (dragging.value === 'train') {
    // 固定右分界，移动训练|验证边界
    const right = rightPos.value
    const minLeft = Math.max(TRAIN_MIN, right - VAL_MAX)
    const maxLeft = Math.min(TRAIN_MAX, right - VAL_MIN)
    const left = Math.min(maxLeft, Math.max(minLeft, p))
    applyPercents(left, right - left)
  } else if (dragging.value === 'val') {
    // 固定左分界，移动验证|测试边界
    const left = leftPos.value
    const minRight = left + VAL_MIN
    const maxRight = Math.min(100 - TEST_MIN, left + VAL_MAX)
    const right = Math.min(maxRight, Math.max(minRight, p))
    applyPercents(left, right - left)
  }
}

function nudge(which: 'train' | 'val', delta: number) {
  dragging.value = which
  if (which === 'train') moveTo(leftPos.value + delta)
  else moveTo(rightPos.value + delta)
  dragging.value = null
}

onUnmounted(() => {
  onPointerUp()
})
</script>

<template>
  <div class="split-ratio">
    <div
      ref="trackRef"
      class="split-track"
      role="group"
      aria-label="数据划分比例"
      @pointerdown="onTrackPointerDown"
    >
      <div class="seg train" :style="{ width: trainPct + '%' }">
        <span v-if="trainPct >= 18">训练 {{ trainPct }}%</span>
      </div>
      <div class="seg val" :style="{ width: valPct + '%' }">
        <span v-if="valPct >= 14">验证 {{ valPct }}%</span>
      </div>
      <div class="seg test" :style="{ width: testPct + '%' }">
        <span v-if="testPct >= 12">测试 {{ testPct }}%</span>
      </div>

      <button
        type="button"
        class="handle"
        :class="{ active: dragging === 'train' }"
        :style="{ left: leftPos + '%' }"
        aria-label="调整训练集与验证集分界"
        :aria-valuenow="trainPct"
        aria-valuemin="50"
        aria-valuemax="90"
        @pointerdown="onPointerDown('train', $event)"
        @keydown.left.prevent="nudge('train', -STEP)"
        @keydown.right.prevent="nudge('train', STEP)"
      />
      <button
        type="button"
        class="handle"
        :class="{ active: dragging === 'val' }"
        :style="{ left: rightPos + '%' }"
        aria-label="调整验证集与测试集分界"
        :aria-valuenow="trainPct + valPct"
        aria-valuemin="55"
        aria-valuemax="95"
        @pointerdown="onPointerDown('val', $event)"
        @keydown.left.prevent="nudge('val', -STEP)"
        @keydown.right.prevent="nudge('val', STEP)"
      />
    </div>

    <div class="split-legend">
      <span class="lg train"><i />{{ formatLegend('训练集', counts.train) }}</span>
      <span class="lg val"><i />{{ formatLegend('验证集', counts.val) }}</span>
      <span class="lg test"><i />{{ formatLegend('测试集', counts.test) }}</span>
    </div>
  </div>
</template>

<style scoped>
.split-ratio {
  display: grid;
  gap: 0.55rem;
  width: 100%;
  user-select: none;
}
.split-track {
  position: relative;
  display: flex;
  height: 36px;
  border-radius: 10px;
  overflow: hidden;
  cursor: pointer;
  box-shadow: inset 0 0 0 1px rgba(15, 61, 79, 0.08);
  background: #eef2f4;
}
.seg {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 0;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  white-space: nowrap;
  transition: width 0.05s linear;
}
.seg span {
  pointer-events: none;
  padding: 0 0.35rem;
  overflow: hidden;
  text-overflow: ellipsis;
}
.seg.train {
  background: linear-gradient(180deg, #2a7a8f 0%, var(--brand) 100%);
  color: #f4fbfc;
}
.seg.val {
  background: linear-gradient(180deg, #5cb3a8 0%, var(--brand-soft) 100%);
  color: #f4fbfc;
}
.seg.test {
  background: linear-gradient(180deg, #d5dde3 0%, #c5ced6 100%);
  color: #4a5a66;
}
.handle {
  position: absolute;
  top: 50%;
  z-index: 2;
  width: 16px;
  height: 28px;
  margin: 0;
  padding: 0;
  border: 2px solid #fff;
  border-radius: 999px;
  background: var(--brand-deep);
  box-shadow: 0 1px 4px rgba(15, 61, 79, 0.35);
  transform: translate(-50%, -50%);
  cursor: ew-resize;
  touch-action: none;
}
.handle::after {
  content: '';
  position: absolute;
  inset: 6px 5px;
  border-radius: 2px;
  background: repeating-linear-gradient(
    to bottom,
    rgba(255, 255, 255, 0.85) 0 2px,
    transparent 2px 5px
  );
}
.handle:hover,
.handle.active,
.handle:focus-visible {
  background: #0a2c38;
  outline: none;
  box-shadow: 0 0 0 3px rgba(26, 95, 122, 0.28);
}
.split-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem 1.1rem;
  font-size: 0.82rem;
  color: var(--ink-muted);
}
.lg {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-weight: 600;
}
.lg i {
  width: 10px;
  height: 10px;
  border-radius: 3px;
  display: inline-block;
}
.lg.train i {
  background: var(--brand);
}
.lg.val i {
  background: var(--brand-soft);
}
.lg.test i {
  background: #c5ced6;
}
</style>
