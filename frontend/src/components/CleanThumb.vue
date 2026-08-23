<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { fetchImageObjectUrl } from '@/api/datasets'

const props = defineProps<{
  datasetId: number
  name: string
  removed?: boolean
}>()

const url = ref('')
let revoke: string | null = null
let loadSeq = 0

async function load() {
  const seq = ++loadSeq
  try {
    const u = await fetchImageObjectUrl(props.datasetId, props.name, Boolean(props.removed))
    if (seq !== loadSeq) {
      URL.revokeObjectURL(u)
      return
    }
    // 新地址就绪后再替换，避免先清空造成缩略图闪烁
    if (revoke) URL.revokeObjectURL(revoke)
    revoke = u
    url.value = u
  } catch {
    if (seq === loadSeq) {
      url.value = ''
    }
  }
}

watch(
  () => [props.datasetId, props.name, props.removed] as const,
  () => {
    load()
  },
)

onMounted(load)
onUnmounted(() => {
  loadSeq += 1
  if (revoke) URL.revokeObjectURL(revoke)
})
</script>

<template>
  <img v-if="url" :src="url" :alt="name" />
  <span v-else class="ph" />
</template>

<style scoped>
img,
.ph {
  width: 100%;
  height: 56px;
  object-fit: cover;
  display: block;
  background: #e8eef2;
  border-radius: 4px;
}
</style>
