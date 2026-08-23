<script setup lang="ts">
/** @deprecated 请优先使用 `@/components/annotate/BBoxAnnotatorCanvas.vue` */
import BBoxAnnotatorCanvas from '@/components/annotate/BBoxAnnotatorCanvas.vue'
import type { BBox } from '@/api/datasets'

const props = defineProps<{
  imageUrl: string
  boxes: BBox[]
  classId: number
  classes: string[]
}>()

const emit = defineEmits<{
  'update:boxes': [boxes: BBox[]]
  navigate: [dir: 'prev' | 'next']
}>()
</script>

<template>
  <BBoxAnnotatorCanvas
    :image-url="props.imageUrl"
    :boxes="props.boxes"
    :class-id="props.classId"
    :classes="props.classes"
    @update:boxes="emit('update:boxes', $event)"
    @navigate="emit('navigate', $event)"
  />
</template>
