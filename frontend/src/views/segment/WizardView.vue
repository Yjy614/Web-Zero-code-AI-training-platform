<script setup lang="ts">
/**
 * 分割训练入口：独立 KeepAlive 名称，避免与检测向导缓存互相干扰。
 * 内部复用 WizardView；taskType 在子组件 setup 时按当前路由冻结。
 */
defineOptions({ name: 'SegmentWizard' })
import WizardView from '@/views/detect/WizardView.vue'
</script>

<template>
  <WizardView />
</template>
