<script setup lang="ts">
/**
 * 姿态估计训练入口：独立 KeepAlive 名称，避免与检测/分割向导缓存互相干扰。
 */
defineOptions({ name: 'PoseWizard' })
import WizardView from '@/views/detect/WizardView.vue'
</script>

<template>
  <WizardView />
</template>
