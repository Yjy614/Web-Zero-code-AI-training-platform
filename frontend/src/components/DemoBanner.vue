<script setup lang="ts">
defineProps<{ visible: boolean }>()
</script>

<template>
  <transition name="banner">
    <div v-if="visible" class="demo-banner" role="status">
      <span class="dot" />
      <strong>演示模式已开启</strong>
      <span class="sep">·</span>
      <span>禁止自动下载权重；训练 / 评估 / 导出将使用 Mock 执行器跑通流程</span>
    </div>
  </transition>
</template>

<style scoped>
.demo-banner {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  flex-wrap: wrap;
  padding: 0.65rem 1.25rem;
  background: linear-gradient(90deg, #eef7f5, #f7f3e8);
  border-bottom: 1px solid #d7e6e2;
  color: #3a4f4a;
  font-size: 0.86rem;
}

.demo-banner strong {
  font-weight: 650;
  color: var(--brand-deep);
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--brand-soft);
  box-shadow: 0 0 0 0 rgba(61, 155, 143, 0.45);
  animation: soft-breathe 2.4s ease-in-out infinite;
}

.sep {
  opacity: 0.45;
}

.banner-enter-active,
.banner-leave-active {
  transition: all 0.28s ease;
}

.banner-enter-from,
.banner-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}
</style>
