<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/stores/auth'
import { clearDetectWizardState } from '@/utils/wizardSession'

const router = useRouter()
const auth = useAuthStore()
const loading = ref(false)
const form = reactive({
  username: 'demo',
  password: 'demo123',
})

onMounted(() => {
  // 进入登录页即清空向导进度，避免未登录态残留
  clearDetectWizardState()
})

async function onSubmit() {
  if (loading.value) return
  if (!form.username || !form.password) {
    ElMessage.warning('请输入用户名和密码')
    return
  }
  loading.value = true
  try {
    await auth.login(form.username, form.password)
    ElMessage.success('登录成功')
    router.push('/app/home')
  } catch {
    // 错误已由拦截器提示
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <section class="hero-panel" aria-label="品牌展示">
      <div class="hero-grid" />
      <div class="hero-glow" />
      <div class="hero-content fade-up">
        <p class="brand-mark">ZERO-CODE AI</p>
        <h1 class="brand-title">零代码 AI 训练平台</h1>
        <p class="brand-desc">面向工业场景的目标检测训练工作台 · 内网私有化部署</p>
        <ul class="hero-points">
          <li>七步向导，对齐桌面检测流程</li>
          <li>数据集上云，便于集群训练调度</li>
          <li>浏览器即可完成标注、训练与模型试用</li>
        </ul>
      </div>
    </section>

    <section class="form-panel">
      <div class="form-card fade-up" style="animation-delay: 0.12s">
        <h2>欢迎登录</h2>
        <p class="sub">请使用分配的账号登录</p>

        <el-form label-position="top" @submit.prevent="onSubmit">
          <el-form-item label="用户名">
            <el-input v-model="form.username" size="large" placeholder="请输入用户名" clearable />
          </el-form-item>
          <el-form-item label="密码">
            <el-input
              v-model="form.password"
              size="large"
              type="password"
              show-password
              placeholder="请输入密码"
            />
          </el-form-item>
          <el-button
            type="primary"
            size="large"
            class="submit-btn"
            :loading="loading"
            native-type="submit"
          >
            进入平台
          </el-button>
        </el-form>

        <div class="hint">
          <span>管理员：admin / admin123</span>
          <span>使用人员：demo / demo123</span>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.login-page {
  min-height: 100vh;
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr);
}

.hero-panel {
  position: relative;
  overflow: hidden;
  color: #f4faf9;
  background:
    radial-gradient(ellipse 80% 60% at 20% 20%, rgba(61, 155, 143, 0.35), transparent 55%),
    radial-gradient(ellipse 70% 50% at 80% 80%, rgba(26, 95, 122, 0.55), transparent 50%),
    linear-gradient(145deg, #0f3d4f 0%, #1a5f7a 48%, #165a63 100%);
  display: flex;
  align-items: flex-end;
  padding: clamp(2rem, 5vw, 4.5rem);
}

.hero-grid {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.05) 1px, transparent 1px);
  background-size: 48px 48px;
  animation: grid-drift 28s linear infinite;
  mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.75), transparent 90%);
}

.hero-glow {
  position: absolute;
  width: 420px;
  height: 420px;
  right: -80px;
  top: -60px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(232, 244, 242, 0.18), transparent 68%);
  animation: soft-breathe 5.5s ease-in-out infinite;
}

.hero-content {
  position: relative;
  z-index: 1;
  max-width: 520px;
}

.brand-mark {
  margin: 0 0 0.75rem;
  font-family: var(--font-display);
  letter-spacing: 0.22em;
  font-size: 0.78rem;
  font-weight: 600;
  opacity: 0.78;
}

.brand-title {
  margin: 0;
  font-family: var(--font-display);
  font-size: clamp(2.4rem, 4.5vw, 3.6rem);
  font-weight: 700;
  line-height: 1.15;
  letter-spacing: 0.02em;
}

.brand-desc {
  margin: 1rem 0 1.75rem;
  font-size: 1.05rem;
  line-height: 1.7;
  opacity: 0.88;
  max-width: 28em;
}

.hero-points {
  margin: 0;
  padding: 0;
  list-style: none;
  display: grid;
  gap: 0.65rem;
}

.hero-points li {
  position: relative;
  padding-left: 1.1rem;
  opacity: 0.9;
  font-size: 0.95rem;
}

.hero-points li::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0.55em;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--brand-soft);
}

.form-panel {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  background:
    linear-gradient(180deg, rgba(232, 244, 242, 0.55), transparent 40%),
    var(--surface);
}

.form-card {
  width: min(100%, 400px);
  background: var(--surface-elevated);
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  padding: 2rem 1.75rem 1.5rem;
  box-shadow: var(--shadow-soft);
}

.form-card h2 {
  margin: 0;
  font-family: var(--font-display);
  font-size: 1.65rem;
  font-weight: 650;
}

.sub {
  margin: 0.4rem 0 1.5rem;
  color: var(--ink-muted);
  font-size: 0.92rem;
}

.submit-btn {
  width: 100%;
  margin-top: 0.35rem;
  height: 44px;
  font-weight: 600;
}

.hint {
  margin-top: 1.25rem;
  display: grid;
  gap: 0.35rem;
  color: var(--ink-faint);
  font-size: 0.8rem;
}

@media (max-width: 900px) {
  .login-page {
    grid-template-columns: 1fr;
  }

  .hero-panel {
    min-height: 42vh;
    align-items: flex-end;
  }

  .brand-title {
    font-size: 2.1rem;
  }

  .hero-points {
    display: none;
  }
}
</style>
