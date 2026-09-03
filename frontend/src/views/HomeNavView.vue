<script setup lang="ts">
/**
 * 登录后导航页：先问「是否明确任务」，再分流到向导或 AI Agent。
 * （试用版；不满意可回退上一版宫格导航）
 */
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import {
  Aim,
  ArrowLeft,
  Box,
  Collection,
  Cpu,
  Crop,
  List,
  MagicStick,
  PictureFilled,
  Setting,
  Share,
  VideoCamera,
} from '@element-plus/icons-vue'
import { useAppStore } from '@/stores/app'
import { useAuthStore } from '@/stores/auth'
import type { MenuVisibility } from '@/api/system'

const router = useRouter()
const app = useAppStore()
const auth = useAuthStore()

/** ask：首问；wizards：已选「明确任务」后选向导 */
const step = ref<'ask' | 'wizards'>('ask')

type NavCard = {
  key: string
  title: string
  desc: string
  path: string
  icon: typeof Aim
  menuKey?: keyof MenuVisibility
}

function visible(menuKey?: keyof MenuVisibility) {
  if (!menuKey) return true
  return app.menuVisibility[menuKey] !== false
}

const showAgent = computed(() => visible('agent_chat'))
const wizardCards = computed((): NavCard[] =>
  (
    [
      {
        key: 'detect',
        title: '目标检测训练',
        desc: '找出目标位置与类别，最常见工业场景',
        path: '/app/detect/wizard',
        icon: Aim,
        menuKey: 'detect_wizard' as const,
      },
      {
        key: 'segment',
        title: '实例分割训练',
        desc: '需要精确轮廓、像素级分割时选用',
        path: '/app/segment/wizard',
        icon: Crop,
        menuKey: 'segment_wizard' as const,
      },
      {
        key: 'pose',
        title: '姿态估计训练',
        desc: '关键点、姿态、骨架类任务',
        path: '/app/pose/wizard',
        icon: Share,
        menuKey: 'pose_wizard' as const,
      },
    ] as NavCard[]
  ).filter((c) => visible(c.menuKey)),
)

const showWizards = computed(() => wizardCards.value.length > 0)

const moreCards = computed((): NavCard[] =>
  (
    [
      {
        key: 'orch',
        title: '流程编排',
        desc: '固定流水线式编排',
        path: '/app/agent/orchestrate',
        icon: List,
        menuKey: 'agent_orchestrate' as const,
      },
      {
        key: 'datasets',
        title: '数据集管理',
        desc: '查看与导入数据',
        path: '/app/resources/datasets',
        icon: Collection,
        menuKey: 'datasets' as const,
      },
      {
        key: 'models',
        title: '模型库',
        desc: '已训练模型',
        path: '/app/resources/models',
        icon: Box,
        menuKey: 'models' as const,
      },
      {
        key: 'infer',
        title: '推理试用',
        desc: '快速看效果',
        path: '/app/resources/infer',
        icon: PictureFilled,
        menuKey: 'infer' as const,
      },
      {
        key: 'weights',
        title: '基础权重',
        desc: '预训练权重',
        path: '/app/resources/weights',
        icon: Cpu,
        menuKey: 'weights' as const,
      },
    ] as NavCard[]
  ).filter((c) => visible(c.menuKey)),
)

function go(path: string) {
  router.push(path)
}

function onClearTask() {
  if (showAgent.value) {
    router.push({ path: '/app/agent/chat', query: { fresh: '1' } })
    return
  }
  // Agent 被关闭时，退化为其它可用入口
  if (showWizards.value) {
    step.value = 'wizards'
    return
  }
  if (moreCards.value[0]) go(moreCards.value[0].path)
}

function onClearYes() {
  if (showWizards.value) {
    step.value = 'wizards'
    return
  }
  // 向导都关了：有 Agent 则引导过去
  if (showAgent.value) {
    router.push({ path: '/app/agent/chat', query: { fresh: '1' } })
    return
  }
  if (moreCards.value[0]) go(moreCards.value[0].path)
}

function backToAsk() {
  step.value = 'ask'
}

function goSettings() {
  router.push('/app/settings')
}
</script>

<template>
  <section class="home-nav">
    <div class="home-stage">
      <header class="hero">
        <p class="eyebrow">ZERO-CODE AI</p>
        <h2 v-if="step === 'ask'">你明确自己的任务吗？</h2>
        <h2 v-else>选择训练向导</h2>
        <p class="lead">
          {{
            step === 'ask'
              ? `你好${auth.user?.username ? `，${auth.user.username}` : ''}。明确就走训练向导；还不清楚就交给 AI Agent 帮你理清。`
              : '按任务类型进入对应流程，从数据到训练一步步完成。'
          }}
        </p>
      </header>

      <!-- 首问：是否明确任务 -->
      <div v-if="step === 'ask'" class="choice-grid">
        <button
          v-if="showWizards || showAgent || moreCards.length"
          type="button"
          class="nav-card choice"
          :class="{ primary: showWizards }"
          @click="onClearYes"
        >
          <span class="icon-wrap" aria-hidden="true">
            <el-icon :size="24"><VideoCamera /></el-icon>
          </span>
          <span class="card-text">
            <strong>明确，进入训练向导</strong>
            <span>已知检测 / 分割 / 姿态等任务类型，按步骤完成训练</span>
          </span>
        </button>

        <button
          v-if="showAgent || showWizards || moreCards.length"
          type="button"
          class="nav-card choice"
          :class="{ primary: !showWizards && showAgent }"
          @click="onClearTask"
        >
          <span class="icon-wrap" aria-hidden="true">
            <el-icon :size="24"><MagicStick /></el-icon>
          </span>
          <span class="card-text">
            <strong>不明确，去 AI Agent</strong>
            <span>用对话描述现场需求，由助手帮你选型与推进</span>
          </span>
        </button>
      </div>

      <!-- 明确任务 → 选向导 -->
      <div v-else class="wizards-step">
        <button type="button" class="back-btn" @click="backToAsk">
          <el-icon :size="16"><ArrowLeft /></el-icon>
          返回上一步
        </button>
        <div class="wizard-row">
          <button
            v-for="card in wizardCards"
            :key="card.key"
            type="button"
            class="nav-card"
            @click="go(card.path)"
          >
            <span class="icon-wrap" aria-hidden="true">
              <el-icon :size="22"><component :is="card.icon" /></el-icon>
            </span>
            <span class="card-text">
              <strong>{{ card.title }}</strong>
              <span>{{ card.desc }}</span>
            </span>
          </button>
        </div>
        <p v-if="showAgent" class="soft-link-row">
          选错了？
          <button
            type="button"
            class="text-link"
            @click="router.push({ path: '/app/agent/chat', query: { fresh: '1' } })"
          >
            改去 AI Agent
          </button>
        </p>
      </div>

      <!-- 次要入口：不抢主决策 -->
      <div v-if="step === 'ask' && moreCards.length" class="more">
        <p class="more-label">其他功能</p>
        <div class="more-row">
          <button
            v-for="card in moreCards"
            :key="card.key"
            type="button"
            class="more-chip"
            @click="go(card.path)"
          >
            <el-icon :size="15"><component :is="card.icon" /></el-icon>
            {{ card.title }}
          </button>
        </div>
      </div>

      <footer v-if="auth.isAdmin || app.menuVisibility.settings !== false" class="foot">
        <button type="button" class="linkish" @click="goSettings">
          <el-icon><Setting /></el-icon>
          系统设置
        </button>
      </footer>
    </div>
  </section>
</template>

<style scoped>
.home-nav {
  min-height: calc(100vh - var(--header-height, 64px) - 3.5rem);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem 0.75rem 1.5rem;
}

.home-stage {
  width: 100%;
  max-width: 820px;
  margin: 0 auto;
}

.hero {
  margin-bottom: 1.5rem;
  text-align: center;
}

.eyebrow {
  margin: 0 0 0.45rem;
  font-family: var(--font-display);
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  color: var(--brand-soft);
  font-weight: 650;
}

.hero h2 {
  margin: 0 0 0.55rem;
  font-family: var(--font-display);
  font-size: clamp(1.65rem, 2.6vw, 2.1rem);
  font-weight: 650;
  color: var(--brand-deep);
}

.lead {
  margin: 0 auto;
  color: var(--ink-faint);
  font-size: 0.95rem;
  max-width: 32rem;
  line-height: 1.55;
}

.choice-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.9rem;
}

.nav-card {
  display: flex;
  align-items: center;
  gap: 0.95rem;
  text-align: left;
  border: 1px solid var(--line);
  background: var(--surface-elevated);
  border-radius: 14px;
  padding: 1.25rem 1.2rem;
  cursor: pointer;
  transition:
    border-color 0.18s ease,
    background 0.18s ease,
    transform 0.18s ease,
    box-shadow 0.18s ease;
}

.nav-card:hover {
  border-color: rgba(61, 155, 143, 0.45);
  background: #f7fbfa;
  transform: translateY(-1px);
  box-shadow: 0 8px 22px rgba(15, 61, 79, 0.06);
}

.nav-card.primary {
  border-color: rgba(61, 155, 143, 0.35);
  background: linear-gradient(135deg, #eef7f5 0%, #ffffff 55%);
}

.nav-card.primary:hover {
  border-color: var(--brand-soft);
}

.nav-card.choice {
  min-height: 148px;
}

.icon-wrap {
  flex-shrink: 0;
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  background: var(--brand-mist);
  color: var(--brand-deep);
}

.nav-card.primary .icon-wrap {
  background: rgba(61, 155, 143, 0.18);
  color: var(--brand);
}

.card-text {
  display: grid;
  gap: 0.35rem;
  min-width: 0;
  align-content: center;
}

.card-text strong {
  font-size: 1.05rem;
  color: var(--ink);
  line-height: 1.3;
}

.card-text span {
  font-size: 0.84rem;
  color: var(--ink-faint);
  line-height: 1.45;
}

.wizards-step {
  display: grid;
  gap: 0.9rem;
}

.back-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  width: fit-content;
  margin: 0 auto;
  border: 0;
  background: transparent;
  color: var(--ink-faint);
  font-size: 0.86rem;
  cursor: pointer;
  padding: 0.15rem 0;
}

.back-btn:hover {
  color: var(--brand);
}

.wizard-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
}

.soft-link-row {
  margin: 0.15rem 0 0;
  font-size: 0.86rem;
  color: var(--ink-faint);
  text-align: center;
}

.text-link {
  border: 0;
  background: transparent;
  color: var(--brand);
  cursor: pointer;
  font-size: inherit;
  padding: 0;
}

.text-link:hover {
  text-decoration: underline;
}

.more {
  margin-top: 1.75rem;
  padding-top: 1.1rem;
  border-top: 1px solid var(--line);
  text-align: center;
}

.more-label {
  margin: 0 0 0.65rem;
  font-size: 0.8rem;
  color: var(--ink-faint);
}

.more-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.45rem;
}

.more-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  border: 1px solid var(--line);
  background: var(--surface-elevated);
  border-radius: 999px;
  padding: 0.35rem 0.75rem;
  font-size: 0.82rem;
  color: var(--ink);
  cursor: pointer;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.more-chip:hover {
  border-color: rgba(61, 155, 143, 0.4);
  background: #f7fbfa;
}

.foot {
  margin-top: 1.35rem;
  text-align: center;
}

.linkish {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  border: 0;
  background: transparent;
  color: var(--ink-faint);
  font-size: 0.86rem;
  cursor: pointer;
  padding: 0.25rem 0;
}

.linkish:hover {
  color: var(--brand);
}

@media (max-width: 820px) {
  .home-nav {
    align-items: flex-start;
    min-height: auto;
    padding-top: 1.25rem;
  }

  .choice-grid,
  .wizard-row {
    grid-template-columns: 1fr;
  }

  .nav-card.choice {
    min-height: 0;
  }
}
</style>
