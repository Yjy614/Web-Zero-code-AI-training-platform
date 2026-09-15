<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  Aim,
  ArrowDown,
  Box,
  Collection,
  Cpu,
  Crop,
  List,
  MagicStick,
  PictureFilled,
  Setting,
  SwitchButton,
  User,
  Share,
  VideoCamera,
} from '@element-plus/icons-vue'
import { useAuthStore } from '@/stores/auth'
import { useAppStore } from '@/stores/app'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const app = useAppStore()
const collapsed = ref(false)
/** 训练向导折叠组：默认收起，仅用户点击展开 */
const wizardOpen = ref(false)

const active = computed(() => route.path)

type MenuItem = { path: string; title: string; icon: typeof Aim; key?: string }

const wizardChildren = computed((): MenuItem[] => {
  const vis = app.menuVisibility
  return [
    { path: '/app/detect/wizard', title: '目标检测训练', icon: Aim, key: 'detect_wizard' },
    { path: '/app/segment/wizard', title: '实例分割训练', icon: Crop, key: 'segment_wizard' },
    { path: '/app/pose/wizard', title: '姿态估计训练', icon: Share, key: 'pose_wizard' },
  ].filter((item) => vis[item.key as keyof typeof vis] !== false)
})

const showWizardGroup = computed(() => wizardChildren.value.length > 0)

const wizardActive = computed(() =>
  wizardChildren.value.some((item) => active.value.startsWith(item.path)),
)

const flatMenus = computed((): MenuItem[] => {
  const vis = app.menuVisibility
  const featureItems: MenuItem[] = [
    { path: '/app/resources/datasets', title: '数据集管理', icon: Collection, key: 'datasets' },
    { path: '/app/resources/models', title: '模型库', icon: Box, key: 'models' },
    { path: '/app/resources/infer', title: '推理试用', icon: PictureFilled, key: 'infer' },
    { path: '/app/agent/chat', title: 'AI Agent', icon: MagicStick, key: 'agent_chat' },
    { path: '/app/agent/orchestrate', title: '流程编排', icon: List, key: 'agent_orchestrate' },
    { path: '/app/resources/weights', title: '基础模型权重仓库', icon: Cpu, key: 'weights' },
  ].filter((item) => vis[item.key as keyof typeof vis] !== false)

  const items = [...featureItems]
  // 系统设置：普通用户按开关；管理员始终可见，避免误关后无法再改配置
  if (vis.settings !== false || auth.isAdmin) {
    items.push({ path: '/app/settings', title: '系统设置', icon: Setting })
  }
  if (auth.isAdmin) {
    items.push({ path: '/app/admin/users', title: '用户管理', icon: User })
  }
  return items
})

const homePath = computed(() => '/app/home')

const pageTitle = computed(() => (route.meta.title as string) || '工作台')

function toggleWizard() {
  wizardOpen.value = !wizardOpen.value
}

onMounted(async () => {
  try {
    await app.bootstrap()
  } catch {
    // 系统信息失败不阻断布局
  }
})

async function onLogout() {
  await auth.logout()
  router.push('/login')
}
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar" :class="{ collapsed }">
      <div class="brand" @click="router.push(homePath)">
        <div class="logo" aria-hidden="true">
          <span />
        </div>
        <div v-show="!collapsed" class="brand-text">
          <strong>零代码 AI 训练平台</strong>
        </div>
      </div>

      <nav class="nav">
        <!-- 三个训练向导：折叠组 -->
        <div v-if="showWizardGroup" class="nav-group" :class="{ open: wizardOpen }">
          <button
            type="button"
            class="nav-item nav-group-head"
            :class="{ active: wizardActive && !wizardOpen }"
            :title="collapsed ? '训练向导' : undefined"
            @click="toggleWizard"
          >
            <el-icon :size="18"><VideoCamera /></el-icon>
            <span v-show="!collapsed" class="nav-group-title">训练向导</span>
            <el-icon v-show="!collapsed" class="nav-caret" :size="14"><ArrowDown /></el-icon>
          </button>
          <div v-show="wizardOpen" class="nav-group-body">
            <button
              v-for="item in wizardChildren"
              :key="item.path"
              type="button"
              class="nav-item nav-sub"
              :class="{ active: active.startsWith(item.path) }"
              :title="collapsed ? item.title : undefined"
              @click="router.push(item.path)"
            >
              <el-icon :size="17"><component :is="item.icon" /></el-icon>
              <span v-show="!collapsed">{{ item.title }}</span>
            </button>
          </div>
        </div>

        <button
          v-for="item in flatMenus"
          :key="item.path"
          type="button"
          class="nav-item"
          :class="{ active: active.startsWith(item.path) }"
          :title="collapsed ? item.title : undefined"
          @click="router.push(item.path)"
        >
          <el-icon :size="18"><component :is="item.icon" /></el-icon>
          <span v-show="!collapsed">{{ item.title }}</span>
        </button>
      </nav>

      <div class="sidebar-foot">
        <button type="button" class="collapse-btn" @click="collapsed = !collapsed">
          {{ collapsed ? '展开' : '收起侧栏' }}
        </button>
      </div>
    </aside>

    <div class="main">
      <header class="topbar">
        <div>
          <h1>{{ pageTitle }}</h1>
        </div>
        <div class="user-box">
          <div class="user-meta">
            <strong>{{ auth.user?.username }}</strong>
            <span>{{ auth.isAdmin ? '管理员' : '使用人员' }}</span>
          </div>
          <el-button :icon="SwitchButton" @click="onLogout">退出</el-button>
        </div>
      </header>
      <main class="content fade-up">
        <RouterView v-slot="{ Component, route: r }">
          <KeepAlive :include="['DetectWizard', 'SegmentWizard', 'PoseWizard', 'AgentOrchestrate', 'AgentChat']" :max="12">
            <component :is="Component" :key="String(r.name || r.path)" />
          </KeepAlive>
        </RouterView>
      </main>
    </div>
  </div>
</template>

<style scoped>
.app-shell {
  min-height: 100vh;
  display: grid;
  grid-template-columns: auto 1fr;
  background:
    radial-gradient(ellipse 60% 40% at 100% 0%, rgba(61, 155, 143, 0.08), transparent 55%),
    var(--surface);
}

.sidebar {
  width: var(--sidebar-width);
  background: linear-gradient(180deg, #123947 0%, #0f3d4f 55%, #0c3342 100%);
  color: #e7f2f0;
  display: flex;
  flex-direction: column;
  padding: 1.1rem 0.85rem;
  transition: width 0.25s ease;
}

.sidebar.collapsed {
  width: 76px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.35rem 0.55rem 1.1rem;
  cursor: pointer;
}

.logo {
  width: 38px;
  height: 38px;
  border-radius: 10px;
  background: rgba(232, 244, 242, 0.12);
  display: grid;
  place-items: center;
  flex-shrink: 0;
}

.logo span {
  width: 14px;
  height: 14px;
  border: 2px solid #9fd4cb;
  border-radius: 4px 4px 2px 2px;
  transform: rotate(45deg);
}

.brand-text {
  display: block;
  line-height: 1.25;
  min-width: 0;
}

.brand-text strong {
  font-family: var(--font-display);
  font-size: 0.98rem;
  font-weight: 650;
  letter-spacing: 0.02em;
  color: #e7f2f0;
  white-space: nowrap;
}

.nav {
  display: grid;
  gap: 0.35rem;
  flex: 1;
  align-content: start; /* 避免侧栏剩余高度把菜单项撑高 */
  grid-auto-rows: max-content;
}

.nav-group {
  display: grid;
  gap: 0.15rem;
}

.nav-group-body {
  display: grid;
  gap: 0.15rem;
}

.nav-item,
.nav-disabled {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  width: 100%;
  border: 0;
  background: transparent;
  color: inherit;
  text-align: left;
  padding: 0.72rem 0.8rem;
  border-radius: 10px;
  font-size: 0.92rem;
  cursor: pointer;
  transition: background 0.2s ease;
  box-sizing: border-box;
}

.nav-group-title {
  flex: 1;
  min-width: 0;
}

.nav-caret {
  margin-left: auto;
  opacity: 0.7;
  transition: transform 0.2s ease;
  flex-shrink: 0;
}

.nav-group.open .nav-caret {
  transform: rotate(180deg);
}

.nav-sub {
  padding-left: 1.15rem;
  font-size: 0.88rem;
  opacity: 0.92;
}

.sidebar.collapsed .nav-sub {
  padding-left: 0.8rem;
}

.nav-item:hover {
  background: rgba(255, 255, 255, 0.08);
}

.nav-item.active {
  background: rgba(61, 155, 143, 0.28);
  box-shadow: inset 3px 0 0 #3d9b8f;
}

.nav-disabled {
  opacity: 0.42;
  cursor: not-allowed;
}

.nav-disabled em {
  margin-left: auto;
  font-style: normal;
  font-size: 0.72rem;
  opacity: 0.85;
}

.sidebar-foot {
  padding: 0.6rem 0.4rem 0.2rem;
}

.collapse-btn {
  width: 100%;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.04);
  color: rgba(231, 242, 240, 0.78);
  border-radius: 8px;
  padding: 0.45rem;
  cursor: pointer;
  font-size: 0.78rem;
}

.main {
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.topbar {
  min-height: var(--header-height);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.9rem 1.5rem;
  border-bottom: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.72);
  backdrop-filter: blur(8px);
}

.topbar h1 {
  margin: 0;
  font-family: var(--font-display);
  font-size: 1.35rem;
  font-weight: 650;
}

.user-box {
  display: flex;
  align-items: center;
  gap: 0.85rem;
}

.user-meta {
  display: grid;
  text-align: right;
  line-height: 1.25;
}

.user-meta strong {
  font-size: 0.92rem;
}

.user-meta span {
  color: var(--ink-faint);
  font-size: 0.78rem;
}

.content {
  padding: 1.25rem 1.5rem 2rem;
  flex: 1;
}

@media (max-width: 768px) {
  .app-shell {
    grid-template-columns: 1fr;
  }

  .sidebar {
    width: 100%;
    flex-direction: row;
    align-items: center;
    overflow-x: auto;
    padding: 0.65rem;
  }

  .nav {
    display: flex;
    flex: 1;
    align-items: center;
  }

  .nav-group {
    display: flex;
    align-items: center;
    gap: 0.2rem;
    flex-shrink: 0;
  }

  .nav-group-body {
    display: flex;
    padding: 0;
  }

  .nav-sub {
    padding-left: 0.8rem;
  }

  .sidebar-foot,
  .nav-disabled,
  .nav-caret {
    display: none;
  }
}
</style>
