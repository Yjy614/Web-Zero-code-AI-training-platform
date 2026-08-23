<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  Aim,
  Box,
  Collection,
  Cpu,
  Setting,
  SwitchButton,
  User,
} from '@element-plus/icons-vue'
import DemoBanner from '@/components/DemoBanner.vue'
import { useAuthStore } from '@/stores/auth'
import { useAppStore } from '@/stores/app'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const app = useAppStore()
const collapsed = ref(false)

const active = computed(() => route.path)

const menus = computed(() => {
  const items = [
    { path: '/app/detect/wizard', title: '目标检测训练', icon: Aim },
    { path: '/app/resources/datasets', title: '数据集管理', icon: Collection },
    { path: '/app/resources/models', title: '模型库', icon: Box },
    { path: '/app/resources/weights', title: '基础模型权重仓库', icon: Cpu },
    { path: '/app/settings', title: '系统设置', icon: Setting },
  ]
  if (auth.isAdmin) {
    items.push({ path: '/app/admin/users', title: '用户管理', icon: User })
  }
  return items
})

const pageTitle = computed(() => (route.meta.title as string) || '工作台')

onMounted(async () => {
  try {
    await app.loadSystemInfo()
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
      <div class="brand" @click="router.push('/app/detect/wizard')">
        <div class="logo" aria-hidden="true">
          <span />
        </div>
        <div v-show="!collapsed" class="brand-text">
          <strong>零代码 AI 训练平台</strong>
        </div>
      </div>

      <nav class="nav">
        <button
          v-for="item in menus"
          :key="item.path"
          type="button"
          class="nav-item"
          :class="{ active: active.startsWith(item.path) }"
          @click="router.push(item.path)"
        >
          <el-icon :size="18"><component :is="item.icon" /></el-icon>
          <span v-show="!collapsed">{{ item.title }}</span>
        </button>

        <div class="nav-disabled" title="即将推出">
          <el-icon :size="18"><Aim /></el-icon>
          <span v-show="!collapsed">实例分割训练</span>
          <em v-show="!collapsed">即将推出</em>
        </div>
      </nav>

      <div class="sidebar-foot">
        <button type="button" class="collapse-btn" @click="collapsed = !collapsed">
          {{ collapsed ? '展开' : '收起侧栏' }}
        </button>
      </div>
    </aside>

    <div class="main">
      <DemoBanner :visible="app.demoMode && auth.isAdmin" />
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
          <KeepAlive :include="['DetectWizard']">
            <component :is="Component" :key="r.name as string" />
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
  }

  .sidebar-foot,
  .nav-disabled {
    display: none;
  }
}
</style>
