import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useAppStore } from '@/stores/app'
import { menuKeyFromPath } from '@/api/system'
import { clearDetectWizardState } from '@/utils/wizardSession'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/login',
      name: 'login',
      component: () => import('@/views/LoginView.vue'),
      meta: { public: true },
    },
    {
      path: '/',
      redirect: '/app',
    },
    {
      path: '/app',
      component: () => import('@/layouts/AppLayout.vue'),
      children: [
        {
          path: '',
          redirect: '/app/home',
        },
        {
          path: 'home',
          name: 'home',
          component: () => import('@/views/HomeNavView.vue'),
          meta: { title: '开始' },
        },
        {
          path: 'detect/wizard',
          name: 'detect-wizard',
          component: () => import('@/views/detect/WizardView.vue'),
          meta: { title: '目标检测训练', taskType: 'detect', menuKey: 'detect_wizard' },
        },
        {
          path: 'segment/wizard',
          name: 'segment-wizard',
          component: () => import('@/views/segment/WizardView.vue'),
          meta: { title: '实例分割训练', taskType: 'segment', menuKey: 'segment_wizard' },
        },
        {
          path: 'pose/wizard',
          name: 'pose-wizard',
          component: () => import('@/views/pose/WizardView.vue'),
          meta: { title: '姿态估计训练', taskType: 'pose', menuKey: 'pose_wizard' },
        },
        {
          path: 'resources/datasets',
          name: 'datasets',
          component: () => import('@/views/resources/DatasetsView.vue'),
          meta: { title: '数据集管理', menuKey: 'datasets' },
        },
        {
          path: 'resources/models',
          name: 'models',
          component: () => import('@/views/resources/ModelsView.vue'),
          meta: { title: '模型库', menuKey: 'models' },
        },
        {
          path: 'resources/active-learn',
          name: 'active-learn',
          component: () => import('@/views/resources/ActiveLearnView.vue'),
          meta: { title: '主动学习', menuKey: 'models' },
        },
        {
          path: 'resources/infer',
          name: 'infer',
          component: () => import('@/views/resources/InferView.vue'),
          meta: { title: '推理试用', menuKey: 'infer' },
        },
        {
          path: 'resources/weights',
          name: 'weights',
          component: () => import('@/views/resources/WeightsView.vue'),
          meta: { title: '基础模型权重仓库', menuKey: 'weights' },
        },
        {
          path: 'agent/chat',
          name: 'agent-chat',
          component: () => import('@/views/agent/AgentChatView.vue'),
          meta: { title: 'AI Agent', menuKey: 'agent_chat' },
        },
        {
          path: 'agent/orchestrate',
          name: 'agent-orchestrate',
          component: () => import('@/views/agent/AgentOrchestrateView.vue'),
          meta: { title: 'AI 流程编排', menuKey: 'agent_orchestrate' },
        },
        {
          path: 'settings',
          name: 'settings',
          component: () => import('@/views/SettingsView.vue'),
          meta: { title: '系统设置', menuKey: 'settings' },
        },
        {
          path: 'admin/users',
          name: 'admin-users',
          component: () => import('@/views/admin/UsersView.vue'),
          meta: { title: '用户管理', admin: true },
        },
      ],
    },
    {
      path: '/:pathMatch(.*)*',
      redirect: '/app/home',
    },
  ],
})

router.beforeEach(async (to) => {
  const auth = useAuthStore()
  if (to.meta.public) {
    if (auth.isLoggedIn && to.path === '/login') {
      const app = useAppStore()
      if (!Object.values(app.menuVisibility).some(Boolean)) {
        await app.loadMenuVisibility()
      }
      return '/app/home'
    }
    if (!auth.isLoggedIn) clearDetectWizardState()
    return true
  }
  if (!auth.isLoggedIn) {
    clearDetectWizardState()
    return '/login'
  }

  const app = useAppStore()
  // 首次进入时确保已拉取菜单可见性
  if (!app.systemInfo) {
    try {
      await app.bootstrap()
    } catch {
      /* ignore */
    }
  }

  if (to.meta.admin && !auth.isAdmin) {
    return '/app/home'
  }

  const menuKey =
    (to.meta.menuKey as string | undefined) || menuKeyFromPath(to.path) || null
  if (menuKey && app.menuVisibility[menuKey as keyof typeof app.menuVisibility] === false) {
    // 系统设置关闭时仅拦普通用户，管理员仍可进入以免无法改回
    if (menuKey === 'settings' && auth.isAdmin) {
      return true
    }
    return '/app/home'
  }
  return true
})

export default router
