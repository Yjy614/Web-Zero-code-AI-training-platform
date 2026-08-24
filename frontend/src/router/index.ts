import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
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
      redirect: '/app/detect/wizard',
    },
    {
      path: '/app',
      component: () => import('@/layouts/AppLayout.vue'),
      children: [
        {
          path: 'detect/wizard',
          name: 'detect-wizard',
          component: () => import('@/views/detect/WizardView.vue'),
          meta: { title: '目标检测训练', taskType: 'detect' },
        },
        {
          path: 'segment/wizard',
          name: 'segment-wizard',
          component: () => import('@/views/segment/WizardView.vue'),
          meta: { title: '实例分割训练', taskType: 'segment' },
        },
        {
          path: 'resources/datasets',
          name: 'datasets',
          component: () => import('@/views/resources/DatasetsView.vue'),
          meta: { title: '数据集管理' },
        },
        {
          path: 'resources/models',
          name: 'models',
          component: () => import('@/views/resources/ModelsView.vue'),
          meta: { title: '模型库' },
        },
        {
          path: 'resources/infer',
          name: 'infer',
          component: () => import('@/views/resources/InferView.vue'),
          meta: { title: '推理试用' },
        },
        {
          path: 'resources/weights',
          name: 'weights',
          component: () => import('@/views/resources/WeightsView.vue'),
          meta: { title: '基础模型权重仓库' },
        },
        {
          path: 'settings',
          name: 'settings',
          component: () => import('@/views/SettingsView.vue'),
          meta: { title: '系统设置' },
        },
        {
          path: 'admin/users',
          name: 'admin-users',
          component: () => import('@/views/admin/UsersView.vue'),
          meta: { title: '用户管理', admin: true },
        },
      ],
    },
    { path: '/:pathMatch(.*)*', redirect: '/app/detect/wizard' },
  ],
})

router.beforeEach((to) => {
  const auth = useAuthStore()
  if (to.meta.public) {
    if (auth.isLoggedIn && to.path === '/login') return '/app/detect/wizard'
    // 未登录访问登录页时清掉向导缓存，避免误恢复
    if (!auth.isLoggedIn) clearDetectWizardState()
    return true
  }
  if (!auth.isLoggedIn) {
    clearDetectWizardState()
    return '/login'
  }
  if (to.meta.admin && !auth.isAdmin) return '/app/detect/wizard'
  return true
})

export default router
