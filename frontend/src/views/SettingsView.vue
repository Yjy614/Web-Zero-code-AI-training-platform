<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import {
  DEFAULT_MENU_VISIBILITY,
  fetchSettings,
  fetchSystemInfo,
  MENU_LABEL_BY_KEY,
  updateSettings,
  type AppSettings,
  type MenuVisibility,
  type ModelEndpointConfig,
  type SystemInfo,
} from '@/api/system'
import { useAuthStore } from '@/stores/auth'
import { useAppStore } from '@/stores/app'

const auth = useAuthStore()
const app = useAppStore()
const loading = ref(false)
const saving = ref(false)
const savingMenu = ref(false)
const info = ref<SystemInfo | null>(null)

const emptyEndpoint = (): ModelEndpointConfig => ({
  base_url: '',
  api_key: '',
  model: '',
  timeout: 60,
  api_key_set: false,
  thinking_enabled: true,
  reasoning_effort: 'high',
})

const settings = reactive<AppSettings>({
  demo_mode: true,
  job_runner: 'mock',
  forbid_weight_download: true,
  free_step_nav: true,
  menu_visibility: { ...DEFAULT_MENU_VISIBILITY },
  llm: emptyEndpoint(),
  vision: emptyEndpoint(),
})

const menuKeys = Object.keys(MENU_LABEL_BY_KEY) as Array<keyof MenuVisibility>

const canEdit = computed(() => auth.isAdmin)

onMounted(async () => {
  loading.value = true
  try {
    const [infoRes, settingsRes] = await Promise.all([fetchSystemInfo(), fetchSettings()])
    info.value = infoRes.data
    Object.assign(settings, settingsRes.data)
    settings.llm = { ...emptyEndpoint(), ...(settingsRes.data.llm || {}) }
    settings.vision = { ...emptyEndpoint(), ...(settingsRes.data.vision || {}) }
    settings.menu_visibility = {
      ...DEFAULT_MENU_VISIBILITY,
      ...(settingsRes.data.menu_visibility || {}),
    }
    app.demoMode = settings.demo_mode
    app.setMenuVisibility(settings.menu_visibility)
  } finally {
    loading.value = false
  }
})

async function onToggleDemo(val: string | number | boolean) {
  if (!canEdit.value) return
  const enabled = Boolean(val)
  try {
    const { data } = await updateSettings({ demo_mode: enabled })
    Object.assign(settings, data)
    settings.llm = { ...emptyEndpoint(), ...(data.llm || {}) }
    settings.vision = { ...emptyEndpoint(), ...(data.vision || {}) }
    settings.menu_visibility = {
      ...DEFAULT_MENU_VISIBILITY,
      ...(data.menu_visibility || {}),
    }
    app.demoMode = data.demo_mode
    ElMessage.success(
      enabled
        ? '已开启演示模式（Mock 训练）'
        : '已关闭演示模式：将使用本机 Ultralytics 真实训练（需已安装依赖并上传权重）',
    )
  } catch {
    settings.demo_mode = !enabled
  }
}

async function saveMenuVisibility() {
  if (!canEdit.value) return
  savingMenu.value = true
  try {
    const { data } = await updateSettings({ menu_visibility: { ...settings.menu_visibility } })
    settings.menu_visibility = {
      ...DEFAULT_MENU_VISIBILITY,
      ...(data.menu_visibility || {}),
    }
    app.setMenuVisibility(settings.menu_visibility)
    ElMessage.success('功能入口显示设置已保存，所有用户立即生效')
  } finally {
    savingMenu.value = false
  }
}

async function saveEndpoint(kind: 'llm' | 'vision') {
  if (!canEdit.value) return
  saving.value = true
  try {
    const payload =
      kind === 'llm'
        ? {
            llm: {
              base_url: settings.llm.base_url,
              api_key: settings.llm.api_key,
              model: settings.llm.model,
              timeout: settings.llm.timeout,
              thinking_enabled: Boolean(settings.llm.thinking_enabled),
              reasoning_effort: settings.llm.reasoning_effort || 'high',
            },
          }
        : {
            vision: {
              base_url: settings.vision.base_url,
              api_key: settings.vision.api_key,
              model: settings.vision.model,
              timeout: settings.vision.timeout,
            },
          }
    const { data } = await updateSettings(payload)
    Object.assign(settings, data)
    settings.llm = { ...emptyEndpoint(), ...(data.llm || {}) }
    settings.vision = { ...emptyEndpoint(), ...(data.vision || {}) }
    settings.menu_visibility = {
      ...DEFAULT_MENU_VISIBILITY,
      ...(data.menu_visibility || {}),
    }
    ElMessage.success(kind === 'llm' ? '大模型配置已保存' : '视觉模型配置已保存')
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <section class="page" v-loading="loading">
    <header>
      <h2>系统设置</h2>
      <p>
        {{
          canEdit
            ? '环境信息、演示模式与大模型配置。管理员可修改系统级设置。'
            : '环境信息与大模型配置。使用人员仅可查看，不可修改系统级配置。'
        }}
      </p>
    </header>

    <div class="grid">
      <div class="block" :class="{ wide: !canEdit }">
        <h3>本地环境</h3>
        <dl :class="{ 'env-row': !canEdit }">
          <div><dt>API 版本</dt><dd>{{ info?.api_version || '-' }}</dd></div>
          <div><dt>存储</dt><dd>{{ info?.storage_configured ? '已配置（内网共享存储）' : '未配置' }}</dd></div>
          <div v-if="canEdit">
            <dt>任务执行器</dt>
            <dd>
              {{
                settings.demo_mode
                  ? 'mock（演示）'
                  : settings.job_runner === 'cluster'
                    ? 'cluster'
                    : 'local（本机）'
              }}
            </dd>
          </div>
          <div><dt>禁止权重下载</dt><dd>{{ settings.forbid_weight_download ? '是' : '否' }}</dd></div>
        </dl>
      </div>

      <div v-if="canEdit" class="block">
        <h3>演示模式</h3>
        <p class="hint">
          开启：Mock 假训练，适合演示。关闭：本机 Ultralytics 真实训练（需 pip install ultralytics，并在权重仓库上传
          .pt）。
        </p>
        <el-switch
          v-model="settings.demo_mode"
          active-text="开启"
          inactive-text="关闭"
          @change="onToggleDemo"
        />
      </div>

      <div class="block wide">
        <h3>功能入口显示</h3>
        <p class="hint">
          由管理员控制侧栏各功能是否对<strong>所有用户</strong>显示。关闭后，用户侧栏不再出现该入口，直接访问对应地址也会被拦截。关闭「系统设置」仅对普通用户生效，管理员仍可进入本页；用户管理仅管理员可见。
        </p>
        <div class="menu-vis-grid">
          <label v-for="key in menuKeys" :key="key" class="menu-vis-item">
            <span>{{ MENU_LABEL_BY_KEY[key] }}</span>
            <el-switch v-model="settings.menu_visibility[key]" :disabled="!canEdit" />
          </label>
        </div>
        <el-button
          v-if="canEdit"
          type="primary"
          :loading="savingMenu"
          style="margin-top: 0.75rem"
          @click="saveMenuVisibility"
        >
          保存功能入口设置
        </el-button>
      </div>

      <div class="block wide">
        <h3>大模型（LLM）</h3>
        <p class="hint">
          对话式 Agent 使用此配置。思考模式对应 DeepSeek 官方
          <code>thinking</code> /
          <code>reasoning_effort</code>
          （见
          <a
            href="https://api-docs.deepseek.com/zh-cn/guides/thinking_mode"
            target="_blank"
            rel="noopener noreferrer"
            >思考模式文档</a
          >）。开启后会返回 <code>reasoning_content</code> 并在对话中展示。Base URL 一般为
          <code>https://api.deepseek.com</code>。
        </p>
        <div class="form-grid">
          <label>Base URL</label>
          <el-input
            v-model="settings.llm.base_url"
            :disabled="!canEdit"
            placeholder="https://api.deepseek.com"
          />
          <label>API Key</label>
          <el-input
            v-model="settings.llm.api_key"
            :disabled="!canEdit"
            type="password"
            show-password
            :placeholder="settings.llm.api_key_set ? '已配置（留空掩码则不修改）' : '可选'"
          />
          <label>Model</label>
          <el-input
            v-model="settings.llm.model"
            :disabled="!canEdit"
            placeholder="如 deepseek-chat / deepseek-v4-pro"
          />
          <label>Timeout(秒)</label>
          <el-input-number v-model="settings.llm.timeout" :disabled="!canEdit" :min="5" :max="600" />
          <label>思考模式</label>
          <el-switch
            v-model="settings.llm.thinking_enabled"
            :disabled="!canEdit"
            active-text="开启"
            inactive-text="关闭"
          />
          <label>思考强度</label>
          <el-select
            v-model="settings.llm.reasoning_effort"
            :disabled="!canEdit || !settings.llm.thinking_enabled"
            style="width: 100%"
          >
            <el-option label="low（更快）" value="low" />
            <el-option label="high（默认）" value="high" />
            <el-option label="max（更强）" value="max" />
          </el-select>
        </div>
        <el-button v-if="canEdit" type="primary" :loading="saving" @click="saveEndpoint('llm')">
          保存大模型配置
        </el-button>
      </div>

      <div class="block wide">
        <h3>视觉模型（Vision）</h3>
        <p class="hint">用于后续 VLM 预标注等能力，一期可先配置占位。</p>
        <div class="form-grid">
          <label>Base URL</label>
          <el-input v-model="settings.vision.base_url" :disabled="!canEdit" placeholder="https://..." />
          <label>API Key</label>
          <el-input
            v-model="settings.vision.api_key"
            :disabled="!canEdit"
            type="password"
            show-password
            :placeholder="settings.vision.api_key_set ? '已配置（留空掩码则不修改）' : '可选'"
          />
          <label>Model</label>
          <el-input v-model="settings.vision.model" :disabled="!canEdit" placeholder="模型名" />
          <label>Timeout(秒)</label>
          <el-input-number v-model="settings.vision.timeout" :disabled="!canEdit" :min="5" :max="600" />
        </div>
        <el-button v-if="canEdit" type="primary" :loading="saving" @click="saveEndpoint('vision')">
          保存视觉模型配置
        </el-button>
      </div>
    </div>
  </section>
</template>

<style scoped>
.page header h2 {
  margin: 0;
  font-family: var(--font-display);
  font-size: 1.35rem;
}
.page header p {
  margin: 0.4rem 0 1.1rem;
  color: var(--ink-muted);
}
.grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
}
.block {
  background: var(--surface-elevated);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  padding: 1.25rem 1.35rem;
}
.block.wide {
  grid-column: 1 / -1;
}
.block h3 {
  margin: 0 0 0.55rem;
  font-size: 1.05rem;
}
.hint,
.readonly {
  color: var(--ink-muted);
  font-size: 0.88rem;
  line-height: 1.6;
}
.readonly {
  margin-top: 0.85rem;
}
.form-grid {
  display: grid;
  grid-template-columns: 110px 1fr;
  gap: 0.7rem 0.85rem;
  align-items: center;
  margin: 0.9rem 0 1rem;
  max-width: none;
  width: 100%;
}
dl {
  margin: 0;
  display: grid;
  gap: 0.75rem;
}
dl.env-row {
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.85rem;
}
dl.env-row > div {
  grid-template-columns: 1fr;
  gap: 0.35rem;
  padding: 0.85rem 1rem;
  background: #f7fafb;
  border: 1px solid var(--line);
  border-radius: 10px;
}
dl.env-row dt {
  font-size: 0.78rem;
}
dl.env-row dd {
  font-family: var(--font-display);
  font-size: 1.05rem;
  color: var(--brand-deep);
  font-weight: 600;
}
dl > div {
  display: grid;
  grid-template-columns: 120px 1fr;
  gap: 0.5rem;
  font-size: 0.9rem;
}
.menu-vis-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 0.65rem 1rem;
  margin-top: 0.85rem;
}
.menu-vis-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.55rem 0.75rem;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #f7fafb;
  font-size: 0.9rem;
  color: var(--ink);
}
dt {
  color: var(--ink-faint);
}
dd {
  margin: 0;
  word-break: break-all;
}
@media (max-width: 900px) {
  .grid {
    grid-template-columns: 1fr;
  }
  .form-grid {
    grid-template-columns: 1fr;
  }
  dl.env-row {
    grid-template-columns: 1fr;
  }
}
</style>
