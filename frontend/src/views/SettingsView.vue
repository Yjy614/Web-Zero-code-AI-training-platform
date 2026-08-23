<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import {
  fetchSettings,
  fetchSystemInfo,
  updateSettings,
  type AppSettings,
  type ModelEndpointConfig,
  type SystemInfo,
} from '@/api/system'
import { useAuthStore } from '@/stores/auth'
import { useAppStore } from '@/stores/app'

const auth = useAuthStore()
const app = useAppStore()
const loading = ref(false)
const saving = ref(false)
const info = ref<SystemInfo | null>(null)

const emptyEndpoint = (): ModelEndpointConfig => ({
  base_url: '',
  api_key: '',
  model: '',
  timeout: 60,
  api_key_set: false,
})

const settings = reactive<AppSettings>({
  demo_mode: true,
  job_runner: 'mock',
  forbid_weight_download: true,
  free_step_nav: true,
  llm: emptyEndpoint(),
  vision: emptyEndpoint(),
})

const canEdit = computed(() => auth.isAdmin)

onMounted(async () => {
  loading.value = true
  try {
    const [infoRes, settingsRes] = await Promise.all([fetchSystemInfo(), fetchSettings()])
    info.value = infoRes.data
    Object.assign(settings, settingsRes.data)
    settings.llm = { ...emptyEndpoint(), ...(settingsRes.data.llm || {}) }
    settings.vision = { ...emptyEndpoint(), ...(settingsRes.data.vision || {}) }
    app.demoMode = settings.demo_mode
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
        <h3>大模型（LLM）</h3>
        <p class="hint">演示期可不真实调用，配置将保存在服务器供后续预标注/建议使用。</p>
        <div class="form-grid">
          <label>Base URL</label>
          <el-input v-model="settings.llm.base_url" :disabled="!canEdit" placeholder="https://..." />
          <label>API Key</label>
          <el-input
            v-model="settings.llm.api_key"
            :disabled="!canEdit"
            type="password"
            show-password
            :placeholder="settings.llm.api_key_set ? '已配置（留空掩码则不修改）' : '可选'"
          />
          <label>Model</label>
          <el-input v-model="settings.llm.model" :disabled="!canEdit" placeholder="模型名" />
          <label>Timeout(秒)</label>
          <el-input-number v-model="settings.llm.timeout" :disabled="!canEdit" :min="5" :max="600" />
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
