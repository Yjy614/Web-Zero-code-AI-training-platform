<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  createUser,
  deleteUser,
  listUsers,
  resetUserPassword,
  updateUser,
  type UserRow,
} from '@/api/users'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const loading = ref(false)
const users = ref<UserRow[]>([])

const createVisible = ref(false)
const createForm = reactive({ username: '', password: '', role: 'user' })

const resetVisible = ref(false)
const resetTarget = ref<UserRow | null>(null)
const resetPassword = ref('')

const roleVisible = ref(false)
const roleTarget = ref<UserRow | null>(null)
const roleValue = ref('user')

async function load() {
  loading.value = true
  try {
    const { data } = await listUsers()
    users.value = data
  } finally {
    loading.value = false
  }
}

function roleLabel(role: string) {
  return role === 'admin' ? '管理员' : '使用人员'
}

async function onCreate() {
  if (!createForm.username.trim() || !createForm.password) {
    ElMessage.warning('请填写用户名和密码')
    return
  }
  await createUser({
    username: createForm.username.trim(),
    password: createForm.password,
    role: createForm.role,
  })
  ElMessage.success('用户已创建')
  createVisible.value = false
  createForm.username = ''
  createForm.password = ''
  createForm.role = 'user'
  await load()
}

function openReset(row: UserRow) {
  resetTarget.value = row
  resetPassword.value = ''
  resetVisible.value = true
}

async function onReset() {
  if (!resetTarget.value || !resetPassword.value) {
    ElMessage.warning('请输入新密码')
    return
  }
  await resetUserPassword(resetTarget.value.id, resetPassword.value)
  ElMessage.success('密码已重置')
  resetVisible.value = false
}

function openRole(row: UserRow) {
  roleTarget.value = row
  roleValue.value = row.role
  roleVisible.value = true
}

async function onRoleSave() {
  if (!roleTarget.value) return
  await updateUser(roleTarget.value.id, { role: roleValue.value })
  ElMessage.success('角色已更新')
  roleVisible.value = false
  await load()
}

async function onDelete(row: UserRow) {
  if (row.id === auth.user?.id) {
    ElMessage.warning('不能删除当前登录账号')
    return
  }
  try {
    await ElMessageBox.confirm(`确认删除用户「${row.username}」？`, '删除确认', { type: 'warning' })
  } catch {
    return
  }
  await deleteUser(row.id)
  ElMessage.success('已删除')
  await load()
}

onMounted(load)
</script>

<template>
  <section class="page" v-loading="loading">
    <header class="head">
      <div>
        <h2>用户管理</h2>
        <p>仅管理员可访问。可创建用户、重置密码、调整角色与删除账号。</p>
      </div>
      <el-button type="primary" @click="createVisible = true">新建用户</el-button>
    </header>

    <el-table :data="users" stripe empty-text="暂无用户">
      <el-table-column prop="id" label="ID" width="70" />
      <el-table-column prop="username" label="用户名" min-width="140" />
      <el-table-column label="角色" width="120">
        <template #default="{ row }">{{ roleLabel(row.role) }}</template>
      </el-table-column>
      <el-table-column
        label="操作"
        width="280"
        align="center"
        header-align="center"
        class-name="ops-col"
        label-class-name="ops-col"
      >
        <template #default="{ row }">
          <div class="ops">
            <el-button link type="primary" @click="openReset(row)">重置密码</el-button>
            <el-button link type="primary" @click="openRole(row)">改角色</el-button>
            <el-button link type="danger" @click="onDelete(row)">删除</el-button>
          </div>
        </template>
      </el-table-column>
    </el-table>

    <el-dialog v-model="createVisible" title="新建用户" width="420px">
      <el-form label-position="top">
        <el-form-item label="用户名">
          <el-input v-model="createForm.username" placeholder="至少 2 个字符" />
        </el-form-item>
        <el-form-item label="初始密码">
          <el-input v-model="createForm.password" type="password" show-password placeholder="至少 4 位" />
        </el-form-item>
        <el-form-item label="角色">
          <el-select v-model="createForm.role" style="width: 100%">
            <el-option label="使用人员" value="user" />
            <el-option label="管理员" value="admin" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createVisible = false">取消</el-button>
        <el-button type="primary" @click="onCreate">创建</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="resetVisible" title="重置密码" width="400px">
      <p class="hint">用户：{{ resetTarget?.username }}</p>
      <el-input v-model="resetPassword" type="password" show-password placeholder="新密码（至少 4 位）" />
      <template #footer>
        <el-button @click="resetVisible = false">取消</el-button>
        <el-button type="primary" @click="onReset">确定</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="roleVisible" title="修改角色" width="400px">
      <p class="hint">用户：{{ roleTarget?.username }}</p>
      <el-select v-model="roleValue" style="width: 100%">
        <el-option label="使用人员" value="user" />
        <el-option label="管理员" value="admin" />
      </el-select>
      <template #footer>
        <el-button @click="roleVisible = false">取消</el-button>
        <el-button type="primary" @click="onRoleSave">保存</el-button>
      </template>
    </el-dialog>
  </section>
</template>

<style scoped>
.head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  margin-bottom: 1rem;
}
.page header h2,
.head h2 {
  margin: 0;
  font-family: var(--font-display);
  font-size: 1.35rem;
}
.head p {
  margin: 0.4rem 0 0;
  color: var(--ink-muted);
}
.hint {
  margin: 0 0 0.75rem;
  color: var(--ink-muted);
}
</style>
