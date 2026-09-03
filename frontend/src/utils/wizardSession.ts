/** 训练向导会话级 UI 状态（登出后应清空）。 */

export function wizardStateKey(taskType: string) {
  return `${taskType || 'detect'}-wizard-ui-state`
}

export const DETECT_WIZARD_STATE_KEY = wizardStateKey('detect')

export function trainConfigDraftKey(taskType: string, datasetId: number) {
  return `train-config-draft:${taskType || 'detect'}:${datasetId}`
}

export function clearWizardState(taskType: string) {
  try {
    sessionStorage.removeItem(wizardStateKey(taskType))
  } catch {
    // 忽略存储失败
  }
}

/** 清除所有训练配置草稿 */
export function clearTrainConfigDrafts() {
  try {
    const keys: string[] = []
    for (let i = 0; i < sessionStorage.length; i++) {
      const k = sessionStorage.key(i)
      if (k && k.startsWith('train-config-draft:')) keys.push(k)
    }
    for (const k of keys) sessionStorage.removeItem(k)
  } catch {
    // 忽略存储失败
  }
}

export function clearDetectWizardState() {
  clearWizardState('detect')
  clearWizardState('segment')
  clearWizardState('pose')
  clearTrainConfigDrafts()
  clearAgentOrchestrateState()
}

const AGENT_ORCHESTRATE_KEY = 'agent-orchestrate-ui-state'

export function agentOrchestrateStateKey() {
  return AGENT_ORCHESTRATE_KEY
}

export function clearAgentOrchestrateState() {
  try {
    sessionStorage.removeItem(AGENT_ORCHESTRATE_KEY)
  } catch {
    // 忽略存储失败
  }
}
