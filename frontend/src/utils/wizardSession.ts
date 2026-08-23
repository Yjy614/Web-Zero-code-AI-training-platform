/** 检测训练向导的会话级 UI 状态（登出后应清空）。 */

export const DETECT_WIZARD_STATE_KEY = 'detect-wizard-ui-state'

export function clearDetectWizardState() {
  try {
    sessionStorage.removeItem(DETECT_WIZARD_STATE_KEY)
  } catch {
    // 忽略存储失败
  }
}
