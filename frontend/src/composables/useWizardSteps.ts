/** 向导步骤定义：按任务类型扩展，检测/分割共用壳。 */
export type WizardTaskType = 'detect' | 'segment'

export interface WizardStepMeta {
  title: string
  desc: string
}

export function getWizardSteps(taskType: WizardTaskType = 'detect'): WizardStepMeta[] {
  const annotateDesc = taskType === 'segment' ? '多边形与类别' : '矩形框与类别'
  return [
    { title: '导入', desc: '创建任务与上传图片' },
    { title: '清洗', desc: '去重与归一化' },
    { title: '标注', desc: annotateDesc },
    { title: '配置', desc: '划分与超参数' },
    { title: '训练', desc: '进度与曲线' },
    { title: '评估', desc: '指标与报告' },
    { title: '导出', desc: 'PT / ONNX 下载' },
  ]
}
