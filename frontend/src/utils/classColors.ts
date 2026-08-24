/**
 * 标注类别配色：按 class_id 取色，尽量彼此区分。
 * 选中时仍用本色，靠线宽/描边区分。
 */

const CLASS_PALETTE = [
  '#2a9d8f', // 青绿
  '#e76f51', // 珊瑚
  '#3a86ff', // 亮蓝
  '#f4a261', // 橙
  '#9b5de5', // 紫
  '#00bbf9', // 天蓝
  '#ef476f', // 玫红
  '#06d6a0', // 薄荷绿
  '#118ab2', // 深蓝
  '#ffd166', // 琥珀（深底可读，标签字用深色时另处理）
  '#8ac926', // 黄绿
  '#ff6b6b', // 浅红
  '#4d908e', // 灰青
  '#f72585', // 品红
  '#4361ee', // 靛蓝
  '#fb8500', // 深橙
] as const

/** 按类别 id 返回描边/标签底色 */
export function classAccent(classId: number): string {
  const i = Math.max(0, Math.floor(Number(classId) || 0))
  return CLASS_PALETTE[i % CLASS_PALETTE.length]
}

/** 半透明填充（多边形区域） */
export function classFill(classId: number, alphaHex = '33'): string {
  return classAccent(classId) + alphaHex
}
