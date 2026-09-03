/** 将助手回复中的 Markdown 转为安全 HTML（聊天气泡用）。 */
import { marked } from 'marked'

marked.setOptions({
  gfm: true,
  breaks: true,
})

/** 极简消毒：去掉脚本/事件处理器等危险片段。 */
function sanitizeHtml(html: string): string {
  return html
    .replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, '')
    .replace(/<iframe[\s\S]*?>[\s\S]*?<\/iframe>/gi, '')
    .replace(/\son\w+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, '')
    .replace(/javascript:/gi, '')
}

export function renderMarkdown(text: string): string {
  const raw = (text || '').trim()
  if (!raw) return ''
  const html = marked.parse(raw, { async: false }) as string
  return sanitizeHtml(html)
}
