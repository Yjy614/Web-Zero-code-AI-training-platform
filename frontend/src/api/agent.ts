/** 对话式 AI Agent API。 */
import http from './http'
import { getAuthToken } from '@/utils/authStorage'

export type AgentChatRole = 'user' | 'assistant' | 'tool' | 'system'

export interface AgentChatMessage {
  id: string
  role: AgentChatRole
  content?: string | null
  tool_call_id?: string | null
  name?: string | null
  tool_calls?: Record<string, unknown>[]
  ui?: Record<string, unknown>
  created_at?: string
}

export interface AgentChatSession {
  id: string
  owner_id: number
  title: string
  status: 'idle' | 'running' | 'waiting_user' | 'waiting_job'
  messages: AgentChatMessage[]
  context: Record<string, unknown>
  pending_ask?: {
    question?: string
    options?: string[]
  } | null
  pending_job?: {
    job_id?: number
    job_type?: string
    task_id?: number
  } | null
  created_at?: string
  updated_at?: string
}

export interface AgentChatSessionSummary {
  id: string
  title: string
  updated_at?: string
  message_count: number
  status: string
}

export interface AgentChatTurn {
  session: AgentChatSession
  assistant_text: string
  tool_events: Array<{
    tool?: string
    label?: string
    ok?: boolean
    summary?: string
  }>
}

export function listAgentChatSessions() {
  return http.get<AgentChatSessionSummary[]>('/agent/chat/sessions')
}

export function createAgentChatSession(title?: string) {
  return http.post<AgentChatSession>('/agent/chat/sessions', title ? { title } : {})
}

export function getAgentChatSession(sessionId: string) {
  return http.get<AgentChatSession>(`/agent/chat/sessions/${sessionId}`)
}

export function renameAgentChatSession(sessionId: string, title: string) {
  return http.patch<AgentChatSession>(`/agent/chat/sessions/${sessionId}`, { title })
}

export function deleteAgentChatSession(sessionId: string) {
  return http.delete<{ ok: boolean }>(`/agent/chat/sessions/${sessionId}`)
}

export function sendAgentChatMessage(sessionId: string, message: string) {
  return http.post<AgentChatTurn>(
    `/agent/chat/sessions/${sessionId}/messages`,
    { message },
    { timeout: 180000 },
  )
}

export type AgentChatStreamEvent =
  | { type: 'status'; phase?: string; text?: string; tool?: string; label?: string; tools?: string[] }
  | { type: 'thinking'; text?: string }
  | { type: 'thinking_delta'; delta?: string }
  | { type: 'answer_delta'; delta?: string }
  | { type: 'session'; session: AgentChatSession }
  | { type: 'cancelled'; session?: AgentChatSession }
  | { type: 'done'; turn: AgentChatTurn }
  | { type: 'error'; message?: string; code?: string; session?: AgentChatSession }

/** 读取 SSE 流并回调事件。 */
async function readAgentSseStream(
  res: Response,
  onEvent: (ev: AgentChatStreamEvent) => void,
  signal?: AbortSignal,
): Promise<AgentChatTurn | null> {
  if (!res.body) throw new Error('浏览器不支持流式响应')

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let lastTurn: AgentChatTurn | null = null
  let streamError: string | null = null

  try {
    while (true) {
      if (signal?.aborted) {
        await reader.cancel()
        break
      }
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const chunks = buffer.split('\n\n')
      buffer = chunks.pop() || ''
      for (const chunk of chunks) {
        const lines = chunk.split('\n')
        const dataLine = lines.find((l) => l.startsWith('data:'))
        if (!dataLine) continue
        const raw = dataLine.slice(5).trim()
        if (!raw) continue
        try {
          const ev = JSON.parse(raw) as AgentChatStreamEvent
          onEvent(ev)
          if (ev.type === 'done' && ev.turn) lastTurn = ev.turn
          if (ev.type === 'error' && ev.message) streamError = ev.message
        } catch {
          /* ignore bad frame */
        }
      }
    }
  } catch (e) {
    if (signal?.aborted || (e instanceof DOMException && e.name === 'AbortError')) {
      return lastTurn
    }
    throw e
  }

  if (streamError && !lastTurn) {
    const err = new Error(streamError)
    ;(err as Error & { alreadyNotified?: boolean }).alreadyNotified = true
    throw err
  }
  return lastTurn
}

async function parseFetchError(res: Response): Promise<string> {
  let msg = `请求失败（${res.status}）`
  try {
    const data = await res.json()
    const detail = data?.detail
    if (typeof detail === 'string') msg = detail
    else if (detail?.message) msg = String(detail.message)
    else if (data?.message) msg = String(data.message)
  } catch {
    /* ignore */
  }
  return msg
}

/** 流式发送：逐步收到思考 / 工具 / 会话快照。 */
export async function streamAgentChatMessage(
  sessionId: string,
  message: string,
  onEvent: (ev: AgentChatStreamEvent) => void,
  signal?: AbortSignal,
): Promise<AgentChatTurn | null> {
  const token = getAuthToken()
  const res = await fetch(`/api/v1/agent/chat/sessions/${sessionId}/messages/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ message }),
    signal,
  })
  if (!res.ok) throw new Error(await parseFetchError(res))
  return readAgentSseStream(res, onEvent, signal)
}

/** Job 结束后流式续跑。 */
export async function streamContinueAgentChatJob(
  sessionId: string,
  onEvent: (ev: AgentChatStreamEvent) => void,
  signal?: AbortSignal,
): Promise<AgentChatTurn | null> {
  const token = getAuthToken()
  const res = await fetch(`/api/v1/agent/chat/sessions/${sessionId}/continue-job/stream`, {
    method: 'POST',
    headers: {
      Accept: 'text/event-stream',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    signal,
  })
  if (!res.ok) throw new Error(await parseFetchError(res))
  return readAgentSseStream(res, onEvent, signal)
}

export function cancelAgentChatTurn(sessionId: string) {
  return http.post<{ ok: boolean; message: string }>(`/agent/chat/sessions/${sessionId}/cancel`)
}

export function continueAgentChatJob(sessionId: string) {
  return http.post<AgentChatTurn>(`/agent/chat/sessions/${sessionId}/continue-job`, {}, { timeout: 180000 })
}
