const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export type Role = 'user' | 'assistant'

export type AnswerLength = 'short' | 'medium' | 'long'

export interface ChatMessage {
  role: Role
  content: string
  sources: string[]
  refused: boolean
}

interface AskResponse {
  answer: string
  sources: string[]
  refused: boolean
}

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    throw new ApiError('Could not reach the assistant. Is the backend running?')
  }

  if (!response.ok) {
    throw new ApiError(await describeFailure(response))
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T)
}

async function describeFailure(response: Response): Promise<string> {
  // The server explains its own failures precisely - an exhausted model quota
  // reads very differently from a cold start - so prefer its message and fall
  // back to a status-based one only when there is nothing to show.
  try {
    const body = await response.json()
    if (typeof body?.detail === 'string' && body.detail.length > 0) return body.detail
  } catch {
    // No JSON body; use the generic messages below.
  }

  if (response.status === 404) return 'This conversation expired. Starting a new one.'
  if (response.status === 429) return 'Too many requests. Please wait a moment.'
  if (response.status === 503) return 'The assistant is still starting up. Try again shortly.'
  return `Request failed (${response.status}).`
}

export const createSession = () =>
  // An explicit body keeps a Content-Length header on the request. Google's
  // load balancer rejects bodyless POSTs with 411, where a local dev server
  // accepts them.
  request<{ session_id: string }>('/sessions', { method: 'POST', body: '{}' }).then(
    (r) => r.session_id,
  )

export const fetchMessages = (sessionId: string) =>
  request<{ messages: ChatMessage[] }>(`/sessions/${sessionId}/messages`).then((r) => r.messages)

export const sendMessage = (sessionId: string, message: string, length: AnswerLength) =>
  request<AskResponse>(`/sessions/${sessionId}/messages`, {
    method: 'POST',
    body: JSON.stringify({ message, length }),
  })

export const deleteSession = (sessionId: string) =>
  request<void>(`/sessions/${sessionId}`, { method: 'DELETE' })
