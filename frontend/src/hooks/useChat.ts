import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ApiError,
  createSession,
  deleteSession,
  fetchMessages,
  sendMessage,
  type AnswerLength,
  type ChatMessage,
} from '../api/client'

const SESSION_KEY = 'consciousness-assistant.session-id'

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isThinking, setIsThinking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const sessionId = useRef<string | null>(null)

  const startSession = useCallback(async () => {
    const id = await createSession()
    sessionId.current = id
    localStorage.setItem(SESSION_KEY, id)
    return id
  }, [])

  // Restore the previous conversation on load so a refresh is not a reset.
  useEffect(() => {
    let cancelled = false

    const restore = async () => {
      const stored = localStorage.getItem(SESSION_KEY)
      try {
        if (stored) {
          const history = await fetchMessages(stored)
          if (cancelled) return
          sessionId.current = stored
          setMessages(history)
          return
        }
        await startSession()
      } catch {
        if (cancelled) return
        // A stale or expired session id is expected; a fresh one replaces it.
        try {
          await startSession()
        } catch (startError) {
          if (!cancelled) setError(messageFor(startError))
        }
      }
    }

    void restore()
    return () => {
      cancelled = true
    }
  }, [startSession])

  const send = useCallback(
    async (text: string, length: AnswerLength = 'medium') => {
      const question = text.trim()
      if (!question || isThinking) return

      setError(null)
      setMessages((current) => [
        ...current,
        { role: 'user', content: question, sources: [], refused: false },
      ])
      setIsThinking(true)

      try {
        const id = sessionId.current ?? (await startSession())
        const reply = await sendMessage(id, question, length)
        setMessages((current) => [
          ...current,
          {
            role: 'assistant',
            content: reply.answer,
            sources: reply.sources,
            refused: reply.refused,
          },
        ])
      } catch (sendError) {
        setError(messageFor(sendError))
        // Drop the unanswered question so the transcript matches the backend.
        setMessages((current) => current.slice(0, -1))
      } finally {
        setIsThinking(false)
      }
    },
    [isThinking, startSession],
  )

  const reset = useCallback(async () => {
    const previous = sessionId.current
    sessionId.current = null
    localStorage.removeItem(SESSION_KEY)
    setMessages([])
    setError(null)

    if (previous) {
      try {
        await deleteSession(previous)
      } catch {
        // The session is already unreachable, which is the desired end state.
      }
    }
    try {
      await startSession()
    } catch (startError) {
      setError(messageFor(startError))
    }
  }, [startSession])

  return { messages, isThinking, error, send, reset }
}

function messageFor(error: unknown): string {
  return error instanceof ApiError ? error.message : 'Something went wrong. Please try again.'
}
