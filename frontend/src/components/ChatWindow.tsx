import { useEffect, useRef } from 'react'
import type { ChatMessage } from '../api/client'
import { MessageBubble } from './MessageBubble'
import { TypingIndicator } from './TypingIndicator'

interface ChatWindowProps {
  messages: ChatMessage[]
  isThinking: boolean
}

export function ChatWindow({ messages, isThinking }: ChatWindowProps) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, isThinking])

  return (
    <div className="conversation">
      <div className="conversation__inner">
        {messages.map((message, index) => (
          <MessageBubble key={`${message.role}-${index}`} message={message} />
        ))}
        {isThinking && <TypingIndicator />}
        <div ref={endRef} />
      </div>
    </div>
  )
}
