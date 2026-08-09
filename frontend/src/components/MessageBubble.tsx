import type { ChatMessage } from '../api/client'
import { SourceChips } from './SourceChips'

interface MessageBubbleProps {
  message: ChatMessage
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const className = [
    'message',
    isUser ? 'message--user' : 'message--assistant',
    message.refused ? 'message--refused' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <article className={className}>
      {message.refused && <span className="message__label">Outside knowledge base</span>}
      <div className="message__bubble">{message.content}</div>
      {!isUser && <SourceChips sources={message.sources} />}
    </article>
  )
}
