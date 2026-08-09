import { useState } from 'react'
import './App.css'
import type { AnswerLength } from './api/client'
import { BackgroundBlobs } from './components/BackgroundBlobs'
import { ChatInput } from './components/ChatInput'
import { ChatWindow } from './components/ChatWindow'
import { Hero } from './components/Hero'
import { useChat } from './hooks/useChat'

export default function App() {
  const { messages, isThinking, error, send, reset } = useChat()
  const [draft, setDraft] = useState('')
  const [length, setLength] = useState<AnswerLength>('medium')

  const hasConversation = messages.length > 0

  const submit = (text: string) => {
    if (!text.trim() || isThinking) return
    setDraft('')
    void send(text, length)
  }

  return (
    <div className="app">
      <BackgroundBlobs />

      <header className="app__header">
        <div className="brand">
          <span className="brand__mark" />
          Consciousness Assistant
        </div>
        <button
          type="button"
          className="ghost-button"
          onClick={() => void reset()}
          disabled={isThinking || !hasConversation}
        >
          New chat
        </button>
      </header>

      <main className="app__main">
        {hasConversation ? (
          <ChatWindow messages={messages} isThinking={isThinking} />
        ) : (
          <Hero onPick={submit} disabled={isThinking} />
        )}

        <div className="composer-dock">
          {error && <div className="error-banner" role="alert">{error}</div>}
          <ChatInput
            value={draft}
            onChange={setDraft}
            onSubmit={() => submit(draft)}
            length={length}
            onLengthChange={setLength}
            disabled={isThinking}
            variant={hasConversation ? 'docked' : 'hero'}
          />
          {!hasConversation && (
            <p className="footnote">
              Answers come only from the indexed research corpus. Off-topic questions are declined.
            </p>
          )}
        </div>
      </main>
    </div>
  )
}
