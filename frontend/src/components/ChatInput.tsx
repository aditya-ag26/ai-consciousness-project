import { useEffect, useRef, type KeyboardEvent } from 'react'
import type { AnswerLength } from '../api/client'

const LENGTHS: AnswerLength[] = ['short', 'medium', 'long']

interface ChatInputProps {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  length: AnswerLength
  onLengthChange: (length: AnswerLength) => void
  disabled: boolean
  variant: 'hero' | 'docked'
}

export function ChatInput({
  value,
  onChange,
  onSubmit,
  length,
  onLengthChange,
  disabled,
  variant,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Grow the box with its content instead of scrolling inside a fixed height.
  useEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return
    textarea.style.height = 'auto'
    textarea.style.height = `${textarea.scrollHeight}px`
  }, [value])

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      onSubmit()
    }
  }

  return (
    <form
      className={`composer ${variant === 'hero' ? 'composer--hero' : ''}`}
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit()
      }}
    >
      <textarea
        ref={textareaRef}
        className="composer__input"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask about consciousness, sentience, awareness…"
        rows={1}
        aria-label="Your question"
      />
      <div className="composer__row">
        <div className="length-picker" role="group" aria-label="Answer length">
          {LENGTHS.map((option) => (
            <button
              key={option}
              type="button"
              aria-pressed={length === option}
              onClick={() => onLengthChange(option)}
              disabled={disabled}
            >
              {option}
            </button>
          ))}
        </div>
        <button
          type="submit"
          className="send-button"
          disabled={disabled || value.trim().length === 0}
          aria-label="Send question"
        >
          <ArrowUpIcon />
        </button>
      </div>
    </form>
  )
}

function ArrowUpIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 19V5M12 5l-6 6M12 5l6 6"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
