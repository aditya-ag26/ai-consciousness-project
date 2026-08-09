const SUGGESTIONS = [
  'What is the hard problem of consciousness?',
  'What are qualia?',
  'Can machines be sentient?',
]

interface HeroProps {
  onPick: (question: string) => void
  disabled: boolean
}

export function Hero({ onPick, disabled }: HeroProps) {
  return (
    <section className="hero">
      <h1 className="hero__title">Explore consciousness research</h1>
      <p className="hero__subtitle">
        Ask questions across academic papers and expert transcripts on consciousness, sentience,
        and the nature of mind. Every answer is grounded in cited sources.
      </p>
      <div className="hero__hints">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            className="hint"
            onClick={() => onPick(suggestion)}
            disabled={disabled}
          >
            {suggestion}
          </button>
        ))}
      </div>
    </section>
  )
}
