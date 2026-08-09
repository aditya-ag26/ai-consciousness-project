interface SourceChipsProps {
  sources: string[]
}

export function SourceChips({ sources }: SourceChipsProps) {
  if (sources.length === 0) return null

  return (
    <div className="sources">
      <span className="sources__label">Sources</span>
      {sources.map((source) => (
        <span key={source} className="source-chip" title={source}>
          {source}
        </span>
      ))}
    </div>
  )
}
