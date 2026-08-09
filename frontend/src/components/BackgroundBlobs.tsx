export function BackgroundBlobs() {
  return (
    <div className="backdrop" aria-hidden="true">
      <div className="backdrop__grid" />
      <div className="backdrop__blob backdrop__blob--left" />
      <div className="backdrop__blob backdrop__blob--right" />
    </div>
  )
}
