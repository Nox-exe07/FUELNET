import { useEffect, useState } from 'react'
import { getHealth, getModelInfo, type ModelInfo } from '../lib/api'

export default function StatusStrip() {
  const [health, setHealth] = useState<'up' | 'down' | 'checking'>('checking')
  const [info, setInfo] = useState<ModelInfo | null>(null)

  useEffect(() => {
    let live = true
    getHealth()
      .then(() => live && setHealth('up'))
      .catch(() => live && setHealth('down'))
    getModelInfo()
      .then((m) => live && setInfo(m))
      .catch(() => {})
    return () => {
      live = false
    }
  }, [])

  return (
    <div className="status-strip" role="status" aria-live="polite">
      <span className="status-dot" data-state={health} aria-hidden="true" />
      <span>
        {health === 'up' ? 'Bench connected' : health === 'down' ? 'Bench offline' : 'Checking bench'}
      </span>
      {info?.model_version && <span className="status-sep" aria-hidden="true">·</span>}
      {info?.model_version && <span>{info.model_version}</span>}
      {typeof info?.test_R2 === 'number' && (
        <>
          <span className="status-sep" aria-hidden="true">·</span>
          <span>Test R² {info.test_R2.toFixed(3)}</span>
        </>
      )}
      {typeof info?.n_features === 'number' && (
        <>
          <span className="status-sep" aria-hidden="true">·</span>
          <span>{info.n_features} measurements</span>
        </>
      )}
    </div>
  )
}
