import { useMemo, useState } from 'react'
import type { PredictRequest, PredictResponse } from '../types'
import { postPredict } from '../lib/api'
import VehicleForm from './VehicleForm'

interface CompareItem {
  req: PredictRequest
  res: PredictResponse | null
  loading: boolean
  error: string | null
}

const STARTERS: PredictRequest[] = [
  { year: 2020, make: 'Toyota', model: 'Corolla', vehicle_class: 'Car', engine_size_l: 1.8, cylinders: 4, transmission: 'A6', fuel_type: 'X', region: 'Canada' },
  { year: 2020, make: 'Ford', model: 'F150', vehicle_class: 'Truck', engine_size_l: 3.5, cylinders: 6, transmission: 'A10', fuel_type: 'X', region: 'Canada' },
]

const NEXT: PredictRequest = {
  year: 2022, make: 'Honda', model: 'Civic', vehicle_class: 'Car',
  engine_size_l: 2.0, cylinders: 4, transmission: 'A6', fuel_type: 'X', region: 'Canada',
}

export default function CompareMode() {
  const [items, setItems] = useState<CompareItem[]>(
    STARTERS.map((req) => ({ req, res: null, loading: false, error: null })),
  )
  const [scoringAll, setScoringAll] = useState(false)

  const predictOne = async (idx: number, req: PredictRequest) => {
    setItems((prev) => prev.map((it, i) => (i === idx ? { ...it, loading: true, error: null } : it)))
    try {
      const j = await postPredict(req)
      setItems((prev) => prev.map((it, i) => (i === idx ? { ...it, res: j, req, loading: false } : it)))
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Scoring failed.'
      setItems((prev) => prev.map((it, i) => (i === idx ? { ...it, loading: false, error: msg } : it)))
    }
  }

  const predictAll = async () => {
    setScoringAll(true)
    try {
      const snapshot = items.map((it) => it.req)
      const settled = await Promise.allSettled(snapshot.map((req) => postPredict(req)))
      setItems((prev) =>
        prev.map((it, i) => {
          const s = settled[i]
          if (s.status === 'fulfilled') return { ...it, res: s.value, loading: false, error: null }
          return {
            ...it,
            loading: false,
            error: s.reason instanceof Error ? s.reason.message : 'Scoring failed.',
          }
        }),
      )
    } finally {
      setScoringAll(false)
    }
  }

  const addItem = () => {
    if (items.length >= 4) return
    setItems([...items, { req: { ...NEXT }, res: null, loading: false, error: null }])
  }

  const winnerIdx = useMemo(() => {
    let best = -1
    let bestCo2 = Infinity
    items.forEach((it, idx) => {
      if (it.res && it.res.co2_g_km < bestCo2) {
        bestCo2 = it.res.co2_g_km
        best = idx
      }
    })
    return best
  }, [items])

  const scored = items.filter((it) => it.res).length

  return (
    <div className="readout">
      <section className="ledger" aria-label="Compare vehicles">
        <h3>Line up two to four cars</h3>
        <p className="ledger-sub">
          Each car gets its own slip. Score them together and the lightest tailpipe takes the bench.
          Lowest CO2 wins{scored > 0 ? `, ${scored} of ${items.length} scored so far` : ''}.
        </p>
        <div className="compare-controls">
          <button onClick={predictAll} className="run-button" style={{ width: 'auto', marginTop: 0 }} disabled={scoringAll}>
            {scoringAll ? 'Scoring…' : 'Score every car'}
          </button>
          <button onClick={addItem} disabled={items.length >= 4} className="ghost-button">
            Add a car ({items.length} of 4)
          </button>
        </div>
        <div className={`compare-grid cols-${items.length}`}>
          {items.map((it, idx) => (
            <article key={idx} className={winnerIdx === idx ? 'compare-card winner' : 'compare-card'}>
              <div className="compare-card-head">
                <span>
                  Car {idx + 1}
                  {winnerIdx === idx && (
                    <span className="winner-stamp" style={{ marginLeft: 8 }}>
                      Lightest
                    </span>
                  )}
                </span>
                {items.length > 1 && (
                  <button
                    onClick={() => setItems(items.filter((_, i) => i !== idx))}
                    className="ghost-button"
                    style={{ padding: '2px 8px' }}
                    aria-label={`Remove car ${idx + 1}`}
                  >
                    Remove
                  </button>
                )}
              </div>
              <div style={{ padding: 12 }}>
                <VehicleForm
                  onPredict={(req) => predictOne(idx, req)}
                  loading={it.loading || scoringAll}
                  initial={it.req}
                  compact
                  submitLabel={it.res ? 'Score again' : 'Score this car'}
                />
                {it.error && (
                  <div className="form-error" role="alert">
                    {it.error}
                  </div>
                )}
                {it.res && (
                  <dl style={{ margin: '12px 0 0' }}>
                    <div className="ledger-row">
                      <dt>Combined</dt>
                      <dd>{it.res.fuel_comb_l_100km.toFixed(2)} L/100 km</dd>
                    </div>
                    <div className="ledger-row">
                      <dt>CO2</dt>
                      <dd>{it.res.co2_g_km.toFixed(0)} g/km</dd>
                    </div>
                    <div className="ledger-row">
                      <dt>Yearly fuel</dt>
                      <dd>${it.res.annual_cost_usd.toFixed(0)}</dd>
                    </div>
                    <div className="ledger-row">
                      <dt>Eco mark</dt>
                      <dd>{it.res.eco_score} / 100</dd>
                    </div>
                  </dl>
                )}
              </div>
            </article>
          ))}
        </div>
      </section>

      {items.some((it) => it.res) && (
        <section className="ledger" aria-label="Side by side figures">
          <h3>Side by side</h3>
          <p className="ledger-sub">The same figures in one ledger so the gap is easy to read.</p>
          <div className="compare-table-wrap">
            <table className="compare-table">
              <thead>
                <tr>
                  <th scope="col">Figure</th>
                  {items.map((_, idx) => (
                    <th key={idx} scope="col" className={winnerIdx === idx ? 'winner-col' : ''}>
                      Car {idx + 1}{winnerIdx === idx ? ' (lightest)' : ''}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">Nameplate</th>
                  {items.map((it, idx) => (
                    <td key={idx}>
                      {it.req.make} {it.req.model}
                    </td>
                  ))}
                </tr>
                <tr>
                  <th scope="row">Combined</th>
                  {items.map((it, idx) => (
                    <td key={idx}>{it.res ? `${it.res.fuel_comb_l_100km.toFixed(2)} L/100 km` : 'Not scored'}</td>
                  ))}
                </tr>
                <tr>
                  <th scope="row">CO2</th>
                  {items.map((it, idx) => (
                    <td key={idx}>
                      {it.res
                        ? `${it.res.co2_g_km.toFixed(0)} g/km${
                            winnerIdx >= 0 && idx !== winnerIdx && items[winnerIdx].res
                              ? ` (+${(it.res.co2_g_km - items[winnerIdx].res!.co2_g_km).toFixed(0)})`
                              : ''
                          }`
                        : 'Not scored'}
                    </td>
                  ))}
                </tr>
                <tr>
                  <th scope="row">Yearly fuel</th>
                  {items.map((it, idx) => (
                    <td key={idx}>{it.res ? `$${it.res.annual_cost_usd.toFixed(0)}` : 'Not scored'}</td>
                  ))}
                </tr>
                <tr>
                  <th scope="row">Eco mark</th>
                  {items.map((it, idx) => (
                    <td key={idx}>{it.res ? `${it.res.eco_score} / 100` : 'Not scored'}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
