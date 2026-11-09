import { useState } from 'react'
import type { PredictRequest, PredictResponse } from '../types'
import { postBatchPredict } from '../lib/api'

interface Row {
  req: PredictRequest
}

const BLANK: PredictRequest = {
  year: 2020,
  make: 'Toyota',
  model: 'Corolla',
  vehicle_class: 'Car',
  engine_size_l: 1.8,
  cylinders: 4,
  transmission: 'A6',
  fuel_type: 'X',
  region: 'Canada',
}

function parseCSV(text: string): PredictRequest[] {
  // Expected header: year,make,model,vehicle_class,engine_size_l,cylinders,transmission,fuel_type,region
  const lines = text.trim().split(/\r?\n/)
  if (lines.length < 2) throw new Error('CSV needs a header row plus at least one vehicle row.')
  const header = lines[0].split(',').map((h) => h.trim().toLowerCase())
  const idx = (name: string) => header.indexOf(name)
  const out: PredictRequest[] = []
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim()
    if (!line) continue
    const cells = line.split(',').map((c) => c.trim())
    const get = (name: string, fb = '') => {
      const k = idx(name)
      return k >= 0 ? cells[k] ?? fb : fb
    }
    out.push({
      year: parseInt(get('year', '2020')) || 2020,
      make: get('make', 'Toyota'),
      model: get('model', 'Corolla'),
      vehicle_class: get('vehicle_class', 'Car'),
      engine_size_l: parseFloat(get('engine_size_l', '1.8')) || 1.8,
      cylinders: parseInt(get('cylinders', '4')) || 4,
      transmission: get('transmission', 'A6'),
      fuel_type: get('fuel_type', 'X'),
      region: get('region', 'Canada') || undefined,
    })
  }
  if (!out.length) throw new Error('No vehicle rows found in the CSV.')
  if (out.length > 500) throw new Error('Batch is capped at 500 vehicles.')
  return out
}

export default function BatchMode() {
  const [rows, setRows] = useState<Row[]>([{ req: { ...BLANK } }, { req: { ...BLANK, make: 'Ford', model: 'F150', engine_size_l: 3.5, cylinders: 6 } }])
  const [results, setResults] = useState<PredictResponse[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const set = (i: number, patch: Partial<PredictRequest>) =>
    setRows((prev) => prev.map((r, k) => (k === i ? { req: { ...r.req, ...patch } } : r)))

  const run = async () => {
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const res = await postBatchPredict(rows.map((r) => r.req))
      setResults(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Batch scoring failed.')
    } finally {
      setLoading(false)
    }
  }

  const onFile = async (f: File | undefined) => {
    if (!f) return
    setError(null)
    try {
      const text = await f.text()
      const parsed = parseCSV(text)
      setRows(parsed.slice(0, 50).map((req) => ({ req })))
      if (parsed.length > 50) setError(`Loaded the first 50 of ${parsed.length} rows to keep the table readable.`)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not read that CSV.')
    }
  }

  return (
    <div className="readout">
      <section className="ledger" aria-label="Batch scoring">
        <h3>Score a batch</h3>
        <p className="ledger-sub">
          Edit rows below or drop in a CSV with columns year, make, model, vehicle class, engine size,
          cylinders, transmission, fuel type, region. One call scores the whole table.
        </p>
        <div className="compare-controls">
          <button type="button" className="run-button" style={{ width: 'auto', marginTop: 0 }} onClick={run} disabled={loading || rows.length === 0}>
            {loading ? 'Scoring batch…' : `Score ${rows.length} vehicle${rows.length === 1 ? '' : 's'}`}
          </button>
          <button
            type="button"
            className="ghost-button"
            disabled={rows.length >= 12}
            onClick={() => setRows((p) => [...p, { req: { ...BLANK } }])}
          >
            Add a row ({rows.length} of 12)
          </button>
          <label className="ghost-button batch-file">
            Load CSV
            <input type="file" accept=".csv,text/csv" hidden onChange={(e) => onFile(e.target.files?.[0])} />
          </label>
        </div>
        {error && (
          <div className="form-error" role="alert" style={{ marginBottom: 12 }}>
            {error}
          </div>
        )}
        <div className="compare-table-wrap">
          <table className="compare-table batch-table">
            <thead>
              <tr>
                <th scope="col">Year</th>
                <th scope="col">Make</th>
                <th scope="col">Model</th>
                <th scope="col">Engine L</th>
                <th scope="col">Cyl</th>
                <th scope="col">Gearbox</th>
                <th scope="col">Fuel</th>
                <th scope="col">Body</th>
                <th scope="col"><span className="sr-only">Remove</span></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <td>
                    <input type="number" min={1995} max={2026} value={r.req.year} onChange={(e) => set(i, { year: parseInt(e.target.value) || 2020 })} aria-label={`Row ${i + 1} year`} />
                  </td>
                  <td>
                    <input type="text" value={r.req.make} onChange={(e) => set(i, { make: e.target.value })} aria-label={`Row ${i + 1} make`} />
                  </td>
                  <td>
                    <input type="text" value={r.req.model} onChange={(e) => set(i, { model: e.target.value })} aria-label={`Row ${i + 1} model`} />
                  </td>
                  <td>
                    <input type="number" min={0.5} max={8} step={0.1} value={r.req.engine_size_l} onChange={(e) => set(i, { engine_size_l: parseFloat(e.target.value) || 1.8 })} aria-label={`Row ${i + 1} engine size`} />
                  </td>
                  <td>
                    <input type="number" min={2} max={16} value={r.req.cylinders} onChange={(e) => set(i, { cylinders: parseInt(e.target.value) || 4 })} aria-label={`Row ${i + 1} cylinders`} />
                  </td>
                  <td>
                    <input type="text" value={r.req.transmission} onChange={(e) => set(i, { transmission: e.target.value })} aria-label={`Row ${i + 1} gearbox`} />
                  </td>
                  <td>
                    <input type="text" value={r.req.fuel_type} onChange={(e) => set(i, { fuel_type: e.target.value })} aria-label={`Row ${i + 1} fuel`} />
                  </td>
                  <td>
                    <input type="text" value={r.req.vehicle_class} onChange={(e) => set(i, { vehicle_class: e.target.value })} aria-label={`Row ${i + 1} body style`} />
                  </td>
                  <td>
                    <button type="button" className="ghost-button batch-remove" disabled={rows.length <= 1} onClick={() => setRows(rows.filter((_, k) => k !== i))} aria-label={`Remove row ${i + 1}`}>
                      ×
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {results && (
        <section className="ledger" aria-label="Batch results" aria-live="polite">
          <h3>Batch results</h3>
          <p className="ledger-sub">Combined burn and tailpipe figure per row, in the order submitted.</p>
          <div className="compare-table-wrap">
            <table className="compare-table">
              <thead>
                <tr>
                  <th scope="col">Vehicle</th>
                  <th scope="col">Combined</th>
                  <th scope="col">CO2</th>
                  <th scope="col">Yearly fuel</th>
                  <th scope="col">Eco</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i}>
                    <th scope="row">
                      {rows[i]?.req.make} {rows[i]?.req.model} {rows[i]?.req.year}
                    </th>
                    <td>{r.fuel_comb_l_100km.toFixed(2)} L/100 km</td>
                    <td>{r.co2_g_km.toFixed(0)} g/km</td>
                    <td>${r.annual_cost_usd.toFixed(0)}</td>
                    <td>{r.eco_score} / 100</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
