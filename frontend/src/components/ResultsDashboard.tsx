import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts'
import type { PredictResponse, PredictRequest, SHAPContribution } from '../types'

interface Props {
  result: PredictResponse | null
  request: PredictRequest | null
  shap: SHAPContribution[] | null
  loadingSHAP: boolean
  loading: boolean
}

const FLEET_AVG = 250

export default function ResultsDashboard({ result, request, shap, loadingSHAP, loading }: Props) {
  if (loading) {
    return (
      <div className="empty-slab" aria-live="polite">
        <h2>Warming up the bench…</h2>
        <p>Scoring 57 measurements against the laboratory record. This usually takes a second.</p>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="empty-slab">
        <h2>Your tailpipe number will land here.</h2>
        <p>
          Fill the test slip on the left and run the bench. You get one large certified
          figure for CO2, then the fuel, cost, and reasoning underneath.
        </p>
      </div>
    )
  }

  const vehicleName = request ? `${request.make} ${request.model}` : 'Test vehicle'
  const vehicleLine = request ? `${vehicleName}, ${request.year}, ${request.engine_size_l.toFixed(1)}L with ${request.cylinders} cylinders` : 'Test vehicle'
  const diff = result.co2_g_km - FLEET_AVG
  const diffText = diff <= 0
    ? `${Math.abs(diff).toFixed(0)} g/km under the fleet average`
    : `${diff.toFixed(0)} g/km over the fleet average`
  const plumePct = Math.min(100, Math.max(4, (result.co2_g_km / 400) * 100))
  const avgPct = (FLEET_AVG / 400) * 100
  const ecoLabel = result.eco_score >= 70 ? 'Light footprint' : result.eco_score >= 40 ? 'Middle of the fleet' : 'Heavy footprint'

  return (
    <div className="readout">
      {result.unseen_vehicle && (
        <div className="notice" role="note">
          This make and model never appeared in lab training, so the bench fell back to
          fleet averages for the nameplate. Treat the figure as a rough guide.
        </div>
      )}
      {result.is_india && (
        <div className="notice ember" role="alert">
          India calibration is experimental and tends to miss high. Compare with a
          Canadian or US run before making a buying decision.
        </div>
      )}
      {result.note && <div className="notice" role="note">{result.note}</div>}

      <section className="slab" aria-live="polite" aria-label={`Predicted CO2 ${result.co2_g_km.toFixed(0)} grams per kilometre`}>
        <h2 className="slab-vehicle">{vehicleLine}</h2>
        <p className="slab-cycle">
          Laboratory cycle estimate, {result.model_version}, scored in {result.inference_ms.toFixed(0)} ms
        </p>
        <div className="slab-digits">
          <span className="slab-number">{result.co2_g_km.toFixed(0)}</span>
          <span className="slab-unit">grams of CO2<br />per kilometre</span>
        </div>
        <div className="slab-scale">
          <div className="slab-scale-track" role="img" aria-label={`${diffText}, fleet average ${FLEET_AVG} grams per kilometre`}>
            <div className="slab-scale-fill" style={{ width: `${plumePct}%` }} />
            <div className="slab-scale-avg" style={{ left: `${avgPct}%` }} />
          </div>
          <div className="slab-scale-meta">
            <span>{diffText}</span>
            <span>Fleet average {FLEET_AVG} g/km, scale 0 to 400</span>
          </div>
        </div>
        <div className="slab-trio">
          <div className="slab-cell">
            <strong>{result.fuel_comb_l_100km.toFixed(2)}</strong>
            <span>L/100 km combined, about {result.fuel_comb_mpg.toFixed(1)} mpg</span>
          </div>
          <div className="slab-cell">
            <strong>${result.annual_cost_usd.toFixed(0)}</strong>
            <span>Fuel a year, 15,000 km at $1.50/L</span>
          </div>
          <div className="slab-cell">
            <strong>{result.eco_score} / 100</strong>
            <span>{ecoLabel}</span>
          </div>
        </div>
      </section>

      <div className="split">
        <section className="ledger" aria-label="Full fuel breakdown">
          <h3>What the bench measured</h3>
          <p className="ledger-sub">City and highway burns plus the uncertainty around each.</p>
          <dl style={{ margin: 0 }}>
            <div className="ledger-row">
              <dt>City</dt>
              <dd>{result.fuel_city_l_100km.toFixed(2)} L/100 km ±{result.uncertainty.fuel_city_std.toFixed(2)}</dd>
            </div>
            <div className="ledger-row">
              <dt>Highway</dt>
              <dd>{result.fuel_hwy_l_100km.toFixed(2)} L/100 km ±{result.uncertainty.fuel_hwy_std.toFixed(2)}</dd>
            </div>
            <div className="ledger-row">
              <dt>Combined</dt>
              <dd>{result.fuel_comb_l_100km.toFixed(2)} L/100 km ±{result.uncertainty.fuel_comb_std.toFixed(2)}</dd>
            </div>
            <div className="ledger-row">
              <dt>Tailpipe CO2</dt>
              <dd>{result.co2_g_km.toFixed(1)} g/km ±{result.uncertainty.co2_std.toFixed(1)}</dd>
            </div>
          </dl>
        </section>

        <section className="teardown" aria-label="Why the number came out this way">
          <h3>Why this number</h3>
          <p className="teardown-sub">The measurements that pushed CO2 up or pulled it down.</p>
          {loadingSHAP && <p className="teardown-sub">Reading the teardown…</p>}
          {!loadingSHAP && !shap && <p className="teardown-sub">Run a test to see the reasoning.</p>}
          {shap && (
            <div style={{ width: '100%', height: 250 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={shap.slice(0, 8).map(s => ({ name: s.feature.replace(/_/g, ' '), value: s.contribution }))}
                  layout="vertical"
                  margin={{ left: 88, right: 12, top: 4, bottom: 4 }}
                >
                  <CartesianGrid strokeDasharray="2 4" stroke="#b9c4b4" horizontal={false} />
                  <XAxis type="number" stroke="#5d6f66" tick={{ fontSize: 11, fontFamily: 'Archivo, sans-serif' }} />
                  <YAxis dataKey="name" type="category" stroke="#33463e" width={110} tick={{ fontSize: 11, fontFamily: 'Archivo, sans-serif' }} />
                  <Tooltip
                    contentStyle={{ background: '#eef1e6', border: '1.5px solid #19251f', borderRadius: 3, fontSize: 13 }}
                  />
                  <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                    {shap.slice(0, 8).map((entry, idx) => (
                      <Cell key={`cell-${idx}`} fill={entry.contribution > 0 ? '#b3351c' : '#2f6b4a'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          <p className="teardown-key">Rust bars raised the figure. Pine bars lowered it. Showing the top eight.</p>
        </section>
      </div>
    </div>
  )
}
