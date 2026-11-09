import { useState } from 'react'
import VehicleForm from './components/VehicleForm'
import ResultsDashboard from './components/ResultsDashboard'
import CompareMode from './components/CompareMode'
import BatchMode from './components/BatchMode'
import Simulator from './components/Simulator'
import StatusStrip from './components/StatusStrip'
import { postExplain, postPredict } from './lib/api'
import type { PredictRequest, PredictResponse, SHAPContribution } from './types'
import './App.css'

type Tab = 'predict' | 'compare' | 'batch' | 'simulator'

export default function App() {
  const [tab, setTab] = useState<Tab>('predict')
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [lastReq, setLastReq] = useState<PredictRequest | null>(null)
  const [loading, setLoading] = useState(false)
  const [shap, setShap] = useState<SHAPContribution[] | null>(null)
  const [loadingSHAP, setLoadingSHAP] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handlePredict = async (req: PredictRequest) => {
    setLoading(true)
    setError(null)
    setShap(null)
    try {
      const j = await postPredict(req)
      setResult(j)
      setLastReq(req)
      setLoadingSHAP(true)
      try {
        const contribs = await postExplain(req)
        setShap(contribs)
      } catch (e) {
        console.warn('SHAP fetch failed', e)
      } finally {
        setLoadingSHAP(false)
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="bench-head">
        <div className="bench-head-inner">
          <div className="plate">
            <div className="plate-mark" aria-hidden="true">FN</div>
            <div>
              <h1 className="plate-name">FuelNet</h1>
              <p className="plate-desc">
                Test-cell estimate for fuel use and CO<sub>2</sub>, trained on the Canadian
                laboratory cycle from 2000 to 2022.
              </p>
              <span className="plate-stamp">CatBoost v3, 57 measurements per run</span>{' '}
              <StatusStrip />
            </div>
          </div>
          <nav className="bench-tabs" role="tablist" aria-label="Workspace">
            <button role="tab" aria-selected={tab === 'predict'} onClick={() => setTab('predict')}>
              Single test
            </button>
            <button role="tab" aria-selected={tab === 'compare'} onClick={() => setTab('compare')}>
              Compare
            </button>
            <button role="tab" aria-selected={tab === 'batch'} onClick={() => setTab('batch')}>
              Batch
            </button>
            <button role="tab" aria-selected={tab === 'simulator'} onClick={() => setTab('simulator')}>
              Road check
            </button>
          </nav>
        </div>
      </header>

      <main className="bench-main">
        {tab === 'predict' && (
          <div className="bench-layout">
            <div>
              <VehicleForm onPredict={handlePredict} loading={loading} />
              {error && <div className="form-error" role="alert" style={{ marginTop: 12 }}>{error}</div>}
            </div>
            <div>
              <ResultsDashboard
                result={result}
                request={lastReq}
                shap={shap}
                loadingSHAP={loadingSHAP}
                loading={loading}
              />
            </div>
          </div>
        )}

        {tab === 'compare' && <CompareMode />}

        {tab === 'batch' && <BatchMode />}

        {tab === 'simulator' && <Simulator result={result} request={lastReq} />}
      </main>

      <footer className="bench-foot">
        <div className="bench-foot-inner">
          <span>FuelNet bench, laboratory cycle not your driveway, 15,000 km per year at $1.50/L for cost math</span>
          <span>Canada and US figures are solid, India figures are experimental</span>
        </div>
      </footer>
    </div>
  )
}
