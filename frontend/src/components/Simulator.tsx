import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { PredictRequest, PredictResponse } from '../types'

interface Props {
  result: PredictResponse | null
  request: PredictRequest | null
}

const TRIPS = [5, 25, 100]
const SPEEDS = [30, 60, 90, 110]
const FUEL_PRICE = 1.5

interface Puff {
  x: number
  y: number
  r: number
  alpha: number
  vx: number
}

function demoDurationSec(tripKm: number): number {
  return Math.min(30, Math.max(10, 8 + tripKm * 0.4))
}

export default function Simulator({ result, request }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef<number>(0)
  const lastTsRef = useRef<number>(0)
  const progressRef = useRef(0)
  const puffsRef = useRef<Puff[]>([])
  const dashOffsetRef = useRef(0)
  const wheelAngleRef = useRef(0)

  const [tripKm, setTripKm] = useState(25)
  const [speedKmh, setSpeedKmh] = useState(60)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0) // mirrored for readout, 0..1
  const [reducedMotion] = useState(
    () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches,
  )

  const duration = useMemo(() => demoDurationSec(tripKm), [tripKm])

  const reset = useCallback(() => {
    progressRef.current = 0
    puffsRef.current = []
    dashOffsetRef.current = 0
    setProgress(0)
    setRunning(false)
    lastTsRef.current = 0
  }, [])

  useEffect(() => {
    reset()
  }, [tripKm, result, reset])

  const distKm = progress * tripKm
  const fuelL = result ? (result.fuel_comb_l_100km / 100) * distKm : 0
  const co2Kg = result ? (result.co2_g_km * distKm) / 1000 : 0
  const cost = fuelL * FUEL_PRICE
  const finished = progress >= 1

  // ---- canvas paint ----
  const paint = useCallback(
    (ctx: CanvasRenderingContext2D, w: number, h: number, prog: number) => {
      const fuelNow = result ? (result.fuel_comb_l_100km / 100) * prog * tripKm : 0
      const co2 = result?.co2_g_km ?? 180
      // paper
      ctx.clearRect(0, 0, w, h)
      ctx.fillStyle = '#eef1e6'
      ctx.fillRect(0, 0, w, h)

      // faint chalk grid ticks (odometer marks)
      ctx.strokeStyle = '#b9c4b4'
      ctx.lineWidth = 1
      const kmMarks = Math.min(tripKm, 10)
      ctx.font = '11px Archivo, sans-serif'
      ctx.fillStyle = '#5d6f66'
      for (let i = 0; i <= kmMarks; i++) {
        const x = 24 + (i / kmMarks) * (w - 48)
        ctx.beginPath()
        ctx.moveTo(x, h - 46)
        ctx.lineTo(x, h - 38)
        ctx.stroke()
        const labelKm = ((tripKm * i) / kmMarks).toFixed(tripKm > 10 ? 0 : 1)
        ctx.fillText(`${labelKm}`, x - 8, h - 24)
      }
      ctx.fillText('km', w - 30, h - 24)

      // road band
      const roadY = Math.round(h * 0.42)
      const roadH = Math.round(h * 0.3)
      ctx.fillStyle = '#22443a'
      ctx.fillRect(0, roadY, w, roadH)
      ctx.fillStyle = '#152b25'
      ctx.fillRect(0, roadY + roadH - 8, w, 8)

      // moving centre dashes — speed of dashes follows selected speed
      const dashW = 34
      const gap = 26
      ctx.fillStyle = '#eef1e6'
      const off = dashOffsetRef.current % (dashW + gap)
      for (let x = -dashW + off; x < w + dashW; x += dashW + gap) {
        ctx.fillRect(x, roadY + roadH / 2 - 2, dashW, 4)
      }

      // car position travels left -> right with progress
      const carW = 132
      const carH = 44
      const carX = 24 + prog * (w - 48 - carW)
      const carY = roadY - carH + 14

      // exhaust puffs behind car (spawn handled in tick; draw here)
      for (const p of puffsRef.current) {
        ctx.beginPath()
        ctx.fillStyle = `rgba(179,53,28,${p.alpha.toFixed(3)})`
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx.fill()
      }

      // shadow
      ctx.fillStyle = 'rgba(21,43,37,0.55)'
      ctx.beginPath()
      ctx.ellipse(carX + carW / 2, roadY + roadH - 10, carW / 2, 7, 0, 0, Math.PI * 2)
      ctx.fill()

      // body
      ctx.fillStyle = '#f4f6ee'
      ctx.strokeStyle = '#19251f'
      ctx.lineWidth = 2
      const bx = carX
      const by = carY
      ctx.beginPath()
      // rounded body path
      if (ctx.roundRect) {
        ctx.roundRect(bx, by + 12, carW, 26, 7)
      } else {
        ctx.rect(bx, by + 12, carW, 26)
      }
      ctx.fill()
      ctx.stroke()
      // cabin
      ctx.beginPath()
      ctx.moveTo(bx + 30, by + 12)
      ctx.lineTo(bx + 44, by - 4)
      ctx.lineTo(bx + 88, by - 4)
      ctx.lineTo(bx + 102, by + 12)
      ctx.closePath()
      ctx.fill()
      ctx.stroke()
      // window split
      ctx.beginPath()
      ctx.moveTo(bx + 66, by - 4)
      ctx.lineTo(bx + 66, by + 12)
      ctx.stroke()

      // wheels (rotate spokes)
      const wheelY = roadY + roadH - 12
      for (const wx of [bx + 30, bx + carW - 30]) {
        ctx.beginPath()
        ctx.fillStyle = '#19251f'
        ctx.arc(wx, wheelY, 13, 0, Math.PI * 2)
        ctx.fill()
        ctx.beginPath()
        ctx.fillStyle = '#eef1e6'
        ctx.arc(wx, wheelY, 5, 0, Math.PI * 2)
        ctx.fill()
        const a = wheelAngleRef.current
        ctx.strokeStyle = '#eef1e6'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(wx + Math.cos(a) * 11, wheelY + Math.sin(a) * 11)
        ctx.lineTo(wx - Math.cos(a) * 11, wheelY - Math.sin(a) * 11)
        ctx.stroke()
        ctx.lineWidth = 2
        ctx.strokeStyle = '#19251f'
      }

      // exhaust pipe marker
      ctx.fillStyle = '#19251f'
      ctx.fillRect(bx - 2, wheelY - 8, 6, 5)

      // fuel tank gauge top-right of canvas
      const tankW = 120
      const tankH = 10
      const tx = w - tankW - 16
      const ty = 14
      ctx.fillStyle = '#5d6f66'
      ctx.font = '12px Archivo, sans-serif'
      ctx.fillText('tank', tx, ty)
      ctx.strokeStyle = '#19251f'
      ctx.lineWidth = 1.5
      ctx.strokeRect(tx, ty + 5, tankW, tankH)
      // tank drains as fuelL grows vs full-trip need
      const fullNeed = result ? Math.max(0.5, (result.fuel_comb_l_100km / 100) * tripKm) : 1
      const remain = Math.max(0, 1 - fuelNow / fullNeed)
      const intensity = Math.min(1, co2 / 400)
      ctx.fillStyle = intensity > 0.62 ? '#b3351c' : '#2f6b4a'
      ctx.fillRect(tx + 1.5, ty + 6.5, (tankW - 3) * remain, tankH - 3)

      // trip progress hairline at very bottom
      ctx.fillStyle = '#8fa093'
      ctx.fillRect(0, h - 6, w * prog, 3)
    },
    [result, tripKm],
  )

  // ---- animation tick ----
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !result) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const fit = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      const rect = canvas.getBoundingClientRect()
      canvas.width = Math.max(1, Math.round(rect.width * dpr))
      canvas.height = Math.round(240 * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    fit()
    window.addEventListener('resize', fit)

    const co2 = result.co2_g_km
    let mirrorAcc = 0

    const tick = (ts: number) => {
      rafRef.current = requestAnimationFrame(tick)
      const rect = canvas.getBoundingClientRect()
      const w = rect.width
      const h = 240
      if (!lastTsRef.current) lastTsRef.current = ts
      const dt = Math.min(0.05, (ts - lastTsRef.current) / 1000)
      lastTsRef.current = ts

      if (running && progressRef.current < 1) {
        progressRef.current = Math.min(1, progressRef.current + dt / duration)
        // road + wheel motion scale with chosen speed
        const pxRate = 60 + speedKmh * 1.6
        dashOffsetRef.current -= pxRate * dt
        wheelAngleRef.current += (speedKmh / 30) * dt * 4
        // spawn exhaust: rate grows with CO2
        const spawnP = Math.min(0.9, 0.12 + co2 / 400)
        if (Math.random() < spawnP) {
          const carW = 132
          const carX = 24 + progressRef.current * (w - 48 - carW)
          const roadY = Math.round(h * 0.42)
          const roadH = Math.round(h * 0.3)
          puffsRef.current.push({
            x: carX + 2,
            y: roadY + roadH - 20 + (Math.random() - 0.5) * 6,
            r: 3 + Math.random() * 4 + co2 / 200,
            alpha: 0.5,
            vx: -(30 + Math.random() * 30),
          })
          if (puffsRef.current.length > 90) puffsRef.current.shift()
        }
        if (progressRef.current >= 1) setRunning(false)
      }
      // age puffs always (so they clear after finish)
      for (const p of puffsRef.current) {
        p.x += p.vx * dt
        p.r += dt * 6
        p.alpha = Math.max(0, p.alpha - dt * 0.5)
      }
      puffsRef.current = puffsRef.current.filter((p) => p.alpha > 0.01)

      mirrorAcc += dt
      if (mirrorAcc > 0.25) {
        mirrorAcc = 0
        setProgress(progressRef.current)
      }
      paint(ctx, w, h, progressRef.current)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', fit)
    }
  }, [result, running, duration, speedKmh, paint])

  // initial static paint for empty/loading states
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !result) return
    if (running) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const rect = canvas.getBoundingClientRect()
    if (rect.width > 0) paint(ctx, rect.width, 240, progressRef.current)
  }, [paint, result, running, tripKm])

  if (!result || !request) {
    return (
      <div className="ledger">
        <h3>Road check</h3>
        <p className="ledger-sub">
          A short 2D drive that burns fuel at your predicted rate. Run a single test first, then come back
          here to see how that number behaves on the road.
        </p>
        <p className="ledger-sub">No test on file yet. Go to Single test and run one vehicle.</p>
      </div>
    )
  }

  const vehicleLine = `${request.make} ${request.model}, ${request.year}`

  const stepKm = () => {
    progressRef.current = Math.min(1, progressRef.current + 1 / tripKm)
    setProgress(progressRef.current)
  }

  return (
    <div className="readout">
      <section className="ledger sim-top" aria-label="Road check setup">
        <div>
          <h3>Road check</h3>
          <p className="ledger-sub">
            {vehicleLine} at {result.fuel_comb_l_100km.toFixed(2)} L/100 km and {result.co2_g_km.toFixed(0)}{' '}
            g/km. The car below burns at exactly that rate.
          </p>
        </div>
        <div className="sim-controls">
          <div className="sim-group" role="group" aria-label="Trip length">
            <span className="sim-label">Trip</span>
            {TRIPS.map((t) => (
              <button
                key={t}
                type="button"
                className="ghost-button sim-chip"
                aria-pressed={tripKm === t}
                data-active={tripKm === t}
                onClick={() => setTripKm(t)}
              >
                {t} km
              </button>
            ))}
          </div>
          <div className="sim-group" role="group" aria-label="Cruising speed">
            <span className="sim-label">Pace</span>
            {SPEEDS.map((s) => (
              <button
                key={s}
                type="button"
                className="ghost-button sim-chip"
                aria-pressed={speedKmh === s}
                data-active={speedKmh === s}
                onClick={() => setSpeedKmh(s)}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="sim-stage" aria-label="Simulated drive">
        <canvas ref={canvasRef} className="sim-canvas" style={{ width: '100%', height: 240 }} role="img" aria-label={`Simulated drive, ${distKm.toFixed(1)} of ${tripKm} kilometres covered`} />
        <div className="sim-buttons">
          {!reducedMotion ? (
            <>
              <button
                type="button"
                className="run-button sim-go"
                onClick={() => (finished ? reset() : setRunning((v) => !v))}
              >
                {finished ? 'Drive it again' : running ? 'Pause the drive' : progress > 0 ? 'Keep driving' : 'Start the drive'}
              </button>
              <button type="button" className="ghost-button" onClick={reset}>
                Reset
              </button>
            </>
          ) : (
            <>
              <button type="button" className="run-button sim-go" onClick={stepKm} disabled={finished}>
                {finished ? 'Trip complete' : 'Advance 1 km'}
              </button>
              <button type="button" className="ghost-button" onClick={reset}>
                Reset
              </button>
            </>
          )}
        </div>
      </section>

      <div className="split">
        <section className="ledger" aria-label="Trip ledger" aria-live="polite">
          <h3>Trip ledger</h3>
          <p className="ledger-sub">
            {finished
              ? `Trip complete over ${tripKm} km at ${speedKmh} km/h.`
              : `Driving ${distKm.toFixed(1)} of ${tripKm} km at ${speedKmh} km/h.`}
          </p>
          <dl style={{ margin: 0 }}>
            <div className="ledger-row">
              <dt>Distance</dt>
              <dd>
                {distKm.toFixed(1)} / {tripKm} km
              </dd>
            </div>
            <div className="ledger-row">
              <dt>Fuel burned</dt>
              <dd>{fuelL.toFixed(2)} L</dd>
            </div>
            <div className="ledger-row">
              <dt>Tailpipe CO2</dt>
              <dd>{co2Kg.toFixed(2)} kg</dd>
            </div>
            <div className="ledger-row">
              <dt>Fuel cost</dt>
              <dd>${cost.toFixed(2)}</dd>
            </div>
          </dl>
        </section>
        <section className="ledger" aria-label="What the smoke means">
          <h3>Reading the smoke</h3>
          <p className="ledger-sub">
            Puff rate follows your certified figure of {result.co2_g_km.toFixed(0)} g/km. A heavier car
            smokes faster at the same pace.
          </p>
          <dl style={{ margin: 0 }}>
            <div className="ledger-row">
              <dt>Full trip fuel</dt>
              <dd>{((result.fuel_comb_l_100km / 100) * tripKm).toFixed(2)} L</dd>
            </div>
            <div className="ledger-row">
              <dt>Full trip CO2</dt>
              <dd>{((result.co2_g_km * tripKm) / 1000).toFixed(2)} kg</dd>
            </div>
            <div className="ledger-row">
              <dt>Full trip cost</dt>
              <dd>${(((result.fuel_comb_l_100km / 100) * tripKm * FUEL_PRICE).toFixed(2))}</dd>
            </div>
          </dl>
        </section>
      </div>
    </div>
  )
}
