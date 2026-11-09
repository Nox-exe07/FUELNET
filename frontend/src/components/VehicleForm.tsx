import { useEffect, useId, useState } from 'react'
import type { PredictRequest } from '../types'
import { getOptions } from '../lib/api'

interface Props {
  onPredict: (data: PredictRequest) => void
  loading: boolean
  initial?: PredictRequest
  compact?: boolean
  submitLabel?: string
}

const FALLBACK = {
  makes: ['Toyota', 'Ford', 'Honda', 'Chevrolet', 'BMW', 'Audi', 'Nissan', 'Hyundai', 'Kia', 'Volkswagen'],
  models_by_make: {
    Toyota: ['Corolla', 'Camry', 'RAV4', 'Tacoma', 'Highlander'],
    Ford: ['F150', 'Mustang', 'Explorer', 'Focus', 'Fusion'],
    Honda: ['Civic', 'Accord', 'CRV', 'Pilot'],
  } as Record<string, string[]>,
  vehicle_classes: ['Car', 'SUV', 'Truck', 'Van', 'Bus', 'Motorcycle', 'Other'],
  fuel_types: ['Petrol', 'Diesel', 'Premium', 'E85', 'CNG', 'Electric'],
  transmissions: ['A4', 'A6', 'A8', 'M5', 'M6', 'AS6', 'AV', 'CVT'],
  road_types: ['City', 'Highway', 'Rural'],
  traffic_conditions: ['Free flow', 'Moderate', 'Heavy'],
  regions: ['India', 'Canada', 'US', 'Global'],
}

const defaultForm: PredictRequest = {
  year: 2020,
  make: 'Toyota',
  model: 'Corolla',
  vehicle_class: 'Car',
  engine_size_l: 1.8,
  cylinders: 4,
  transmission: 'A6',
  fuel_type: 'X',
  region: 'Canada',
  road_type: 'Highway',
  traffic_condition: 'Moderate',
}

export default function VehicleForm({ onPredict, loading, initial, compact, submitLabel }: Props) {
  const [form, setForm] = useState<PredictRequest>(initial ?? defaultForm)
  const [makes, setMakes] = useState<string[]>(FALLBACK.makes)
  const [modelsByMake, setModelsByMake] = useState<Record<string, string[]>>(FALLBACK.models_by_make)
  const [modelsLower, setModelsLower] = useState<Record<string, string[]>>({})
  const [vehicleClasses, setVehicleClasses] = useState<string[]>(FALLBACK.vehicle_classes)
  const [fuelTypes, setFuelTypes] = useState<string[]>(FALLBACK.fuel_types)
  const [transmissions, setTransmissions] = useState<string[]>(FALLBACK.transmissions)
  const [roadTypes, setRoadTypes] = useState<string[]>(FALLBACK.road_types)
  const [traffic, setTraffic] = useState<string[]>(FALLBACK.traffic_conditions)
  const [regions, setRegions] = useState<string[]>(FALLBACK.regions)
  const [optionsReady, setOptionsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const uid = useId().replace(/[^a-zA-Z0-9]/g, '')

  useEffect(() => {
    let live = true
    getOptions()
      .then((o) => {
        if (!live) return
        if (o.makes?.length) setMakes(o.makes)
        if (o.models_by_make) setModelsByMake(o.models_by_make)
        if (o.models_by_make_lower) setModelsLower(o.models_by_make_lower)
        if (o.vehicle_classes?.length) setVehicleClasses(o.vehicle_classes)
        if (o.fuel_types?.length) setFuelTypes(o.fuel_types)
        if (o.transmissions?.length) setTransmissions(o.transmissions)
        if (o.road_types?.length) setRoadTypes(o.road_types)
        if (o.traffic_conditions?.length) setTraffic(o.traffic_conditions)
        if (o.regions?.length) setRegions(o.regions)
        setOptionsReady(true)
      })
      .catch(() => {
        if (live) setOptionsReady(true)
      })
    return () => {
      live = false
    }
  }, [])

  useEffect(() => {
    if (initial) setForm(initial)
  }, [initial])

  const getModelsForMake = (make: string): string[] => {
    if (modelsByMake[make]) return modelsByMake[make]
    const low = make.toLowerCase()
    if (modelsLower[low]) return modelsLower[low]
    for (const k of Object.keys(modelsByMake)) {
      if (k.toLowerCase() === low) return modelsByMake[k]
    }
    return []
  }

  const modelsForMake = getModelsForMake(form.make)
  const update = (patch: Partial<PredictRequest>) => setForm((f) => ({ ...f, ...patch }))

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    if (!form.make.trim() || !form.model.trim()) {
      setError('Name the make and model, even if the bench has never seen them. Unseen nameplates get a fleet-average fallback.')
      return
    }
    if (form.engine_size_l < 0.5 || form.engine_size_l > 8) {
      setError('Engine size needs to sit between 0.5 and 8.0 litres. Adjust the slider and try again.')
      return
    }
    if (form.year < 1995 || form.year > 2026) {
      setError('Year needs to sit between 1995 and 2026. The lab only certified cars in that window.')
      return
    }
    if (form.cylinders < 2 || form.cylinders > 16) {
      setError('Cylinders need to sit between 2 and 16.')
      return
    }
    onPredict({ ...form, make: form.make.trim(), model: form.model.trim() })
  }

  return (
    <form onSubmit={handleSubmit} className={compact ? 'logbook compact' : 'logbook'} aria-label="Vehicle test slip">
      {!compact && (
        <div className="logbook-head">
          <h2>Vehicle test slip</h2>
          <p>Describe the car as it left the factory. The bench fills in the rest.</p>
        </div>
      )}
      <div className="logbook-body">
        {error && (
          <div className="form-error" role="alert">
            {error}
          </div>
        )}

        <div className="field-row">
          <label className="field">
            <span>Model year</span>
            <input
              type="number"
              min={1995}
              max={2026}
              value={form.year}
              onChange={(e) => update({ year: parseInt(e.target.value) || 2020 })}
            />
          </label>
          <label className="field">
            <span>Make</span>
            <input
              type="text"
              list={`${uid}-makes`}
              value={form.make}
              autoComplete="off"
              placeholder="Toyota"
              onChange={(e) => {
                const mk = e.target.value
                const models = getModelsForMake(mk)
                update({
                  make: mk,
                  model: models.length === 1 ? models[0] : form.model,
                })
              }}
              onBlur={(e) => {
                const mk = e.target.value.trim()
                if (!mk) return
                const models = getModelsForMake(mk)
                if (models.length && !models.some((m) => m.toLowerCase() === form.model.toLowerCase())) {
                  update({ model: models[0] })
                }
              }}
            />
            <datalist id={`${uid}-makes`}>
              {makes.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </label>
        </div>

        <div className="field-row">
          <label className="field field-full">
            <span>Model {modelsForMake.length ? '' : '(unseen nameplate, fleet average applies)'}</span>
            <input
              type="text"
              list={`${uid}-models`}
              value={form.model}
              autoComplete="off"
              placeholder="Corolla"
              onChange={(e) => update({ model: e.target.value })}
            />
            <datalist id={`${uid}-models`}>
              {modelsForMake.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </label>
        </div>

        <div className="field-row">
          <label className="field">
            <span>Body style</span>
            <select value={form.vehicle_class} onChange={(e) => update({ vehicle_class: e.target.value })}>
              {vehicleClasses.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Cylinders</span>
            <input
              type="number"
              min={2}
              max={16}
              value={form.cylinders}
              onChange={(e) => update({ cylinders: parseInt(e.target.value) || 4 })}
            />
          </label>
        </div>

        <div className="field-row">
          <div className="field field-full">
            <span>Engine size, {form.engine_size_l.toFixed(1)} litres</span>
            <input
              type="range"
              min={0.5}
              max={8}
              step={0.1}
              value={form.engine_size_l}
              onChange={(e) => update({ engine_size_l: parseFloat(e.target.value) })}
              aria-label="Engine size in litres"
            />
          </div>
        </div>

        <div className="field-row">
          <label className="field">
            <span>Gearbox</span>
            <input
              type="text"
              list={`${uid}-gear`}
              value={form.transmission}
              autoComplete="off"
              onChange={(e) => update({ transmission: e.target.value })}
            />
            <datalist id={`${uid}-gear`}>
              {transmissions.map((t) => (
                <option key={t} value={t} />
              ))}
            </datalist>
          </label>
          <label className="field">
            <span>Fuel</span>
            <select value={form.fuel_type} onChange={(e) => update({ fuel_type: e.target.value })}>
              {fuelTypes.map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="field-row">
          <label className="field field-full">
            <span>Where it will be driven</span>
            <select value={form.region || 'Canada'} onChange={(e) => update({ region: e.target.value })}>
              {regions.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </label>
        </div>

        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="advanced-toggle"
          aria-expanded={showAdvanced}
        >
          {showAdvanced ? 'Hide road and weather details' : 'Add road and weather details'}
        </button>

        {showAdvanced && (
          <div style={{ marginTop: 4 }}>
            <div className="field-row">
              <label className="field">
                <span>Power, hp</span>
                <input
                  type="number"
                  placeholder="Leave blank to estimate"
                  value={form.power_hp ?? ''}
                  onChange={(e) => update({ power_hp: e.target.value ? parseFloat(e.target.value) : undefined })}
                />
              </label>
              <label className="field">
                <span>Mileage, km</span>
                <input
                  type="number"
                  placeholder="Leave blank to estimate"
                  value={form.mileage_km ?? ''}
                  onChange={(e) => update({ mileage_km: e.target.value ? parseFloat(e.target.value) : undefined })}
                />
              </label>
            </div>
            <div className="field-row">
              <label className="field">
                <span>Cruising speed, km/h</span>
                <input
                  type="number"
                  placeholder="Leave blank to estimate"
                  value={form.speed_kmh ?? ''}
                  onChange={(e) => update({ speed_kmh: e.target.value ? parseFloat(e.target.value) : undefined })}
                />
              </label>
              <label className="field">
                <span>Outside temperature, C</span>
                <input
                  type="number"
                  placeholder="Leave blank to estimate"
                  value={form.temperature_c ?? ''}
                  onChange={(e) => update({ temperature_c: e.target.value ? parseFloat(e.target.value) : undefined })}
                />
              </label>
            </div>
            <div className="field-row">
              <label className="field">
                <span>Road</span>
                <select value={form.road_type || 'Highway'} onChange={(e) => update({ road_type: e.target.value })}>
                  {roadTypes.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Traffic</span>
                <select
                  value={form.traffic_condition || 'Moderate'}
                  onChange={(e) => update({ traffic_condition: e.target.value })}
                >
                  {traffic.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="field-row">
              <label className="field">
                <span>Humidity, percent</span>
                <input
                  type="number"
                  placeholder="Leave blank to estimate"
                  value={form.humidity_pct ?? ''}
                  onChange={(e) => update({ humidity_pct: e.target.value ? parseFloat(e.target.value) : undefined })}
                />
              </label>
              <label className="field">
                <span>Driven wheels</span>
                <select value={form.drive || 'UNKNOWN'} onChange={(e) => update({ drive: e.target.value })}>
                  <option value="UNKNOWN">Not sure</option>
                  <option value="FWD">Front</option>
                  <option value="RWD">Rear</option>
                  <option value="AWD">All</option>
                </select>
              </label>
            </div>
          </div>
        )}

        {!compact && (
          <p className="form-note">
            Blank road details are filled with typical lab values.
            {!optionsReady ? ' Option lists are still loading, factory defaults shown.' : ''} Cars set to
            India use a separate calibration that is still experimental.
          </p>
        )}

        <button type="submit" disabled={loading} className="run-button">
          {loading ? 'Running the bench test…' : (submitLabel ?? 'Predict emissions')}
        </button>
      </div>
    </form>
  )
}
