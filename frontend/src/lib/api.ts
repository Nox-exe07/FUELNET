import type { PredictRequest, PredictResponse, SHAPContribution } from '../types'

const RAW_BASE = import.meta.env.VITE_API_URL as string | undefined
// Strip trailing slashes; empty string means same-origin (vite proxy handles /predict etc.)
export const API_BASE = (RAW_BASE ?? '').replace(/\/+$/, '')

function url(path: string): string {
  return `${API_BASE}${path}`
}

async function parseError(r: Response): Promise<string> {
  try {
    const txt = await r.text()
    if (!txt) return `Request failed with status ${r.status}`
    try {
      const j = JSON.parse(txt) as { detail?: unknown }
      if (typeof j.detail === 'string') return j.detail
      if (Array.isArray(j.detail)) {
        return j.detail
          .map((d) => (typeof d === 'object' && d !== null && 'msg' in d ? String((d as { msg: unknown }).msg) : JSON.stringify(d)))
          .join('; ')
      }
    } catch {
      return txt.slice(0, 400)
    }
    return txt.slice(0, 400)
  } catch {
    return `Request failed with status ${r.status}`
  }
}

export async function postPredict(req: PredictRequest): Promise<PredictResponse> {
  let r: Response
  try {
    r = await fetch(url('/predict'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
  } catch {
    throw new Error(
      'The prediction service is not responding. Start the FastAPI backend on port 8000, then try again.',
    )
  }
  if (!r.ok) throw new Error(await parseError(r))
  return (await r.json()) as PredictResponse
}

export async function postBatchPredict(reqs: PredictRequest[]): Promise<PredictResponse[]> {
  let r: Response
  try {
    r = await fetch(url('/batch-predict'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(reqs),
    })
  } catch {
    throw new Error(
      'The prediction service is not responding. Start the FastAPI backend on port 8000, then try again.',
    )
  }
  if (!r.ok) throw new Error(await parseError(r))
  return (await r.json()) as PredictResponse[]
}

export async function postExplain(req: PredictRequest): Promise<SHAPContribution[]> {
  const r = await fetch(url('/explain'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!r.ok) throw new Error(await parseError(r))
  const j = (await r.json()) as { contributions: SHAPContribution[] }
  return j.contributions
}

export interface OptionsResponse {
  makes: string[]
  models_by_make: Record<string, string[]>
  models_by_make_lower?: Record<string, string[]>
  vehicle_classes: string[]
  fuel_types: string[]
  transmissions: string[]
  road_types?: string[]
  traffic_conditions?: string[]
  regions?: string[]
  regulatory_cycles?: string[]
  data_sources?: string[]
}

export async function getOptions(): Promise<OptionsResponse> {
  const r = await fetch(url('/options'))
  if (!r.ok) throw new Error(await parseError(r))
  return (await r.json()) as OptionsResponse
}

export interface ModelInfo {
  model_version?: string
  test_R2?: number
  val_R2?: number
  india_R2?: number
  n_features?: number
  target?: string
  per_source?: Array<Record<string, string>>
  known_limitations?: string[]
}

export async function getModelInfo(): Promise<ModelInfo> {
  const r = await fetch(url('/model/info'))
  if (!r.ok) throw new Error(await parseError(r))
  return (await r.json()) as ModelInfo
}

export async function getHealth(): Promise<{ status: string }> {
  const r = await fetch(url('/health'))
  if (!r.ok) throw new Error(await parseError(r))
  return (await r.json()) as { status: string }
}
