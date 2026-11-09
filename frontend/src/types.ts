export interface PredictRequest {
  year: number
  make: string
  model: string
  vehicle_class: string
  engine_size_l: number
  cylinders: number
  transmission: string
  fuel_type: string
  // CatBoost advanced optional (median-imputed if omitted)
  power_hp?: number
  mileage_km?: number
  speed_kmh?: number
  acceleration_ms2?: number
  temperature_c?: number
  humidity_pct?: number
  wind_speed_ms?: number
  air_pressure_hpa?: number
  vehicle_age?: number
  road_type?: string
  traffic_condition?: string
  region?: string
  regulatory_cycle?: string
  data_source?: string
  drive?: string
  is_india?: number
}

export interface Uncertainty {
  fuel_city_std: number
  fuel_hwy_std: number
  fuel_comb_std: number
  co2_std: number
}

export interface PredictResponse {
  fuel_city_l_100km: number
  fuel_hwy_l_100km: number
  fuel_comb_l_100km: number
  fuel_comb_mpg: number
  co2_g_km: number
  co2_g_per_km?: number
  uncertainty: Uncertainty
  annual_cost_usd: number
  eco_score: number
  unseen_vehicle: boolean
  model_version: string
  inference_ms: number
  is_india?: boolean
  note?: string
}

export interface SHAPContribution {
  feature: string
  contribution: number
}
