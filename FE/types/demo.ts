export interface PredictionResponse {
  threshold: number;
  probability_map: string;
  mask: string;
  overlay: string;
  raw_prob_min: number;
  raw_prob_mean: number;
  raw_prob_max: number;
  pred_area_ratio: number;
}

export interface MetricCardProps {
  label: string;
  value: number | string;
  decimals?: number;
  unit?: string;
  description?: string;
}

export interface ResultPanelProps {
  title: string;
  src: string | null;
  placeholder?: string;
  badge?: string;
}
