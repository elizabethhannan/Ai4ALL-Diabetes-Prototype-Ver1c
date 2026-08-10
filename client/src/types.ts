export interface FeatureMeta {
  key: string;
  label: string;
  domain: string;
  description: string;
  unit: string;
  typical_min: number;
  typical_max: number;
  reference_low: number;
  reference_high: number;
  allow_missing?: boolean;
}

export interface FeatureStats {
  mean: number;
  median: number;
  min: number;
  max: number;
  p25: number;
  p75: number;
  missing_count: number;
}

export interface FeaturesResponse {
  features: FeatureMeta[];
  stats: Record<string, FeatureStats>;
  feature_importance: Record<string, number>;
  n_samples: number;
}

export interface ModelPrediction {
  prediction: 0 | 1;
  label: string;
  probability_no_impairment: number;
  probability_impaired: number;
}

export interface PredictResponse {
  predictions: Record<string, ModelPrediction>;
  input_used: Record<string, number | null>;
  disclaimer: string;
}

export interface ModelMetric {
  model: string;
  accuracy: number;
  f1_macro: number;
  recall_impaired: number;
  roc_auc: number;
  pr_auc: number;
  confusion_matrix: [[number, number], [number, number]];
  color: string;
}
