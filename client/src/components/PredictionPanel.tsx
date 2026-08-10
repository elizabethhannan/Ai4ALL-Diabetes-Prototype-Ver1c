import type { PredictResponse } from '../types'
import './PredictionPanel.css'

interface Props {
  prediction: PredictResponse
}

const MODEL_ORDER = ['Logistic Regression', 'Decision Tree', 'Random Forest']
const MODEL_COLORS: Record<string, string> = {
  'Logistic Regression': '#4ECDC4',
  'Decision Tree': '#45B7D1',
  'Random Forest': '#96CEB4',
}
const MODEL_SUBTITLES: Record<string, string> = {
  'Logistic Regression': 'Baseline · most explainable',
  'Decision Tree': 'Interpretable · decision rules',
  'Random Forest': 'Ensemble · highest accuracy',
}

export function PredictionPanel({ prediction }: Props) {
  const { predictions, disclaimer } = prediction

  // Consensus: how many models agree on each class
  const impaired = MODEL_ORDER.filter(m => predictions[m]?.prediction === 1).length
  const noImpairment = MODEL_ORDER.length - impaired
  const consensusLabel = impaired > noImpairment ? 'Impaired' : 'No Impairment'
  const consensusStrength = Math.max(impaired, noImpairment)

  return (
    <div className="pred-panel">
      {/* Consensus banner */}
      <div className={`consensus ${impaired > noImpairment ? 'consensus--impaired' : 'consensus--normal'}`}>
        <div className="consensus-label">Model Consensus</div>
        <div className="consensus-center">
          <div className="consensus-result">
            <span className="consensus-icon">
              {impaired > noImpairment ? '⚠' : '✓'}
            </span>
            <span className="consensus-text">{consensusLabel}</span>
          </div>
          <div className="consensus-definition">
            {impaired > noImpairment
              ? 'Mild or moderately impaired based on biomarker profile'
              : 'Not mild or moderately impaired based on biomarker profile'}
          </div>
        </div>
        <div className="consensus-votes">
          {consensusStrength} of {MODEL_ORDER.length} models agree
        </div>
      </div>

      {/* Per-model results */}
      <div className="pred-cards">
        {MODEL_ORDER.map(modelName => {
          const result = predictions[modelName]
          if (!result) return null
          const color = MODEL_COLORS[modelName]
          const isImpaired = result.prediction === 1
          const probPct = Math.round(result.probability_impaired * 100)

          return (
            <div
              key={modelName}
              className={`pred-card ${isImpaired ? 'pred-card--impaired' : ''}`}
              style={{ '--model-color': color } as React.CSSProperties}
            >
              <div className="pred-card-header">
                <div>
                  <div className="pred-model-name" style={{ color }}>
                    {modelName}
                  </div>
                  <div className="pred-model-subtitle">{MODEL_SUBTITLES[modelName]}</div>
                </div>
                <div className={`pred-label-badge ${isImpaired ? 'badge--impaired' : 'badge--normal'}`}>
                  {result.label}
                </div>
              </div>

              {/* Probability bar */}
              <div className="prob-bars">
                <div className="prob-row">
                  <span className="prob-label">No Impairment</span>
                  <div className="prob-bar-wrap">
                    <div
                      className="prob-bar prob-bar--normal"
                      style={{ width: `${Math.round(result.probability_no_impairment * 100)}%` }}
                    />
                  </div>
                  <span className="prob-pct">
                    {Math.round(result.probability_no_impairment * 100)}%
                  </span>
                </div>
                <div className="prob-row">
                  <span className="prob-label">Impaired</span>
                  <div className="prob-bar-wrap">
                    <div
                      className="prob-bar prob-bar--impaired"
                      style={{ width: `${probPct}%` }}
                    />
                  </div>
                  <span className="prob-pct">{probPct}%</span>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Disclaimer */}
      <div className="pred-disclaimer">{disclaimer}</div>
    </div>
  )
}
