import { useEffect, useState } from 'react'
import type { FeaturesResponse, PredictResponse, ModelMetric } from './types'
import { BiomarkerForm } from './components/BiomarkerForm'
import { PredictionPanel } from './components/PredictionPanel'
import { ModelMetricsPanel } from './components/ModelMetricsPanel'
import { RadarChart } from './components/RadarChart'
import './index.css'
import './App.css'

export default function App() {
  const [featuresData, setFeaturesData] = useState<FeaturesResponse | null>(null)
  const [metrics, setMetrics] = useState<ModelMetric[]>([])
  const [values, setValues] = useState<Record<string, number | null>>({})
  const [prediction, setPrediction] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'predict' | 'metrics'>('predict')

  useEffect(() => {
    fetch('/api/features')
      .then(r => r.json())
      .then((data: FeaturesResponse) => {
        setFeaturesData(data)
        // seed inputs with cohort medians
        const defaults: Record<string, number | null> = {}
        data.features.forEach(f => {
          defaults[f.key] = data.stats[f.key]?.median ?? null
        })
        setValues(defaults)
      })
      .catch(console.error)

    fetch('/api/model-metrics')
      .then(r => r.json())
      .then(d => setMetrics(d.models))
      .catch(console.error)
  }, [])

  const handlePredict = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: values }),
      })
      const data: PredictResponse = await res.json()
      setPrediction(data)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    if (!featuresData) return
    const defaults: Record<string, number | null> = {}
    featuresData.features.forEach(f => {
      defaults[f.key] = featuresData.stats[f.key]?.median ?? null
    })
    setValues(defaults)
    setPrediction(null)
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-inner">
          <div className="header-brand">
            <span className="header-tag">GE-79 · AI4ALL 2026 · Group 6C</span>
            <h1 className="header-title">Cognitive-Status Prototype</h1>
            <p className="header-subtitle">
              Interactive biomarker explorer — Logistic Regression · Decision Tree · Random Forest
            </p>
          </div>
          <div className="header-badge">
            <span className="badge-dot" />
            Research Only · Not Diagnostic
          </div>
        </div>
        <nav className="tabs">
          <button
            className={`tab ${activeTab === 'predict' ? 'tab-active' : ''}`}
            onClick={() => setActiveTab('predict')}
          >
            Predict
          </button>
          <button
            className={`tab ${activeTab === 'metrics' ? 'tab-active' : ''}`}
            onClick={() => setActiveTab('metrics')}
          >
            Model Metrics
          </button>
        </nav>
      </header>

      {/* Main content */}
      <main className="app-main">
        {activeTab === 'predict' && featuresData && (
          <div className="predict-layout">
            {/* Left: form */}
            <section className="form-section">
              <BiomarkerForm
                features={featuresData.features}
                stats={featuresData.stats}
                values={values}
                onChange={(key, val) => setValues(prev => ({ ...prev, [key]: val }))}
              />
              <div className="form-actions">
                <button className="btn-reset" onClick={handleReset}>Reset to Medians</button>
                <button className="btn-predict" onClick={handlePredict} disabled={loading}>
                  {loading ? 'Predicting…' : 'Run Prediction'}
                </button>
              </div>
            </section>

            {/* Right: results */}
            <section className="results-section">
              <RadarChart
                features={featuresData.features}
                stats={featuresData.stats}
                importance={featuresData.feature_importance}
                values={values}
              />
              {prediction && (
                <PredictionPanel prediction={prediction} />
              )}
              {!prediction && (
                <div className="placeholder-card">
                  <div className="placeholder-icon">⚡</div>
                  <p className="placeholder-text">
                    Adjust the biomarkers on the left, then click <strong>Run Prediction</strong> to see outputs from all three classifiers.
                  </p>
                </div>
              )}
            </section>
          </div>
        )}

        {activeTab === 'metrics' && (
          <ModelMetricsPanel metrics={metrics} />
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          GE-79 · CDED 1.0.1 · n=75 · 5-fold CV · AI4ALL Ignite 2026 · Group 6C ·
          Elizabeth Hannan &amp; Agastyya Kola ·
          Research prototype — not a screening or diagnostic tool.
        </p>
      </footer>
    </div>
  )
}
