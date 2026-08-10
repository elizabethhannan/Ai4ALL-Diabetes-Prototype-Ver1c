import { useEffect, useState } from 'react'
import type { FeaturesResponse, PredictResponse, ModelMetric } from './types'
import { BiomarkerForm } from './components/BiomarkerForm'
import { PredictionPanel } from './components/PredictionPanel'
import { ModelMetricsPanel } from './components/ModelMetricsPanel'
import { RadarChart } from './components/RadarChart'
import { BrainAnimation } from './components/BrainAnimation'
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
          {/* Left — AI4ALL branding */}
          <div className="header-left">
            <img src="/ai4all-logo.png" className="header-brain-icon" alt="AI4ALL logo" />
            <div className="header-brand-text">
              <span className="header-brand-name">AI4ALL</span>
              <span className="header-brand-sub">Summer Cohort 2026</span>
            </div>
          </div>

          {/* Center — title */}
          <div className="header-center">
            <div className="header-title-row">
              <h1 className="header-title">GE-79 MCI Explorer</h1>
              <span className="header-version-badge">ProtoApp Version 8</span>
              <div className="header-badge">
                <span className="badge-dot" />
                Research Only · Not Diagnostic
              </div>
            </div>
            <p className="header-subtitle">
              Can supervised machine learning classify Mild Cognitive Impairment in older adults with Type 2 Diabetes?
            </p>
          </div>

          {/* Right — authors */}
          <div className="header-right">
            <div className="header-author">
              <div className="author-qr">
                <svg viewBox="0 0 32 32" width="40" height="40">
                  <rect width="32" height="32" fill="#111" rx="3"/>
                  <rect x="2" y="2" width="12" height="12" fill="none" stroke="#34d399" strokeWidth="1.5"/>
                  <rect x="5" y="5" width="6" height="6" fill="#34d399"/>
                  <rect x="18" y="2" width="12" height="12" fill="none" stroke="#34d399" strokeWidth="1.5"/>
                  <rect x="21" y="5" width="6" height="6" fill="#34d399"/>
                  <rect x="2" y="18" width="12" height="12" fill="none" stroke="#34d399" strokeWidth="1.5"/>
                  <rect x="5" y="21" width="6" height="6" fill="#34d399"/>
                  <rect x="18" y="18" width="4" height="4" fill="#34d399"/>
                  <rect x="24" y="18" width="4" height="4" fill="#34d399"/>
                  <rect x="18" y="24" width="4" height="4" fill="#34d399"/>
                  <rect x="24" y="24" width="4" height="4" fill="#34d399"/>
                </svg>
              </div>
              <div className="author-info">
                <span className="author-name">Elizabeth H.</span>
                <span className="author-link">Author · LinkedIn</span>
              </div>
            </div>
            <div className="header-author">
              <div className="author-qr">
                <svg viewBox="0 0 32 32" width="40" height="40">
                  <rect width="32" height="32" fill="#111" rx="3"/>
                  <rect x="2" y="2" width="12" height="12" fill="none" stroke="#34d399" strokeWidth="1.5"/>
                  <rect x="5" y="5" width="6" height="6" fill="#34d399"/>
                  <rect x="18" y="2" width="12" height="12" fill="none" stroke="#34d399" strokeWidth="1.5"/>
                  <rect x="21" y="5" width="6" height="6" fill="#34d399"/>
                  <rect x="2" y="18" width="12" height="12" fill="none" stroke="#34d399" strokeWidth="1.5"/>
                  <rect x="5" y="21" width="6" height="6" fill="#34d399"/>
                  <rect x="22" y="18" width="4" height="4" fill="#34d399"/>
                  <rect x="18" y="24" width="4" height="4" fill="#34d399"/>
                  <rect x="26" y="22" width="4" height="4" fill="#34d399"/>
                  <rect x="20" y="26" width="4" height="4" fill="#34d399"/>
                </svg>
              </div>
              <div className="author-info">
                <span className="author-name">Agastyya K.</span>
                <span className="author-link">Author · LinkedIn</span>
              </div>
            </div>
          </div>
        </div>

        {/* Accent bar */}
        <div className="header-accent-bar" />

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
              <BrainAnimation />
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
        <div className="footer-inner">
          <div className="footer-col">
            <svg className="footer-icon" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
            <div>
              <span className="footer-col-title">NOT A DIAGNOSTIC TOOL.</span>
              <p className="footer-col-body">For research and educational purposes only. Do not use for clinical decision-making.</p>
            </div>
          </div>
          <div className="footer-divider" />
          <div className="footer-col">
            <svg className="footer-icon" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
              <path d="M6 12v5c3 3 9 3 12 0v-5"/>
            </svg>
            <div>
              <span className="footer-col-title">AI4ALL SUMMER COHORT 2026</span>
              <p className="footer-col-body">Building AI for equity. Advancing diversity in science and technology.</p>
            </div>
          </div>
          <div className="footer-divider" />
          <div className="footer-col">
            <svg className="footer-icon" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
            </svg>
            <div>
              <span className="footer-col-title">SPECIAL ACKNOWLEDGMENT</span>
              <p className="footer-col-body">We gratefully acknowledge Professor Joyce D. Williams for her guidance, mentorship, and unwavering support.</p>
            </div>
          </div>
          <div className="footer-divider" />
          <div className="footer-col">
            <svg className="footer-icon" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
            </svg>
            <div>
              <span className="footer-col-title">DATASET CITATION (APA 7)</span>
              <p className="footer-col-body">Novak, V., &amp; Quispe, R. (2022). <em>Cerebromicrovascular disease in elderly with diabetes</em> [Version 1.0.1] [Data set]. PhysioNet. https://doi.org/10.13026/00bm-2x81</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
