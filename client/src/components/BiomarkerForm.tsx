import { useState } from 'react'
import type { FeatureMeta, FeatureStats } from '../types'
import './BiomarkerForm.css'

const DOMAIN_ORDER = ['Glycemic', 'Cardiovascular', 'Inflammation', 'Cerebrovascular', 'Body Composition']

interface Props {
  features: FeatureMeta[]
  stats: Record<string, FeatureStats>
  values: Record<string, number | null>
  onChange: (key: string, val: number | null) => void
}

export function BiomarkerForm({ features, stats, values, onChange }: Props) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})

  const byDomain = DOMAIN_ORDER.map(domain => ({
    domain,
    features: features.filter(f => f.domain === domain),
  }))

  const toggle = (domain: string) =>
    setCollapsed(p => ({ ...p, [domain]: !p[domain] }))

  const DOMAIN_COLORS: Record<string, string> = {
    Glycemic: '#fbbf24',
    Cardiovascular: '#f87171',
    Inflammation: '#c084fc',
    Cerebrovascular: '#38bdf8',
    'Body Composition': '#34d399',
  }

  return (
    <div className="bio-form">
      <div className="bio-form-header">
        <h2 className="bio-form-title">Biomarker Profile</h2>
        <span className="bio-form-hint">Adjust sliders or type values</span>
      </div>

      {byDomain.map(({ domain, features: domainFeatures }) => {
        const isCollapsed = collapsed[domain]
        const color = DOMAIN_COLORS[domain] ?? '#64748b'
        return (
          <div key={domain} className="domain-group">
            <button
              className="domain-header"
              onClick={() => toggle(domain)}
              style={{ '--domain-color': color } as React.CSSProperties}
            >
              <span className="domain-dot" style={{ background: color }} />
              <span className="domain-name">{domain}</span>
              <span className="domain-count">{domainFeatures.length}</span>
              <span className="domain-chevron">{isCollapsed ? '▸' : '▾'}</span>
            </button>

            {!isCollapsed && (
              <div className="domain-features">
                {domainFeatures.map(f => (
                  <FeatureRow
                    key={f.key}
                    feature={f}
                    stat={stats[f.key]}
                    value={values[f.key] ?? null}
                    onChange={val => onChange(f.key, val)}
                  />
                ))}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

interface RowProps {
  feature: FeatureMeta
  stat: FeatureStats
  value: number | null
  onChange: (val: number | null) => void
}

function FeatureRow({ feature, stat, value, onChange }: RowProps) {
  const [showInfo, setShowInfo] = useState(false)
  const min = feature.typical_min
  const max = feature.typical_max
  const step = (max - min) > 5 ? 0.5 : 0.01
  const missing = value === null

  const pct = value !== null
    ? Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100))
    : 50

  // Highlight if value is in "at risk" zone (above reference high)
  const atRisk = value !== null && value > feature.reference_high

  return (
    <div className={`feature-row ${atRisk ? 'feature-row--risk' : ''}`}>
      <div className="feature-row-top">
        <div className="feature-label-group">
          <span className="feature-label">{feature.label}</span>
          <button
            className="info-btn"
            onClick={() => setShowInfo(p => !p)}
            title={feature.description}
          >
            ℹ
          </button>
        </div>
        <div className="feature-input-group">
          {feature.allow_missing && (
            <label className="missing-label">
              <input
                type="checkbox"
                checked={missing}
                onChange={e => onChange(e.target.checked ? null : (stat?.median ?? feature.typical_min))}
              />
              <span>Missing</span>
            </label>
          )}
          {!missing && (
            <input
              type="number"
              value={value ?? ''}
              min={min}
              max={max}
              step={step}
              onChange={e => {
                const v = parseFloat(e.target.value)
                onChange(isNaN(v) ? null : v)
              }}
            />
          )}
          {feature.unit && <span className="feature-unit">{feature.unit}</span>}
        </div>
      </div>

      {showInfo && (
        <p className="feature-info">{feature.description}</p>
      )}

      {!missing && (
        <div className="slider-wrap">
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value ?? stat?.median ?? min}
            onChange={e => onChange(parseFloat(e.target.value))}
            style={{ '--pct': `${pct}%` } as React.CSSProperties}
          />
          <div className="slider-refs">
            <span>{min}</span>
            <span className="ref-range">
              Normal: {feature.reference_low}–{feature.reference_high}
            </span>
            <span>{max}</span>
          </div>
        </div>
      )}

      {missing && (
        <p className="missing-note">Will be imputed with cohort median ({stat?.median?.toFixed(1) ?? '—'})</p>
      )}
    </div>
  )
}
