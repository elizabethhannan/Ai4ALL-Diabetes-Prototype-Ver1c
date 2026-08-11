import { useState } from 'react'
import type { FeatureMeta, FeatureStats } from '../types'
import './BiomarkerForm.css'

const DOMAIN_ORDER = ['Glycemic', 'Cardiovascular', 'Inflammation', 'Cerebrovascular', 'Body Composition']

const CLINICAL_ROLES: Record<string, string> = {
  fasting_glucose_mg_dl:                  'Strongest single predictor; glucose dysregulation drives cognitive decline',
  glucose_mg_dl:                          'Confirms glycemic signal; acute glucose level independent of fasting',
  global_vasoreactivity:                  'Measures cerebral endothelial response; core CDED marker',
  daytime_sbp:                            'Elevated BP during activity; vascular stress indicator',
  wmh_registered:                         'MRI marker of diabetic small-vessel disease',
  perfusion_whole_brain_baseline_whole:   'Cerebral blood flow; low perfusion correlates with impairment',
  svcam_ng_ml:                            'Endothelial dysfunction marker; systemic inflammation',
  ldl_calc_mg_dl:                         'Lipid burden; atherosclerotic vascular disease',
  nighttime_sbp:                          'Non-dipping nocturnal BP; cerebrovascular risk',
  mass_kg:                                'Metabolic burden; insulin resistance correlate',
  perfusion_lepto_pca_baseline_whole:     'Posterior cerebral artery perfusion; region-specific',
  wmh_registered_masked:                  'Independent WMH replication',
  hba1c_percent:                          'Long-term glycemic control',
  diabetes_duration:                      'Cumulative vascular exposure',
}

interface Props {
  features: FeatureMeta[]
  stats: Record<string, FeatureStats>
  values: Record<string, number | null>
  onChange: (key: string, val: number | null) => void
}

export function BiomarkerForm({ features, stats, values, onChange }: Props) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})
  const [filterDomain, setFilterDomain] = useState<string>('All')

  const byDomain = DOMAIN_ORDER.map(domain => ({
    domain,
    features: features.filter(f => f.domain === domain),
  }))

  const visibleDomains = filterDomain === 'All'
    ? byDomain
    : byDomain.filter(d => d.domain === filterDomain)

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
        <h2 className="bio-form-title">Biomarker Feature Profile</h2>
        <span className="bio-form-hint">Adjust sliders or type values</span>
      </div>

      <div className="bio-filter-bar">
        <select
          className="bio-filter-select"
          value={filterDomain}
          onChange={e => setFilterDomain(e.target.value)}
        >
          <option value="All">All Categories</option>
          {DOMAIN_ORDER.map(d => <option key={d} value={d}>{d}</option>)}
        </select>
      </div>

      {visibleDomains.map(({ domain, features: domainFeatures }) => {
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
          <span className="feature-label-wrap">
            <span className="feature-label">{feature.label}</span>
            {CLINICAL_ROLES[feature.key] && (
              <span className="feature-tooltip">{CLINICAL_ROLES[feature.key]}</span>
            )}
          </span>
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
          {/* Normal range indicator bar */}
          <div className="normal-range-track">
            <div
              className="normal-range-fill"
              style={{
                left: `${Math.max(0, ((feature.reference_low - min) / (max - min)) * 100)}%`,
                width: `${Math.min(100, ((feature.reference_high - feature.reference_low) / (max - min)) * 100)}%`,
              }}
            />
          </div>
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
