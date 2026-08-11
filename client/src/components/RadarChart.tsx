import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import type { FeatureMeta, FeatureStats } from '../types'
import './RadarChart.css'

interface Props {
  features: FeatureMeta[]
  stats: Record<string, FeatureStats>
  importance: Record<string, number>
  values: Record<string, number | null>
}

const DOMAIN_COLORS: Record<string, string> = {
  Glycemic:           '#fbbf24',
  Cardiovascular:     '#f87171',
  Inflammation:       '#c084fc',
  Cerebrovascular:    '#38bdf8',
  'Body Composition': '#34d399',
}

export function RadarChart({ features, stats, importance, values }: Props) {
  const ref = useRef<SVGSVGElement>(null)

  // Use top 8 features by importance for the radar
  const radarFeatures = [...features]
    .sort((a, b) => (importance[b.key] ?? 0) - (importance[a.key] ?? 0))
    .slice(0, 8)

  useEffect(() => {
    if (!ref.current) return
    const svg = d3.select(ref.current)
    svg.selectAll('*').remove()

    const W = ref.current.clientWidth || 400
    const H = W   // square — chart fills the card
    const cx = W / 2
    const cy = H / 2
    const R = Math.min(cx, cy) - 90  // smaller chart → more room for wrapped labels

    svg.attr('width', W).attr('height', H)

    const N = radarFeatures.length
    const angleStep = (2 * Math.PI) / N

    const angle = (i: number) => i * angleStep - Math.PI / 2

    // Normalize each feature value to [0,1] within its typical range
    const normalize = (f: FeatureMeta, val: number | null): number => {
      if (val === null) return 0.5
      const range = f.typical_max - f.typical_min
      if (range === 0) return 0.5
      return Math.max(0, Math.min(1, (val - f.typical_min) / range))
    }

    // Cohort median profile (reference)
    const medianVals = radarFeatures.map(f =>
      normalize(f, stats[f.key]?.median ?? null)
    )

    // User-input profile
    const userVals = radarFeatures.map(f =>
      normalize(f, values[f.key] ?? null)
    )

    const toXY = (r: number, i: number): { x: number; y: number } => ({
      x: cx + r * Math.cos(angle(i)),
      y: cy + r * Math.sin(angle(i)),
    });

    // Grid circles
    const gridLevels: number[] = [0.25, 0.5, 0.75, 1.0];
    gridLevels.forEach(t => {
      svg.append('circle')
        .attr('cx', cx).attr('cy', cy)
        .attr('r', R * t)
        .attr('fill', 'none')
        .attr('stroke', '#1e2d45')
        .attr('stroke-dasharray', t < 1 ? '4,3' : 'none')
        .attr('stroke-width', t === 1 ? 1.5 : 1)
    })

    // Spokes
    radarFeatures.forEach((_, i) => {
      const pt = toXY(R, i)
      svg.append('line')
        .attr('x1', cx).attr('y1', cy)
        .attr('x2', pt.x).attr('y2', pt.y)
        .attr('stroke', '#1e2d45')
        .attr('stroke-width', 1)
    })

    // Polygon helper
    const polyPath = (vals: number[]) =>
      vals.map((v, i) => {
        const pt = toXY(R * v, i)
        return `${i === 0 ? 'M' : 'L'}${pt.x},${pt.y}`
      }).join(' ') + 'Z'

    // Median polygon (reference)
    svg.append('path')
      .attr('d', polyPath(medianVals))
      .attr('fill', 'rgba(100,116,139,0.10)')
      .attr('stroke', '#475569')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3')

    // User polygon
    svg.append('path')
      .attr('d', polyPath(userVals))
      .attr('fill', 'rgba(56,189,248,0.15)')
      .attr('stroke', '#38bdf8')
      .attr('stroke-width', 2)

    // Dots on user polygon — coloured by domain
    radarFeatures.forEach((f, i) => {
      const v = userVals[i]
      const pt = toXY(R * v, i)
      const domainColor = DOMAIN_COLORS[f.domain] ?? '#38bdf8'
      svg.append('circle')
        .attr('cx', pt.x).attr('cy', pt.y).attr('r', 6)
        .attr('fill', domainColor)
        .attr('stroke', '#0a0f1e')
        .attr('stroke-width', 1.5)
    })

    // Labels
    radarFeatures.forEach((f, i) => {
      const pt = toXY(R + 20, i)
      const shortLabel = f.label.replace(/\(.*?\)/g, '').trim()
      const words = shortLabel.split(' ')   // always wrap every word
      const anchor = pt.x < cx - 5 ? 'end' : pt.x > cx + 5 ? 'start' : 'middle'
      const txt = svg.append('text')
        .attr('x', pt.x)
        .attr('y', pt.y)
        .attr('text-anchor', anchor)
        .attr('dominant-baseline', 'central')
        .attr('font-size', 11)
        .attr('fill', '#ffffff')

      words.forEach((w, wi) => {
        txt.append('tspan')
          .attr('x', pt.x)
          .attr('dy', wi === 0 ? (words.length > 1 ? `-${(words.length - 1) * 0.55}em` : '0') : '1.1em')
          .text(w)
      })
    })

  }, [radarFeatures, stats, values, importance])

  return (
    <div className="radar-wrap">
      <div className="radar-header">
        <span className="radar-title">Biomarker Profile (top 8 by importance)</span>
        <div className="radar-legend">
          <span className="legend-item">
            <span className="legend-line legend-line--user" />
            Your Profile
          </span>
          <span className="legend-item">
            <span className="legend-line legend-line--median" />
            Cohort Median
          </span>
        </div>
      </div>
      <svg ref={ref} style={{ width: '100%', overflow: 'visible' }} />
    </div>
  )
}
