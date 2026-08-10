import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import type { ModelMetric } from '../types'
import './ModelMetricsPanel.css'

interface Props {
  metrics: ModelMetric[]
}

export function ModelMetricsPanel({ metrics }: Props) {
  if (!metrics.length) return <div className="metrics-loading">Loading metrics…</div>

  return (
    <div className="metrics-layout">
      <div className="metrics-header">
        <h2>Model Performance — GE-79 Cohort (n=75, 5-fold CV)</h2>
        <p className="metrics-note">
          All three models trained on the same 14 FINAL_FEATURES. Class-weighted to account for imbalance (55 No Impairment / 20 Impaired).
        </p>
      </div>

      {/* Metric bar chart */}
      <MetricBarChart metrics={metrics} />

      {/* Confusion matrices */}
      <div className="confusion-row">
        {metrics.map(m => (
          <ConfusionMatrix key={m.model} metric={m} />
        ))}
      </div>

      {/* Table */}
      <div className="metrics-table-wrap">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
              <th>Macro F1</th>
              <th>Impaired Recall</th>
              <th>ROC-AUC</th>
              <th>PR-AUC</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map(m => (
              <tr key={m.model}>
                <td>
                  <span className="model-dot" style={{ background: m.color }} />
                  {m.model}
                </td>
                <td>{m.accuracy.toFixed(3)}</td>
                <td>{m.f1_macro.toFixed(3)}</td>
                <td className={m.recall_impaired >= 0.5 ? 'cell-good' : 'cell-warn'}>
                  {m.recall_impaired.toFixed(3)}
                </td>
                <td>{m.roc_auc.toFixed(3)}</td>
                <td>{m.pr_auc.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="metrics-footnote">
        <strong>Majority-class baseline:</strong> 0.733 accuracy (always predict No Impairment).
        Impaired Recall and Macro F1 are the primary success criteria because the dataset is class-imbalanced.
        A model that catches more impaired participants (higher Impaired Recall) is preferable even if accuracy drops.
      </div>
    </div>
  )
}

/* ── Grouped bar chart ───────────────────────────────────────────────── */
function MetricBarChart({ metrics }: { metrics: ModelMetric[] }) {
  const ref = useRef<SVGSVGElement>(null)

  const metricKeys: Array<keyof ModelMetric> = ['accuracy', 'f1_macro', 'recall_impaired', 'roc_auc']
  const metricLabels: Record<string, string> = {
    accuracy: 'Accuracy',
    f1_macro: 'Macro F1',
    recall_impaired: 'Impaired Recall',
    roc_auc: 'ROC-AUC',
  }

  useEffect(() => {
    if (!ref.current || !metrics.length) return
    const svg = d3.select(ref.current)
    svg.selectAll('*').remove()

    const W = ref.current.clientWidth || 700
    const H = 260
    const margin = { top: 20, right: 20, bottom: 50, left: 40 }
    const width = W - margin.left - margin.right
    const height = H - margin.top - margin.bottom

    const g = svg
      .attr('width', W)
      .attr('height', H)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const x0 = d3.scaleBand()
      .domain(metricKeys.map(k => metricLabels[k]))
      .range([0, width])
      .paddingInner(0.3)

    const x1 = d3.scaleBand()
      .domain(metrics.map(m => m.model))
      .range([0, x0.bandwidth()])
      .padding(0.08)

    const y = d3.scaleLinear().domain([0, 1]).range([height, 0])

    // Grid lines
    g.append('g')
      .attr('class', 'grid')
      .call(
        d3.axisLeft(y)
          .tickValues([0.25, 0.5, 0.75, 1.0])
          .tickSize(-width)
          .tickFormat(() => '')
      )
      .call(gg => {
        gg.select('.domain').remove()
        gg.selectAll('line').attr('stroke', '#1e2d45').attr('stroke-dasharray', '4,3')
      })

    // Bars
    metricKeys.forEach(mk => {
      const label = metricLabels[mk]
      const gMetric = g.append('g').attr('transform', `translate(${x0(label)},0)`)

      metrics.forEach(m => {
        const val = m[mk] as number
        gMetric
          .append('rect')
          .attr('x', x1(m.model) ?? 0)
          .attr('y', y(val))
          .attr('width', x1.bandwidth())
          .attr('height', height - y(val))
          .attr('fill', m.color)
          .attr('rx', 3)
          .attr('opacity', 0.85)

        gMetric
          .append('text')
          .attr('x', (x1(m.model) ?? 0) + x1.bandwidth() / 2)
          .attr('y', y(val) - 4)
          .attr('text-anchor', 'middle')
          .attr('font-size', 9)
          .attr('fill', '#94a3b8')
          .text(val.toFixed(2))
      })
    })

    // X axis
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x0))
      .call(gg => {
        gg.select('.domain').attr('stroke', '#1e2d45')
        gg.selectAll('text')
          .attr('fill', '#94a3b8')
          .attr('font-size', 12)
        gg.selectAll('line').attr('stroke', '#1e2d45')
      })

    // Y axis
    g.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .call(gg => {
        gg.select('.domain').remove()
        gg.selectAll('text').attr('fill', '#64748b').attr('font-size', 10)
        gg.selectAll('line').remove()
      })

    // Legend
    const legend = g.append('g').attr('transform', `translate(0,${height + 32})`)
    metrics.forEach((m, i) => {
      const lx = i * 170
      legend.append('rect').attr('x', lx).attr('y', 0).attr('width', 10).attr('height', 10).attr('fill', m.color).attr('rx', 2)
      legend.append('text').attr('x', lx + 14).attr('y', 9).attr('fill', '#94a3b8').attr('font-size', 11).text(m.model)
    })
  }, [metrics])

  return (
    <div className="chart-wrap">
      <svg ref={ref} style={{ width: '100%' }} />
    </div>
  )
}

/* ── Confusion matrix ────────────────────────────────────────────────── */
function ConfusionMatrix({ metric }: { metric: ModelMetric }) {
  const [[tp_n, fp_i], [fn_n, tp_i]] = metric.confusion_matrix
  const total = tp_n + fp_i + fn_n + tp_i

  const cells = [
    { label: 'True No Imp.', value: tp_n, type: 'good', row: 'No Imp.', col: 'No Imp.' },
    { label: 'False Alarm', value: fp_i, type: 'bad', row: 'No Imp.', col: 'Impaired' },
    { label: 'Missed Impaired', value: fn_n, type: 'bad', row: 'Impaired', col: 'No Imp.' },
    { label: 'True Impaired', value: tp_i, type: 'good', row: 'Impaired', col: 'Impaired' },
  ]

  return (
    <div className="confusion-card" style={{ '--model-color': metric.color } as React.CSSProperties}>
      <div className="confusion-title" style={{ color: metric.color }}>{metric.model}</div>
      <div className="confusion-grid">
        <div className="conf-axis-label col-label">Predicted →</div>
        <div className="conf-col-labels">
          <span>No Imp.</span>
          <span>Impaired</span>
        </div>
        <div className="conf-row-label-col">
          <div className="conf-row-axis">Actual ↓</div>
          <div className="conf-row-labels">
            <span>No Imp.</span>
            <span>Impaired</span>
          </div>
        </div>
        <div className="conf-cells">
          {cells.map(c => (
            <div key={c.label} className={`conf-cell conf-cell--${c.type}`}>
              <div className="conf-cell-value">{c.value}</div>
              <div className="conf-cell-pct">{Math.round((c.value / total) * 100)}%</div>
              <div className="conf-cell-label">{c.label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
