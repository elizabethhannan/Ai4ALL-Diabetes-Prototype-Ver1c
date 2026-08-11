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
      <div className="cm-key-banner">
        <strong>How to read each confusion matrix:</strong> rows are the participant's actual
        cognitive-status class; columns are the model's predicted class.{' '}
        <span className="cm-key-good">Green cells</span> are correct classifications;{' '}
        <span className="cm-key-bad">coral cells</span> are errors. This shared key applies
        to all three models. <em>Source: GE-79 Streamlit ML Visualizations App.</em>
      </div>
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
const MODEL_META: Record<string, { rank: string; subtitle: string; takeaway: string }> = {
  'Logistic Regression': {
    rank: 'MODEL 1 · LOGISTIC REGRESSION',
    subtitle: 'Interpretable baseline',
    takeaway: 'The baseline found 10 of 20 impaired participants, but created 24 false alarms.',
  },
  'Decision Tree': {
    rank: 'MODEL 2 · DECISION TREE',
    subtitle: 'Interpretable branching classifier',
    takeaway: 'It identified the largest share of impaired participants: 11 of 20.',
  },
  'Random Forest': {
    rank: 'MODEL 3 · RANDOM FOREST',
    subtitle: 'Ensemble classifier',
    takeaway: 'It achieved the highest overall accuracy, but missed 15 of 20 impaired participants.',
  },
}

function ConfusionMatrix({ metric }: { metric: ModelMetric }) {
  const [[tn, fp], [fn, tp]] = metric.confusion_matrix
  const meta = MODEL_META[metric.model] ?? {
    rank: metric.model.toUpperCase(),
    subtitle: metric.model,
    takeaway: '',
  }
  const accuracy    = Math.round(metric.accuracy * 100)
  const recall      = Math.round(metric.recall_impaired * 100)

  return (
    <div className="confusion-card" style={{ '--model-color': metric.color } as React.CSSProperties}>
      {/* Header */}
      <div className="cm-rank" style={{ color: metric.color }}>{meta.rank}</div>
      <div className="cm-subtitle">{meta.subtitle}</div>

      {/* Legend */}
      <div className="cm-legend">
        <span className="cm-legend-dot cm-legend-dot--good" />correct prediction
        <span className="cm-legend-dot cm-legend-dot--bad" style={{ marginLeft: 12 }} />model error
      </div>

      {/* Matrix table */}
      <div className="cm-table-wrap">
        <table className="cm-table">
          <thead>
            <tr>
              <th className="cm-th-blank" />
              <th className="cm-th">Predicted:<br />No impairment</th>
              <th className="cm-th">Predicted:<br />Impaired</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="cm-row-label">Actual:<br />No impairment</td>
              <td className="cm-cell cm-cell--good">{tn}</td>
              <td className="cm-cell cm-cell--bad">{fp}</td>
            </tr>
            <tr>
              <td className="cm-row-label">Actual:<br />Impaired</td>
              <td className="cm-cell cm-cell--bad">{fn}</td>
              <td className="cm-cell cm-cell--good">{tp}</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Stats */}
      <div className="cm-stats">
        <div className="cm-stat">
          <span className="cm-stat-val">{accuracy}%</span>
          <span className="cm-stat-lbl">accuracy</span>
        </div>
        <div className="cm-stat">
          <span className="cm-stat-val">{recall}%</span>
          <span className="cm-stat-lbl">impaired recall</span>
        </div>
      </div>

      {/* Takeaway */}
      {meta.takeaway && (
        <div className="cm-takeaway">
          <strong>Takeaway:</strong> {meta.takeaway}
        </div>
      )}
    </div>
  )
}
