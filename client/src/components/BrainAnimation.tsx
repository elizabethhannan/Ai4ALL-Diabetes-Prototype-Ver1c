/**
 * GE-79 MCI Explorer — D3.js Brain Animation with Brain Shape
 * Ported from brain-animation-shaped.js to a React/TypeScript component.
 *
 * Features:
 * - Circles positioned to form realistic brain shape (cerebrum + cerebellum)
 * - D3 force simulation for organic clustering
 * - Canvas rendering (high performance)
 * - Mouse interaction via central attractor node
 */

import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import './BrainAnimation.css'

// ── Config ────────────────────────────────────────────────────────────────────
const CFG = {
  width: 800,
  height: 500,
  particleCount: 200,
  alphaTarget: 0.3,
  velocityDecay: 0.1,
  collideIterations: 3,
  chargeStrength: 280,   // positive here; attractor gets negated below
}

interface Node extends d3.SimulationNodeDatum {
  id: number
  r: number
  group: number   // 0 = attractor, 1 = cerebrum, 2 = cerebellum
}

// ── Particle generation ───────────────────────────────────────────────────────
function generateBrainParticles(count: number): Node[] {
  const particles: Node[] = []

  // Node 0: invisible central attractor for mouse control
  particles.push({ id: 0, r: 1, group: 0, x: 0, y: 0, vx: 0, vy: 0 })

  let idx = 1

  // CEREBRUM (~85 % of particles)
  const cerebrumCount = Math.floor(count * 0.85)
  for (let i = 0; i < cerebrumCount && idx < count; i++) {
    const angle = (i / cerebrumCount) * Math.PI * 2

    const noise1 = Math.sin(angle * 4) * 30
    const noise2 = Math.sin(angle * 8) * 15
    const noise3 = Math.cos(angle * 2) * 20
    const verticalTaper = Math.sin(angle * 3) * 10

    const radius = 110 + noise1 + noise2 + noise3 + verticalTaper
    const x = Math.cos(angle) * radius + (Math.random() - 0.5) * 20
    const y = Math.sin(angle) * radius * 0.8 - 40 + (Math.random() - 0.5) * 20

    const sv = Math.random()
    const r = sv < 0.2 ? 7 + Math.random() * 3
             : sv < 0.6 ? 4 + Math.random() * 2.5
             :             2 + Math.random() * 2

    particles.push({ id: idx++, r, group: 1, x, y, vx: 0, vy: 0 })
  }

  // CEREBELLUM (remaining particles — small lobe below)
  const cerebellumCount = count - idx
  for (let i = 0; i < cerebellumCount && idx < count; i++) {
    const angle = (i / cerebellumCount) * Math.PI * 2
    const wrinkles = Math.sin(angle * 10) * 5
    const radius = 45 + wrinkles
    const x = Math.cos(angle) * radius * 0.95 + (Math.random() - 0.5) * 12
    const y = Math.sin(angle) * radius * 0.9 + 120 + (Math.random() - 0.5) * 12
    const r = 3 + Math.random() * 2
    particles.push({ id: idx++, r, group: 2, x, y, vx: 0, vy: 0 })
  }

  return particles
}

// ── Component ─────────────────────────────────────────────────────────────────
export function BrainAnimation() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const { width, height } = CFG

    // Build canvas
    const canvas = document.createElement('canvas')
    canvas.width  = width
    canvas.height = height
    canvas.style.display = 'block'
    canvas.style.width   = '100%'
    canvas.style.height  = 'auto'
    // Insert canvas before the img overlay so the head renders on top
    container.insertBefore(canvas, container.firstChild)

    const ctx = canvas.getContext('2d')!
    const nodes = generateBrainParticles(CFG.particleCount)

    // Force simulation
    const simulation = d3.forceSimulation<Node>(nodes)
      .alphaTarget(CFG.alphaTarget)
      .velocityDecay(CFG.velocityDecay)
      .force('x', d3.forceX<Node>().strength(0.01))
      .force('y', d3.forceY<Node>().strength(0.01))
      .force('collide', d3.forceCollide<Node>().radius(d => d.r + 1.5).iterations(CFG.collideIterations))
      .force('charge', d3.forceManyBody<Node>().strength((_d, i) =>
        i === 0 ? -CFG.chargeStrength : 0
      ))
      .on('tick', ticked)

    // Mouse interaction
    function pointermoved(event: MouseEvent) {
      const rect = canvas.getBoundingClientRect()
      // Map CSS pixels → simulation coordinate space (origin at centre)
      const scaleX = width  / rect.width
      const scaleY = height / rect.height
      const x = (event.clientX - rect.left) * scaleX - width  / 2
      const y = (event.clientY - rect.top)  * scaleY - height / 2

      const padding = 80
      if (nodes[0]) {
        nodes[0].fx = Math.max(-width  / 2 + padding, Math.min(width  / 2 - padding, x))
        nodes[0].fy = Math.max(-height / 2 + padding, Math.min(height / 2 - padding, y))
      }
    }

    function pointerleave() {
      if (nodes[0]) { nodes[0].fx = null; nodes[0].fy = null }
    }

    canvas.addEventListener('mousemove', pointermoved)
    canvas.addEventListener('mouseleave', pointerleave)
    canvas.addEventListener('touchmove', e => e.preventDefault(), { passive: false })

    function ticked() {
      // Hard boundary clamp
      const maxR = Math.max(...nodes.slice(1).map(d => d.r)) + 5
      const bx = width  / 2 - maxR
      const by = height / 2 - maxR

      for (let i = 1; i < nodes.length; i++) {
        const d = nodes[i]
        if ((d.x ?? 0) - d.r < -bx) d.x = -bx + d.r
        if ((d.x ?? 0) + d.r >  bx) d.x =  bx - d.r
        if ((d.y ?? 0) - d.r < -by) d.y = -by + d.r
        if ((d.y ?? 0) + d.r >  by) d.y =  by - d.r
      }

      ctx.clearRect(0, 0, width, height)
      ctx.save()
      ctx.translate(width / 2, height / 2)

      for (let i = 1; i < nodes.length; i++) {
        const d = nodes[i]
        ctx.beginPath()
        ctx.moveTo((d.x ?? 0) + d.r, d.y ?? 0)
        ctx.arc(d.x ?? 0, d.y ?? 0, d.r, 0, Math.PI * 2)

        ctx.fillStyle = d.r > 6 ? '#d8d8d8'
                      : d.r > 3.5 ? '#b0b0b0'
                      :              '#888888'
        ctx.globalAlpha = 0.75
        ctx.fill()

        ctx.strokeStyle = 'rgba(255,255,255,0.08)'
        ctx.lineWidth = 0.5
        ctx.stroke()
      }

      ctx.restore()
      ctx.globalAlpha = 1.0
    }

    return () => {
      simulation.stop()
      canvas.removeEventListener('mousemove', pointermoved)
      canvas.removeEventListener('mouseleave', pointerleave)
      container.removeChild(canvas)
    }
  }, [])

  return (
    <div className="brain-anim-wrap">
      <div className="brain-anim-header">
        <span className="brain-anim-title">Abstract Neural Pattern Visualizer</span>
      </div>
      <div ref={containerRef} className="brain-anim-canvas-container">
        <img
          src="/brain-head.png"
          alt=""
          className="brain-anim-overlay"
        />
      </div>
    </div>
  )
}
