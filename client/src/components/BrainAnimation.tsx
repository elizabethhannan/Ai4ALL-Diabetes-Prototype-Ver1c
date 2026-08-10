import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import './BrainAnimation.css'

interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  originalX: number
  originalY: number
  radius: number
}

const CONFIG = {
  particleCount: 280,
  particleRadius: 3.5,
  particleRadiusVariance: 2.5,
  particleColor: '#34d399',
  particleOpacity: 0.85,
  damping: 0.92,
  repelForce: 0.6,
  attractForce: 0.012,
  repelRadius: 120,
}

function generateBrainShape(centerX: number, centerY: number, w: number, h: number): { x: number; y: number }[] {
  const points: { x: number; y: number }[] = []
  const scale = Math.min(w, h) / 340
  for (let i = 0; i < CONFIG.particleCount; i++) {
    const angle = (i / CONFIG.particleCount) * Math.PI * 2

    // Two-lobe brain shape with convolutions (v2)
    const lobeInfluence = Math.cos(angle * 2)
    const convolution = Math.sin(angle * 6) * 8
    const heightVariation = Math.sin(angle * 3) * 15
    const baseRadius = 85 + convolution + (lobeInfluence > 0 ? 5 : 0)
    const radius = (baseRadius + heightVariation) * scale

    const x = centerX + Math.cos(angle) * radius
    const y = centerY + Math.sin(angle) * radius * 0.75

    points.push({
      x: x + (Math.random() - 0.5) * 15,
      y: y + (Math.random() - 0.5) * 15,
    })
  }
  return points
}

export function BrainAnimation() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const W = container.clientWidth || 600
    const H = 200
    const cx = W / 2
    const cy = H / 2

    let mouseX = cx
    let mouseY = cy
    let animId: number

    const brainPoints = generateBrainShape(cx, cy, W, H)
    const particles: Particle[] = brainPoints.map(p => ({
      x: p.x,
      y: p.y,
      vx: (Math.random() - 0.5) * 2,
      vy: (Math.random() - 0.5) * 2,
      originalX: p.x,
      originalY: p.y,
      radius: CONFIG.particleRadius + (Math.random() - 0.5) * CONFIG.particleRadiusVariance,
    }))

    const svg = d3.select(container)
      .append('svg')
      .attr('width', W)
      .attr('height', H)
      .style('background', 'transparent')
      .style('cursor', 'crosshair')
      .style('display', 'block')

    const circles = svg.selectAll<SVGCircleElement, Particle>('circle')
      .data(particles)
      .enter()
      .append('circle')
      .attr('r', d => d.radius)
      .attr('fill', CONFIG.particleColor)
      .attr('opacity', CONFIG.particleOpacity)

    svg.on('mousemove', function (event: MouseEvent) {
      const [x, y] = d3.pointer(event)
      mouseX = x
      mouseY = y
    })
    svg.on('mouseleave', function () {
      mouseX = cx
      mouseY = cy
    })

    function tick() {
      particles.forEach(p => {
        const dx = mouseX - p.x
        const dy = mouseY - p.y
        const dist = Math.sqrt(dx * dx + dy * dy)

        if (dist < CONFIG.repelRadius && dist > 0) {
          const angle = Math.atan2(dy, dx)
          const force = CONFIG.repelForce * (1 - dist / CONFIG.repelRadius)
          p.vx -= Math.cos(angle) * force
          p.vy -= Math.sin(angle) * force
        }

        p.vx += (p.originalX - p.x) * CONFIG.attractForce
        p.vy += (p.originalY - p.y) * CONFIG.attractForce
        p.vx *= CONFIG.damping
        p.vy *= CONFIG.damping
        p.x += p.vx
        p.y += p.vy

        if (p.x < 0) { p.x = 0; p.vx *= -0.5 }
        if (p.x > W) { p.x = W; p.vx *= -0.5 }
        if (p.y < 0) { p.y = 0; p.vy *= -0.5 }
        if (p.y > H) { p.y = H; p.vy *= -0.5 }
      })

      circles.attr('cx', d => d.x).attr('cy', d => d.y)
      animId = requestAnimationFrame(tick)
    }

    tick()

    return () => {
      cancelAnimationFrame(animId)
      d3.select(container).selectAll('svg').remove()
    }
  }, [])

  return (
    <div className="brain-anim-wrap">
      <div className="brain-anim-header">
        <span className="brain-anim-title">Neural Pattern Visualizer</span>
        <span className="brain-anim-hint">Move cursor over to interact</span>
      </div>
      <div ref={containerRef} className="brain-anim-canvas" />
    </div>
  )
}
