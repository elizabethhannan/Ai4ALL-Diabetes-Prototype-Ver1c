import { useRef, useEffect } from 'react'
import './BrainAnimation.css'

// ── Fixed canvas coordinate space ─────────────────────
const W = 680
const H = 220

interface Particle {
  x: number; y: number
  vx: number; vy: number
  ox: number; oy: number
  r: number
}

const CFG = {
  count: 1200,
  minR: 3,
  maxR: 5.5,
  color: '52, 211, 153',
  opacity: 0.80,
  damping: 0.91,
  repel: 0.70,
  attract: 0.013,
  repelRadius: 110,
}

// Brain silhouette in normalised coords [-1, 1] × [-1, 1]
function insideBrain(nx: number, ny: number): boolean {
  // Wide, slightly flattened ellipse — the brain mass
  return (nx / 0.88) ** 2 + (ny / 0.80) ** 2 <= 1
}

function buildParticles(cx: number, cy: number, scale: number): Particle[] {
  const pts: Particle[] = []
  let attempts = 0
  while (pts.length < CFG.count && attempts < 60_000) {
    attempts++
    const nx = (Math.random() - 0.5) * 2   // [-1, 1]
    const ny = (Math.random() - 0.5) * 2   // [-1, 1]
    if (insideBrain(nx, ny)) {
      const x = cx + nx * scale
      const y = cy + ny * scale
      pts.push({
        x, y, vx: 0, vy: 0, ox: x, oy: y,
        r: CFG.minR + Math.random() * (CFG.maxR - CFG.minR),
      })
    }
  }
  return pts
}

export function BrainAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // Use fixed coordinate space — avoids layout-timing issues
    canvas.width  = W
    canvas.height = H

    const ctx = canvas.getContext('2d')!
    const cx = W / 2          // 340
    const cy = H / 2          // 110
    // Scale: brain fills ~88 % of height
    const scale = H * 0.43    // ≈ 94.6 px → ellipse 166 × 151 px

    const particles = buildParticles(cx, cy, scale)

    // Mouse starts far off-canvas — nothing gets repelled until user hovers
    let mouseX = -9999
    let mouseY = -9999
    let animId: number

    const toCanvasCoords = (e: MouseEvent) => {
      const r   = canvas.getBoundingClientRect()
      const scX = W / r.width   // coordinate → CSS-pixel ratio
      const scY = H / r.height
      mouseX = (e.clientX - r.left) * scX
      mouseY = (e.clientY - r.top)  * scY
    }
    canvas.addEventListener('mousemove', toCanvasCoords)
    canvas.addEventListener('mouseleave', () => { mouseX = cx; mouseY = cy })

    function tick() {
      ctx.clearRect(0, 0, W, H)

      for (const p of particles) {
        // Repel
        const dx   = p.x - mouseX
        const dy   = p.y - mouseY
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < CFG.repelRadius && dist > 0) {
          const f = CFG.repel * (1 - dist / CFG.repelRadius)
          p.vx += (dx / dist) * f
          p.vy += (dy / dist) * f
        }
        // Spring home
        p.vx += (p.ox - p.x) * CFG.attract
        p.vy += (p.oy - p.y) * CFG.attract
        // Damping
        p.vx *= CFG.damping
        p.vy *= CFG.damping
        p.x  += p.vx
        p.y  += p.vy

        ctx.beginPath()
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${CFG.color},${CFG.opacity})`
        ctx.fill()
      }

      animId = requestAnimationFrame(tick)
    }

    tick()

    return () => {
      cancelAnimationFrame(animId)
      canvas.removeEventListener('mousemove', toCanvasCoords)
    }
  }, [])

  return (
    <div className="brain-anim-wrap">
      <div className="brain-anim-header">
        <span className="brain-anim-title">Neural Pattern Visualizer</span>
        <span className="brain-anim-hint">Move cursor over to interact</span>
      </div>
      <canvas ref={canvasRef} className="brain-anim-canvas" />
    </div>
  )
}
