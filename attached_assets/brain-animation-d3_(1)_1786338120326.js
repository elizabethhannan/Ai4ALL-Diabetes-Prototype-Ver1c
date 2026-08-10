/**
 * GE-79 MCI Explorer — D3.js Particle Brain Animation
 * Version 2.0 — Inspired by Seeing Theory particle visualizations
 * 
 * Renders an interactive brain made of particles that:
 * - Form a brain silhouette
 * - Respond to mouse movement
 * - Particles flee from cursor (within 120px radius)
 * - Smoothly return to original positions
 * - Physics-based animation (forces, acceleration, damping)
 * 
 * Usage:
 * <div id="brain-animation"></div>
 * <script src="brain-animation-d3.js"></script>
 * <script>
 *   BrainAnimation.init("#brain-animation", { width: 800, height: 600 });
 * </script>
 */

const BrainAnimation = (function() {
  'use strict';

  // Configuration
  const config = {
    particleCount: 250,
    particleRadius: 3,
    particleRadiusVariance: 3,
    particleColor: '#2d9a96', // Teal
    particleOpacity: 0.7,
    damping: 0.95,
    repelForce: 0.5,
    attractForce: 0.01,
    repelRadius: 120,
    speed: 0.02,
    width: 800,
    height: 600
  };

  // State
  let particles = [];
  let svg = null;
  let circles = null;
  let mouseX = config.width / 2;
  let mouseY = config.height / 2;
  let animationId = null;

  /**
   * Particle constructor
   */
  function Particle(x, y) {
    this.x = x;
    this.y = y;
    this.vx = (Math.random() - 0.5) * 2;
    this.vy = (Math.random() - 0.5) * 2;
    this.originalX = x;
    this.originalY = y;
    this.radius = config.particleRadius + (Math.random() - 0.5) * config.particleRadiusVariance;
  }

  /**
   * Generate brain-shaped point cloud
   * Uses parametric equations to create brain silhouette
   */
  function generateBrainShape(centerX, centerY, scale) {
    const points = [];
    const particleCount = config.particleCount;

    // Brain outline using multiple overlapping circles and curves
    for (let i = 0; i < particleCount; i++) {
      const angle = (i / particleCount) * Math.PI * 2;
      
      // Multi-lobed brain shape using sine waves
      const baseRadius = scale * (90 + Math.sin(angle * 3) * 25 + Math.sin(angle * 2) * 15);
      const x = centerX + Math.cos(angle) * baseRadius;
      const y = centerY + Math.sin(angle) * baseRadius * 0.8; // Compress vertically

      // Add some jitter to make it less perfect
      const jitterX = (Math.random() - 0.5) * 20;
      const jitterY = (Math.random() - 0.5) * 20;

      points.push({
        x: x + jitterX,
        y: y + jitterY
      });
    }

    return points;
  }

  /**
   * Initialize animation
   */
  function init(selector, options = {}) {
    // Merge options
    Object.assign(config, options);

    // Create SVG
    svg = d3.select(selector)
      .append('svg')
      .attr('width', config.width)
      .attr('height', config.height)
      .attr('viewBox', `0 0 ${config.width} ${config.height}`)
      .style('background', 'transparent')
      .style('cursor', 'pointer');

    // Generate brain shape points
    const brainPoints = generateBrainShape(
      config.width / 2,
      config.height / 2,
      1.0
    );

    // Create particles
    particles = brainPoints.map(point => new Particle(point.x, point.y));

    // Create SVG circles for particles
    circles = svg.selectAll('circle')
      .data(particles)
      .enter()
      .append('circle')
      .attr('r', d => d.radius)
      .attr('fill', config.particleColor)
      .attr('opacity', config.particleOpacity);

    // Add interactivity
    svg.on('mousemove', function(event) {
      const [x, y] = d3.pointer(event);
      mouseX = x;
      mouseY = y;
    });

    svg.on('mouseleave', function() {
      mouseX = config.width / 2;
      mouseY = config.height / 2;
    });

    // Start animation loop
    animate();
  }

  /**
   * Update particle positions based on forces
   */
  function updateParticles() {
    particles.forEach((particle, i) => {
      // Calculate distance to mouse
      const dx = mouseX - particle.x;
      const dy = mouseY - particle.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      // Repel from mouse
      if (dist < config.repelRadius && dist > 0) {
        const angle = Math.atan2(dy, dx);
        const force = config.repelForce * (1 - dist / config.repelRadius);
        particle.vx -= Math.cos(angle) * force;
        particle.vy -= Math.sin(angle) * force;
      }

      // Attract to original position
      const attractDx = particle.originalX - particle.x;
      const attractDy = particle.originalY - particle.y;
      particle.vx += attractDx * config.attractForce;
      particle.vy += attractDy * config.attractForce;

      // Apply damping
      particle.vx *= config.damping;
      particle.vy *= config.damping;

      // Update position
      particle.x += particle.vx;
      particle.y += particle.vy;

      // Boundary conditions (soft bounce)
      if (particle.x < 0) {
        particle.x = 0;
        particle.vx *= -0.5;
      }
      if (particle.x > config.width) {
        particle.x = config.width;
        particle.vx *= -0.5;
      }
      if (particle.y < 0) {
        particle.y = 0;
        particle.vy *= -0.5;
      }
      if (particle.y > config.height) {
        particle.y = config.height;
        particle.vy *= -0.5;
      }
    });
  }

  /**
   * Render particles
   */
  function render() {
    circles
      .attr('cx', d => d.x)
      .attr('cy', d => d.y);
  }

  /**
   * Animation loop
   */
  function animate() {
    updateParticles();
    render();
    animationId = requestAnimationFrame(animate);
  }

  /**
   * Stop animation
   */
  function stop() {
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  /**
   * Public API
   */
  return {
    init: init,
    stop: stop,
    updateConfig: (newConfig) => Object.assign(config, newConfig),
    getParticles: () => particles,
    getConfig: () => ({ ...config })
  };
})();

// Export for Node.js / module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = BrainAnimation;
}
