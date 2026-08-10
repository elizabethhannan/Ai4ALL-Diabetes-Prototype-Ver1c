/**
 * GE-79 MCI Explorer — D3.js Realistic Brain Animation
 * Version 2.0 — Matches ProtoApp Version 8 design
 * 
 * Renders an interactive brain that:
 * - Forms a realistic brain shape (two lobes, convoluted surface)
 * - Responds to mouse movement
 * - Particles flee from cursor (within 120px radius)
 * - Smoothly returns to original positions
 * - Physics-based animation
 */

const BrainAnimation = (function() {
  'use strict';

  // Configuration
  const config = {
    particleCount: 280,
    particleRadius: 3.5,
    particleRadiusVariance: 2.5,
    particleColor: '#2d9a96', // Teal
    particleOpacity: 0.85,
    damping: 0.92,
    repelForce: 0.6,
    attractForce: 0.012,
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
   * Generate realistic brain shape point cloud
   * Creates a two-lobed brain with convoluted surfaces
   */
  function generateBrainShape(centerX, centerY, scale) {
    const points = [];
    const particleCount = config.particleCount;

    // Create a more realistic brain shape with two lobes
    for (let i = 0; i < particleCount; i++) {
      const angle = (i / particleCount) * Math.PI * 2;
      
      // Brain shape calculation with realistic lobes
      let radius;
      
      // Create two distinct lobes (left and right hemispheres)
      const lobeInfluence = Math.cos(angle * 2); // Creates left/right distinction
      const convolution = Math.sin(angle * 6) * 8; // Add brain-like wrinkles
      const heightVariation = Math.sin(angle * 3) * 15; // Top/bottom shaping
      
      // Base brain radius with lobe shaping
      const baseRadius = 85 + convolution + (lobeInfluence > 0 ? 5 : 0);
      radius = baseRadius + heightVariation;
      
      // Position calculation
      const x = centerX + Math.cos(angle) * radius * scale;
      const y = centerY + Math.sin(angle) * radius * 0.75 * scale; // Slightly compressed vertically
      
      // Add some organic jitter to make it less perfect
      const jitterX = (Math.random() - 0.5) * 15;
      const jitterY = (Math.random() - 0.5) * 15;

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

      // Repel from mouse (strong repulsion)
      if (dist < config.repelRadius && dist > 0) {
        const angle = Math.atan2(dy, dx);
        const force = config.repelForce * (1 - dist / config.repelRadius);
        particle.vx -= Math.cos(angle) * force;
        particle.vy -= Math.sin(angle) * force;
      }

      // Attract to original position (smooth return)
      const attractDx = particle.originalX - particle.x;
      const attractDy = particle.originalY - particle.y;
      particle.vx += attractDx * config.attractForce;
      particle.vy += attractDy * config.attractForce;

      // Apply damping (smooth movement)
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
   * Animation loop (60 FPS)
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
