/**
 * GE-79 MCI Explorer — D3.js Brain Animation with Brain Shape
 * Uses D3 force simulation to arrange circles in actual brain silhouette
 * 
 * Features:
 * - Circles positioned to form realistic brain shape
 * - Large cerebrum lobe with gyri (wrinkles)
 * - Small cerebellum below with brainstem
 * - D3 force simulation for organic clustering
 * - Canvas rendering (high performance)
 * - Mouse interaction with central attractor
 */

const BrainAnimation = (function() {
  'use strict';

  const config = {
    width: 800,
    height: 600,
    particleCount: 200,
    particleColor: '#2d9a96',
    alphaTarget: 0.3,
    velocityDecay: 0.1,
    collideIterations: 3,
    chargeStrength: -800
  };

  let canvas = null;
  let context = null;
  let nodes = [];
  let simulation = null;

  /**
   * Create brain-shaped initial positions
   */
  function generateBrainParticles(count) {
    const particles = [];
    
    // Node 0: Central attractor (invisible, for mouse control)
    particles.push({
      id: 0,
      r: 1,
      group: 0,
      x: 0,
      y: 0,
      vx: 0,
      vy: 0
    });

    let particleIndex = 1;

    // CEREBRUM (main brain lobe) - ~170 particles
    const cerebrumParticles = Math.floor(count * 0.85);
    for (let i = 0; i < cerebrumParticles && particleIndex < count; i++) {
      // Create brain outline with gyri patterns
      const angle = (i / cerebrumParticles) * Math.PI * 2;
      
      // Brain lobe shape with wrinkles
      const noise1 = Math.sin(angle * 4) * 30;
      const noise2 = Math.sin(angle * 8) * 15;
      const noise3 = Math.cos(angle * 2) * 20;
      const verticalTaper = Math.sin(angle * 3) * 10;
      
      let radius = 110 + noise1 + noise2 + noise3 + verticalTaper;
      
      // Position particles along brain outline
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius * 0.8 - 40; // Shift up slightly
      
      // Add jitter for organic look
      const jitterX = (Math.random() - 0.5) * 20;
      const jitterY = (Math.random() - 0.5) * 20;
      
      // Varied particle sizes
      const sizeVariation = Math.random();
      let particleRadius;
      if (sizeVariation < 0.2) {
        particleRadius = 7 + Math.random() * 3;
      } else if (sizeVariation < 0.6) {
        particleRadius = 4 + Math.random() * 2.5;
      } else {
        particleRadius = 2 + Math.random() * 2;
      }

      particles.push({
        id: particleIndex++,
        r: particleRadius,
        group: 1,
        x: x + jitterX,
        y: y + jitterY,
        vx: 0,
        vy: 0
      });
    }

    // CEREBELLUM (small bottom lobe) - ~20 particles
    const cerebellumParticles = count - particleIndex;
    const cerebellumRadius = 45;
    for (let i = 0; i < cerebellumParticles && particleIndex < count; i++) {
      const angle = (i / cerebellumParticles) * Math.PI * 2;
      
      // Wrinkled, compact structure
      const wrinkles = Math.sin(angle * 10) * 5;
      const radius = cerebellumRadius + wrinkles;
      
      const x = Math.cos(angle) * radius * 0.95;
      const y = Math.sin(angle) * radius * 0.9 + 120; // Position below cerebrum
      
      const jitterX = (Math.random() - 0.5) * 12;
      const jitterY = (Math.random() - 0.5) * 12;
      
      // Cerebellum has smaller circles
      const particleRadius = 3 + Math.random() * 2;

      particles.push({
        id: particleIndex++,
        r: particleRadius,
        group: 2,
        x: x + jitterX,
        y: y + jitterY,
        vx: 0,
        vy: 0
      });
    }
    
    return particles;
  }

  /**
   * Initialize canvas and simulation
   */
  function init(selector, options = {}) {
    Object.assign(config, options);

    // Create canvas
    const container = d3.select(selector);
    canvas = container.append('canvas')
      .attr('width', config.width)
      .attr('height', config.height)
      .style('display', 'block')
      .node();

    context = canvas.getContext('2d');

    // Generate brain-shaped particles
    nodes = generateBrainParticles(config.particleCount);

    // Create force simulation
    simulation = d3.forceSimulation(nodes)
      .alphaTarget(config.alphaTarget)
      .velocityDecay(config.velocityDecay)
      .force('x', d3.forceX().strength(0.01)) // Increased to push particles back to center
      .force('y', d3.forceY().strength(0.01)) // Increased to push particles back to center
      .force('collide', d3.forceCollide().radius(d => d.r + 1.5).iterations(config.collideIterations))
      .force('charge', d3.forceManyBody().strength((d, i) => {
        // Central attractor pulls others inward to maintain brain shape
        if (i === 0) return -config.chargeStrength;
        return 0;
      }))
      .on('tick', ticked);

    // Mouse interaction
    d3.select(canvas)
      .on('mousemove', pointermoved)
      .on('mouseleave', () => {
        // Release attractor
        if (nodes[0]) {
          nodes[0].fx = null;
          nodes[0].fy = null;
        }
      });

    // Prevent scroll on touch
    d3.select(canvas)
      .on('touchmove', event => event.preventDefault());
  }

  /**
   * Handle mouse movement - pull attractor toward cursor (constrained within card)
   */
  function pointermoved(event) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    if (nodes[0]) {
      // Constrain attractor to stay within card bounds with padding
      const padding = 80; // Keep attraction inside card edges
      const constrainedX = Math.max(-config.width / 2 + padding, 
                                    Math.min(config.width / 2 - padding, x - config.width / 2));
      const constrainedY = Math.max(-config.height / 2 + padding, 
                                    Math.min(config.height / 2 - padding, y - config.height / 2));
      
      nodes[0].fx = constrainedX;
      nodes[0].fy = constrainedY;
    }
  }

  /**
   * Render frame using canvas
   */
  function ticked() {
    // Keep all particles within card bounds
    const maxRadius = Math.max(...nodes.slice(1).map(d => d.r)) + 5;
    const boundaryX = config.width / 2 - maxRadius;
    const boundaryY = config.height / 2 - maxRadius;
    
    for (let i = 1; i < nodes.length; ++i) {
      const d = nodes[i];
      
      // Hard constraints: keep particles inside card
      if (d.x - d.r < -boundaryX) d.x = -boundaryX + d.r;
      if (d.x + d.r > boundaryX) d.x = boundaryX - d.r;
      if (d.y - d.r < -boundaryY) d.y = -boundaryY + d.r;
      if (d.y + d.r > boundaryY) d.y = boundaryY - d.r;
    }
    
    context.clearRect(0, 0, config.width, config.height);
    context.save();
    context.translate(config.width / 2, config.height / 2);

    // Draw all circles except the attractor
    for (let i = 1; i < nodes.length; ++i) {
      const d = nodes[i];
      context.beginPath();
      context.moveTo(d.x + d.r, d.y);
      context.arc(d.x, d.y, d.r, 0, 2 * Math.PI);
      
      // Teal color with slight variation by size
      if (d.r > 6) {
        context.fillStyle = '#2d9a96'; // Bright teal for large
      } else if (d.r > 3.5) {
        context.fillStyle = '#1f8a8a'; // Medium teal
      } else {
        context.fillStyle = '#166b6b'; // Dark teal for small
      }
      
      context.globalAlpha = 0.85;
      context.fill();
      
      // Optional: add subtle stroke
      context.strokeStyle = 'rgba(0, 0, 0, 0.1)';
      context.lineWidth = 0.5;
      context.stroke();
    }

    context.restore();
    context.globalAlpha = 1.0;
  }

  /**
   * Stop simulation
   */
  function stop() {
    if (simulation) {
      simulation.stop();
    }
  }

  /**
   * Public API
   */
  return {
    init: init,
    stop: stop,
    updateConfig: (newConfig) => Object.assign(config, newConfig),
    getNodes: () => nodes,
    getConfig: () => ({ ...config })
  };
})();

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = BrainAnimation;
}
