// ── Mohr-Coulomb Circle Visualizer ────────────────────────────────────────────
// Given principal stresses and shear strength parameters, draws Mohr circles
// with failure envelope and reports factor of safety.

// Compute stress state at depth from unit parameters
export function stressAtDepth(unit, depth, gwt = 0, deltaSigma = 0) {
  const gamma = unit.params?.gamma ?? 19;
  const sigma_v  = gamma * depth;                   // total vertical stress (kPa)
  const u        = Math.max(0, depth - gwt) * 9.81; // pore pressure (kPa)
  const sigma_v0 = sigma_v - u;                     // effective vertical stress

  // At-rest K0 (Jaky): K0 = 1 - sinφ'
  const phi  = (unit.params?.phi  ?? 30) * Math.PI / 180;
  const K0   = Math.max(0.3, 1 - Math.sin(phi));
  const sigma_h0 = K0 * sigma_v0;                  // effective horizontal stress

  // Applied deviatoric load: adds to vertical stress
  const sigma1 = sigma_v0 + deltaSigma;
  const sigma3 = sigma_h0;

  return { sigma1, sigma3, u, sigma_v, sigma_v0 };
}

// Factor of safety on Mohr-Coulomb failure criterion
// Ratio of radius to distance from centre to failure envelope
function _fs(centre, radius, cPrime, tanPhi) {
  // Distance from origin to envelope line (τ = c' + σ'tanφ)
  // |c' - centre × tanφ| / √(1 + tan²φ) but using perpendicular dist from line to centre
  // Line: tanφ×σ - τ + c' = 0  →  dist = |tanφ×centre + c'| / √(1+tan²φ)
  const dist = (tanPhi * centre + cPrime) / Math.sqrt(1 + tanPhi * tanPhi);
  return dist > 0 ? dist / radius : Infinity;
}

// ── computeMohrCircle ─────────────────────────────────────────────────────────
export function computeMohrCircle(unit, depth, gwt = 0, deltaSigma = 0, undrained = false) {
  const { sigma1, sigma3, u, sigma_v0 } = stressAtDepth(unit, depth, gwt, deltaSigma);
  const centre = (sigma1 + sigma3) / 2;
  const radius = (sigma1 - sigma3) / 2;

  const cPrime = unit.params?.cprime ?? 0;
  const phiDeg = unit.params?.phi    ?? 30;
  const Cu     = unit.params?.cu     ?? null;
  const tanPhi = Math.tan(phiDeg * Math.PI / 180);

  let Fs, mode, cEff, phiEff, tanPhiEff;
  if (undrained && Cu !== null) {
    // Undrained: τf = Cu, φu = 0
    mode = 'undrained';
    cEff = Cu; phiEff = 0; tanPhiEff = 0;
    Fs = radius > 0 ? Cu / radius : Infinity;
  } else {
    mode = 'drained';
    cEff = cPrime; phiEff = phiDeg; tanPhiEff = tanPhi;
    Fs = _fs(centre, radius, cPrime, tanPhi);
  }

  // Tangent point on circle to failure envelope
  const angTan = Math.atan(tanPhiEff);
  const sinA   = Math.sin(angTan);
  const tanX   = centre - radius * sinA;
  const tanY   = cEff + tanPhiEff * tanX;

  return {
    sigma1, sigma3, centre, radius, Fs, mode,
    cEff, phiEff, tanPhiEff,
    tangentPoint: { x: tanX, y: tanY },
    u, sigma_v0,
  };
}

// ── renderMohrCircle ──────────────────────────────────────────────────────────
export function renderMohrCircle(result, width = 280, height = 220) {
  const { sigma1, sigma3, centre, radius, Fs, cEff, phiEff, tanPhiEff,
          tangentPoint, mode } = result;

  // Axis range: 0 to 1.3×σ1 with a bit of headroom
  const maxSig = Math.max(sigma1 * 1.3, cEff / Math.max(tanPhiEff, 0.001) + 20, 200);
  const maxTau = Math.max(radius * 1.6, cEff + tanPhiEff * maxSig * 0.5, 60);

  const pad = { t: 12, r: 10, b: 28, l: 36 };
  const W = width  - pad.l - pad.r;
  const H = height - pad.t - pad.b;

  const sx = s => pad.l + (s / maxSig) * W;
  const sy = t => pad.t + H - (t / maxTau) * H;

  // Mohr circle arc (0..π semicircle)
  const steps = 60;
  let arc = '';
  for (let i = 0; i <= steps; i++) {
    const theta = (i / steps) * Math.PI;
    const sx_ = sx(centre + radius * Math.cos(theta));
    const sy_ = sy(radius * Math.sin(theta));
    arc += i === 0 ? `M${sx_},${sy_}` : `L${sx_},${sy_}`;
  }

  // Failure envelope line: τ = cEff + tanPhiEff × σ
  const envX0 = 0, envX1 = maxSig;
  const envY0 = cEff, envY1 = cEff + tanPhiEff * maxSig;

  const fsColor = Fs < 1.0 ? '#c0392b' : Fs < 1.5 ? '#e67e22' : '#4a7c59';
  const fsLabel = isFinite(Fs) ? `FS = ${Fs.toFixed(2)}` : 'FS = ∞';

  const sigTicks = Array.from({ length: 5 }, (_, i) => Math.round(maxSig / 4 * i));
  const tauTicks = Array.from({ length: 4 }, (_, i) => Math.round(maxTau / 3 * i));

  return `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg"
    style="font-family:monospace;font-size:9px">
    <!-- Grid -->
    ${sigTicks.map(s => `<line x1="${sx(s)}" y1="${pad.t}" x2="${sx(s)}" y2="${pad.t+H}"
      stroke="#333" stroke-width="0.5"/>`).join('')}
    ${tauTicks.map(t => `<line x1="${pad.l}" y1="${sy(t)}" x2="${pad.l+W}" y2="${sy(t)}"
      stroke="#333" stroke-width="0.5"/>`).join('')}
    <!-- Failure envelope shaded zone -->
    <polygon points="${sx(envX0)},${sy(envY0)} ${sx(envX1)},${sy(envY1)} ${sx(envX1)},${pad.t} ${sx(envX0)},${pad.t}"
      fill="#c0392b" opacity="0.08"/>
    <!-- Failure envelope line -->
    <line x1="${sx(envX0)}" y1="${sy(envY0)}" x2="${sx(envX1)}" y2="${sy(envY1)}"
      stroke="#c0392b" stroke-width="1.5"/>
    <!-- c' intercept label -->
    ${cEff > 0 ? `<text x="${sx(0)+4}" y="${sy(cEff)-3}" fill="#c0392b" font-size="8">c'=${cEff}kPa</text>` : ''}
    <!-- φ' label -->
    <text x="${sx(maxSig*0.75)}" y="${sy(envY1*0.72)}" fill="#c0392b" font-size="8">φ'=${phiEff}°</text>
    <!-- Mohr circle -->
    <path d="${arc}" fill="none" stroke="${fsColor}" stroke-width="2"/>
    <!-- σ3 and σ1 markers -->
    <circle cx="${sx(sigma3)}" cy="${sy(0)}" r="3" fill="${fsColor}"/>
    <circle cx="${sx(sigma1)}" cy="${sy(0)}" r="3" fill="${fsColor}"/>
    <!-- Centre -->
    <circle cx="${sx(centre)}" cy="${sy(0)}" r="2" fill="#888"/>
    <!-- Tangent point -->
    ${isFinite(Fs) && Fs < 99 ? `<circle cx="${sx(tangentPoint.x)}" cy="${sy(tangentPoint.y)}" r="3" fill="#f1c40f"/>` : ''}
    <!-- Axes -->
    <line x1="${pad.l}" y1="${sy(0)}" x2="${pad.l+W}" y2="${sy(0)}" stroke="#888" stroke-width="1.5"/>
    <line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t+H}" stroke="#888" stroke-width="1.5"/>
    <!-- Tick labels σ -->
    ${sigTicks.map(s => `<text x="${sx(s)}" y="${pad.t+H+10}" text-anchor="middle" fill="#aaa">${s}</text>`).join('')}
    <!-- Tick labels τ -->
    ${tauTicks.map(t => `<text x="${pad.l-4}" y="${sy(t)+3}" text-anchor="end" fill="#aaa">${t}</text>`).join('')}
    <!-- Axis labels -->
    <text x="${pad.l+W/2}" y="${height-2}" text-anchor="middle" fill="#ccc">σ' (kPa)</text>
    <text x="${pad.l-28}" y="${pad.t+H/2}" transform="rotate(-90,${pad.l-28},${pad.t+H/2})" text-anchor="middle" fill="#ccc">τ (kPa)</text>
    <!-- FS badge -->
    <rect x="${pad.l+W-62}" y="${pad.t}" width="62" height="20" rx="3" fill="${fsColor}" opacity="0.9"/>
    <text x="${pad.l+W-31}" y="${pad.t+13}" text-anchor="middle" fill="#fff" font-weight="bold" font-size="10">${fsLabel}</text>
    <!-- Mode badge -->
    <text x="${pad.l+4}" y="${pad.t+10}" fill="#888" font-size="8">${mode}</text>
  </svg>`;
}
