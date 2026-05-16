// ── Seismic Liquefaction Assessment — Robertson & Wride (1998) / Robertson (2009) ──
// CPT-based method: CSR vs CRR framework with Ic soil classification.
// Returns per-depth FS and Liquefaction Potential Index (LPI).

const Pa = 100; // atmospheric pressure (kPa)

// Liao & Whitman (1986) stress reduction factor
function _rd(z) {
  if (z <= 9.15)  return 1.0 - 0.00765 * z;
  if (z <= 23.0)  return 1.174 - 0.0267 * z;
  return Math.max(0.33, 1.174 - 0.0267 * z);
}

// Vertical stresses at each depth given GWT depth
function _stresses(depths, gamma = 18, gwt = 2.0) {
  const sigma = [], sigmaEff = [];
  for (const z of depths) {
    const sv = gamma * z;
    const u  = Math.max(0, (z - gwt)) * 9.81;
    sigma.push(sv);
    sigmaEff.push(Math.max(sv - u, 1)); // ≥1 kPa to avoid division by zero
  }
  return { sigma, sigmaEff };
}

// Magnitude Scaling Factor (Idriss 1999)
function _msf(Mw) {
  return Math.min(1.8, 10 ** 2.24 / (Mw ** 2.56));
}

// Robertson (2009) iterative Ic and normalised qc1N
function _normalise(qc_MPa, fs_kPa, sigmav, sigmaEff) {
  const qc = qc_MPa * 1000; // kPa
  let n = 1.0;
  let qc1N = 0, Ic = 0;
  for (let iter = 0; iter < 10; iter++) {
    const CQ  = Math.min(1.7, (Pa / sigmaEff) ** n);
    qc1N      = CQ * (qc / Pa);
    const Qt  = Math.max(0.001, (qc - sigmav) / sigmaEff);
    const Fr  = Qt > 0.001
      ? Math.max(0.001, fs_kPa / (qc - sigmav)) * 100
      : 0.1;
    const IcNew = Math.sqrt((3.47 - Math.log10(Qt)) ** 2 + (Math.log10(Fr) + 1.22) ** 2);
    n = Math.min(1.0, 0.381 * IcNew + 0.05 * (sigmaEff / Pa) - 0.15);
    if (Math.abs(IcNew - Ic) < 0.001) { Ic = IcNew; break; }
    Ic = IcNew;
  }
  return { qc1N, Ic };
}

// Fines correction → qc1Ncs (Robertson & Wride 1998)
function _qc1Ncs(qc1N, Ic) {
  if (Ic < 1.64) return qc1N;
  const Kc = -0.403 * Ic ** 4 + 5.581 * Ic ** 3 - 21.63 * Ic ** 2 + 33.75 * Ic - 17.88;
  return Math.max(Kc, 1) * qc1N;
}

// CRR for Mw=7.5 (Robertson & Wride 1998)
function _crr75(qc1Ncs) {
  if (qc1Ncs >= 160) return Infinity; // Dense — not susceptible
  if (qc1Ncs < 50)  return 0.833 * (qc1Ncs / 1000) + 0.05;
  return 93 * (qc1Ncs / 1000) ** 3 + 0.08;
}

// ── assessLiquefaction ────────────────────────────────────────────────────────
// cptLog: { depths[], qc[] (MPa), fs[] (kPa), groundLevel }
// options: { amax, Mw, gwt, gamma, gammaEff }
// returns { depths[], Ic[], qc1N[], FS[], susceptible[], lpi, lpiRating }
export function assessLiquefaction(cptLog, options = {}) {
  const amax  = options.amax  ?? 0.15;  // PGA as fraction of g
  const Mw    = options.Mw    ?? 7.0;   // earthquake magnitude
  const gwt   = options.gwt   ?? 2.0;   // groundwater table depth (m)
  const gamma = options.gamma ?? 18.0;  // total unit weight (kN/m³)

  const msf = _msf(Mw);
  const { depths, qc, fs } = cptLog;
  const { sigma, sigmaEff } = _stresses(depths, gamma, gwt);

  const IcArr = [], qc1NArr = [], FSArr = [], susceptibleArr = [];

  for (let i = 0; i < depths.length; i++) {
    const z     = depths[i];
    const { qc1N, Ic } = _normalise(qc[i], fs[i], sigma[i], sigmaEff[i]);
    IcArr.push(Ic);
    qc1NArr.push(qc1N);

    let FS = Infinity, susc = false;

    if (Ic < 2.6) { // potentially susceptible sandy soil
      susc = true;
      const qcNcs = _qc1Ncs(qc1N, Ic);
      const crr   = _crr75(qcNcs);
      const rd_z  = _rd(z);
      const csr   = 0.65 * (sigma[i] / sigmaEff[i]) * amax * rd_z;
      FS = crr === Infinity ? Infinity : (crr * msf) / csr;
    }
    FSArr.push(Math.min(FS, 9.99));
    susceptibleArr.push(susc);
  }

  // LPI — integrate to 20 m depth (Iwasaki et al. 1978)
  let lpi = 0;
  for (let i = 0; i < depths.length - 1; i++) {
    const z  = depths[i];
    if (z > 20) break;
    const dz = depths[i + 1] - z;
    const F  = susceptibleArr[i] && FSArr[i] < 1.0 ? (1 - FSArr[i]) : 0;
    const w  = Math.max(0, 10 - 0.5 * z);
    lpi += F * w * dz;
  }

  const lpiRating = lpi < 2 ? 'Low' : lpi < 5 ? 'Moderate' : lpi < 15 ? 'High' : 'Very High';

  return { depths, Ic: IcArr, qc1N: qc1NArr, FS: FSArr,
           susceptible: susceptibleArr, lpi, lpiRating, amax, Mw, gwt };
}

// ── renderLiquefactionProfile ─────────────────────────────────────────────────
// Returns an SVG string showing FS profile, colour-coded by risk.
export function renderLiquefactionProfile(result, width = 200, height = 320) {
  const { depths, FS, susceptible, lpi, lpiRating } = result;
  if (!depths.length) return '<p class="hint">No CPT data</p>';

  const maxDepth = Math.min(Math.max(...depths), 25);
  const pad = { t: 20, r: 10, b: 30, l: 36 };
  const W = width  - pad.l - pad.r;
  const H = height - pad.t - pad.b;

  const xScale = x  => pad.l + (Math.min(x, 3) / 3) * W;
  const yScale = z  => pad.t + (z / maxDepth) * H;

  // Build path
  let path = '';
  for (let i = 0; i < depths.length; i++) {
    if (depths[i] > 25) break;
    const px = xScale(FS[i]);
    const py = yScale(depths[i]);
    path += i === 0 ? `M${px},${py}` : `L${px},${py}`;
  }

  // Colour-coded zones
  const zones = depths.map((z, i) => {
    if (z > 25) return '';
    const fs  = FS[i];
    const col = !susceptible[i] ? '#4a7c59'
              : fs < 1.0 ? '#c0392b'
              : fs < 1.2 ? '#e67e22'
              : '#4a7c59';
    const y1 = yScale(z);
    const y2 = i < depths.length - 1 ? yScale(depths[i + 1]) : y1 + 2;
    const x0 = xScale(0);
    const x1 = xScale(Math.min(fs, 3));
    return `<rect x="${x0}" y="${y1}" width="${x1 - x0}" height="${Math.max(0, y2 - y1)}" fill="${col}" opacity="0.35"/>`;
  }).join('');

  // FS=1 and FS=1.2 lines
  const fs1  = xScale(1.0), fs12 = xScale(1.2);
  const gwtY = yScale(result.gwt);

  const lpiCol = lpi < 2 ? '#4a7c59' : lpi < 5 ? '#f1c40f' : lpi < 15 ? '#e67e22' : '#c0392b';

  const depthTicks = [0, 5, 10, 15, 20, 25].filter(d => d <= maxDepth);
  const fsTicks    = [0, 1, 2, 3];

  return `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg"
    style="font-family:monospace;font-size:9px">
    <!-- GWT line -->
    <line x1="${pad.l}" y1="${gwtY}" x2="${pad.l + W}" y2="${gwtY}"
      stroke="#5b9bd5" stroke-width="1" stroke-dasharray="4,2" opacity="0.7"/>
    <text x="${pad.l + W - 2}" y="${gwtY - 2}" fill="#5b9bd5" text-anchor="end">GWT</text>
    ${zones}
    <!-- FS=1 -->
    <line x1="${fs1}" y1="${pad.t}" x2="${fs1}" y2="${pad.t + H}"
      stroke="#c0392b" stroke-width="1" stroke-dasharray="4,2" opacity="0.7"/>
    <!-- FS=1.2 -->
    <line x1="${fs12}" y1="${pad.t}" x2="${fs12}" y2="${pad.t + H}"
      stroke="#e67e22" stroke-width="1" stroke-dasharray="3,2" opacity="0.6"/>
    <!-- profile -->
    <path d="${path}" fill="none" stroke="#e0e0e0" stroke-width="1.5"/>
    <!-- depth axis -->
    ${depthTicks.map(d =>
      `<line x1="${pad.l - 3}" y1="${yScale(d)}" x2="${pad.l}" y2="${yScale(d)}" stroke="#666"/>
       <text x="${pad.l - 5}" y="${yScale(d) + 3}" text-anchor="end" fill="#aaa">${d}</text>`
    ).join('')}
    <!-- FS axis -->
    ${fsTicks.map(f =>
      `<text x="${xScale(f)}" y="${pad.t + H + 12}" text-anchor="middle" fill="#aaa">${f}</text>`
    ).join('')}
    <!-- axis labels -->
    <text x="${pad.l - 24}" y="${pad.t + H / 2}" transform="rotate(-90,${pad.l - 24},${pad.t + H / 2})"
      text-anchor="middle" fill="#ccc">Depth (m)</text>
    <text x="${pad.l + W / 2}" y="${height - 2}" text-anchor="middle" fill="#ccc">Factor of Safety</text>
    <!-- LPI badge -->
    <rect x="${pad.l + W - 60}" y="${pad.t}" width="60" height="22" rx="3" fill="${lpiCol}" opacity="0.85"/>
    <text x="${pad.l + W - 30}" y="${pad.t + 9}" text-anchor="middle" fill="#fff" font-weight="bold" font-size="8">LPI ${lpi.toFixed(1)}</text>
    <text x="${pad.l + W - 30}" y="${pad.t + 18}" text-anchor="middle" fill="#fff" font-size="8">${lpiRating}</text>
  </svg>`;
}

// ── summarizeCPTLiquefaction ──────────────────────────────────────────────────
// Run assessment on an array of CPT logs; returns summary table rows HTML.
export function summarizeCPTLiquefaction(cptLogs, options) {
  return cptLogs.map(log => {
    const r = assessLiquefaction(log, options);
    const liqueDepths = r.depths.filter((z, i) => r.susceptible[i] && r.FS[i] < 1.0);
    const minFS = Math.min(...r.FS.filter((_, i) => r.susceptible[i]));
    const lpiCol = r.lpi < 2 ? '#4a7c59' : r.lpi < 5 ? '#f1c40f' : r.lpi < 15 ? '#e67e22' : '#c0392b';
    return { log, result: r, liqueDepths, minFS, lpiCol };
  });
}
