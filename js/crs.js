/**
 * Coordinate reference system utilities
 * Implements Helmert 7-parameter + OSGB36 National Grid transforms.
 *
 * Conversions:
 *   bngToWGS84(E, N)     → {lat, lon}  (OSGB36 easting/northing → WGS84 degrees)
 *   wgs84ToBNG(lat, lon) → {E, N}       (WGS84 degrees → OSGB36 easting/northing)
 *   localToWGS84(x, y, origin) → {lat, lon}  (local site grid → WGS84, with collar origin)
 *
 * Accuracy: ~1 m across mainland GB when using the Helmert approximation
 * (OSTN02/OSTN15 would give cm-level but requires a 700k datum grid).
 */

// ── Ellipsoid parameters ──────────────────────────────────────────────────────
const AIRY = { a: 6377563.396, b: 6356256.910 };
const GRS80 = { a: 6378137.000, b: 6356752.3141 };

function ellipsoidF(ell) {
  const e2 = (ell.a * ell.a - ell.b * ell.b) / (ell.a * ell.a);
  return { a: ell.a, b: ell.b, e2 };
}

// ── OSGB National Grid projection parameters ──────────────────────────────────
const NG = {
  lat0: 49 * Math.PI / 180,    // true origin latitude
  lon0: -2 * Math.PI / 180,    // true origin longitude (central meridian)
  N0: -100000,                  // northing of true origin
  E0: 400000,                   // easting of true origin
  F0: 0.9996012717,             // scale factor on central meridian
};

// ── Helmert transformation OSGB36 → WGS84 (7-parameter) ──────────────────────
// From OS document "A guide to coordinate systems in Great Britain" v2.3
const HELMERT_OSGB_TO_WGS84 = {
  tx:  446.448, ty: -125.157, tz:  542.060,   // translation (m)
  rx: -0.1502,  ry:  -0.2470, rz:   -0.8421,  // rotation (arcseconds)
  s:   20.4894,                                 // scale (ppm)
};

function helmert(xyz, params) {
  const { tx, ty, tz, rx, ry, rz, s } = params;
  const sec = Math.PI / (180 * 3600);
  const f = 1 + s * 1e-6;
  const Rx = rx * sec, Ry = ry * sec, Rz = rz * sec;
  return {
    x: f * (xyz.x + Rz * xyz.y - Ry * xyz.z) + tx,
    y: f * (-Rz * xyz.x + xyz.y + Rx * xyz.z) + ty,
    z: f * (Ry * xyz.x - Rx * xyz.y + xyz.z) + tz,
  };
}

// ── Cartesian ↔ geographic ──────────────────────────────────────────────────

function geodeticToCartesian(lat, lon, h, ell) {
  const { a, e2 } = ellipsoidF(ell);
  const N = a / Math.sqrt(1 - e2 * Math.sin(lat) * Math.sin(lat));
  return {
    x: (N + h) * Math.cos(lat) * Math.cos(lon),
    y: (N + h) * Math.cos(lat) * Math.sin(lon),
    z: (N * (1 - e2) + h) * Math.sin(lat),
  };
}

function cartesianToGeodetic(xyz, ell) {
  const { a, b, e2 } = ellipsoidF(ell);
  const e2dash = (a * a - b * b) / (b * b);
  const p = Math.sqrt(xyz.x * xyz.x + xyz.y * xyz.y);
  const lon = Math.atan2(xyz.y, xyz.x);
  // Bowring iteration for latitude
  let lat = Math.atan2(xyz.z, p * (1 - e2));
  for (let i = 0; i < 10; i++) {
    const N  = a / Math.sqrt(1 - e2 * Math.sin(lat) * Math.sin(lat));
    const latNew = Math.atan2(xyz.z + e2 * N * Math.sin(lat), p);
    if (Math.abs(latNew - lat) < 1e-12) { lat = latNew; break; }
    lat = latNew;
  }
  const N = a / Math.sqrt(1 - e2 * Math.sin(lat) * Math.sin(lat));
  const h = p / Math.cos(lat) - N;
  return { lat, lon, h };
}

// ── Transverse Mercator projection (OSGB National Grid) ───────────────────────

function tmProject(lat, lon, ell) {
  const { a, e2 } = ellipsoidF(ell);
  const { lat0, lon0, N0, E0, F0 } = NG;
  const n = (a - ell.b) / (a + ell.b);

  const M = (lat_) => {
    const n2 = n * n, n3 = n * n2, n4 = n * n3;
    return a * F0 * (
      (1 + n + 5/4 * n2 + 5/4 * n3) * (lat_ - lat0)
      - (3*n + 3*n2 + 21/8*n3) * Math.sin(lat_ - lat0) * Math.cos(lat_ + lat0)
      + (15/8 * n2 + 15/8 * n3) * Math.sin(2*(lat_ - lat0)) * Math.cos(2*(lat_ + lat0))
      - (35/24 * n3) * Math.sin(3*(lat_ - lat0)) * Math.cos(3*(lat_ + lat0))
    );
  };

  const sinLat = Math.sin(lat), cosLat = Math.cos(lat), tanLat = Math.tan(lat);
  const nu  = a * F0 / Math.sqrt(1 - e2 * sinLat * sinLat);
  const rho = a * F0 * (1 - e2) / Math.pow(1 - e2 * sinLat * sinLat, 1.5);
  const eta2 = nu / rho - 1;
  const dLon  = lon - lon0;

  const I   = M(lat) + N0;
  const II  = nu / 2 * sinLat * cosLat;
  const III = nu / 24 * sinLat * Math.pow(cosLat, 3) * (5 - Math.pow(tanLat, 2) + 9 * eta2);
  const IIIA= nu / 720 * sinLat * Math.pow(cosLat, 5) * (61 - 58 * Math.pow(tanLat, 2) + Math.pow(tanLat, 4));
  const IV  = nu * cosLat;
  const V   = nu / 6 * Math.pow(cosLat, 3) * (nu / rho - Math.pow(tanLat, 2));
  const VI  = nu / 120 * Math.pow(cosLat, 5) * (5 - 18 * Math.pow(tanLat, 2) + Math.pow(tanLat, 4) + 14 * eta2 - 58 * Math.pow(tanLat, 2) * eta2);

  const N = I + II * dLon * dLon + III * Math.pow(dLon, 4) + IIIA * Math.pow(dLon, 6);
  const E = E0 + IV * dLon + V * Math.pow(dLon, 3) + VI * Math.pow(dLon, 5);
  return { E, N };
}

function tmInverse(E, N, ell) {
  const { a, e2 } = ellipsoidF(ell);
  const { lat0, lon0, N0, E0, F0 } = NG;
  const b = ell.b;
  const n = (a - b) / (a + b);

  // Iterative latitude from meridional arc
  let phi = (N - N0) / (a * F0) + lat0;
  for (let i = 0; i < 20; i++) {
    const n2 = n * n, n3 = n * n2, n4 = n * n3;
    const M = a * F0 * (
      (1 + n + 5/4 * n2 + 5/4 * n3) * (phi - lat0)
      - (3*n + 3*n2 + 21/8*n3) * Math.sin(phi - lat0) * Math.cos(phi + lat0)
      + (15/8 * n2 + 15/8 * n3) * Math.sin(2*(phi - lat0)) * Math.cos(2*(phi + lat0))
      - (35/24 * n3) * Math.sin(3*(phi - lat0)) * Math.cos(3*(phi + lat0))
    );
    const dPhi = (N - N0 - M) / (a * F0);
    phi += dPhi;
    if (Math.abs(dPhi) < 1e-12) break;
  }

  const sinPhi = Math.sin(phi), cosPhi = Math.cos(phi), tanPhi = Math.tan(phi);
  const nu  = a * F0 / Math.sqrt(1 - e2 * sinPhi * sinPhi);
  const rho = a * F0 * (1 - e2) / Math.pow(1 - e2 * sinPhi * sinPhi, 1.5);
  const eta2 = nu / rho - 1;
  const dE    = E - E0;

  const VII  = tanPhi / (2 * rho * nu);
  const VIII = tanPhi / (24 * rho * Math.pow(nu, 3)) * (5 + 3 * tanPhi * tanPhi + eta2 - 9 * tanPhi * tanPhi * eta2);
  const IX   = tanPhi / (720 * rho * Math.pow(nu, 5)) * (61 + 90 * tanPhi * tanPhi + 45 * Math.pow(tanPhi, 4));
  const X    = 1 / (cosPhi * nu);
  const XI   = 1 / (cosPhi * 6 * Math.pow(nu, 3)) * (nu / rho + 2 * tanPhi * tanPhi);
  const XII  = 1 / (cosPhi * 120 * Math.pow(nu, 5)) * (5 + 28 * tanPhi * tanPhi + 24 * Math.pow(tanPhi, 4));
  const XIIA = 1 / (cosPhi * 5040 * Math.pow(nu, 7)) * (61 + 662 * tanPhi * tanPhi + 1320 * Math.pow(tanPhi, 4) + 720 * Math.pow(tanPhi, 6));

  const lat = phi - VII * dE * dE + VIII * Math.pow(dE, 4) - IX * Math.pow(dE, 6);
  const lon = lon0 + X * dE - XI * Math.pow(dE, 3) + XII * Math.pow(dE, 5) - XIIA * Math.pow(dE, 7);
  return { lat, lon };
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Convert OSGB36 National Grid easting/northing (m) to WGS84 lat/lon (degrees).
 */
export function bngToWGS84(E, N) {
  const airy = ellipsoidF(AIRY);
  const { lat: lat36, lon: lon36 } = tmInverse(E, N, AIRY);
  const xyz36 = geodeticToCartesian(lat36, lon36, 0, AIRY);
  const xyz84 = helmert(xyz36, HELMERT_OSGB_TO_WGS84);
  const { lat, lon } = cartesianToGeodetic(xyz84, GRS80);
  return { lat: lat * 180 / Math.PI, lon: lon * 180 / Math.PI };
}

/**
 * Convert WGS84 lat/lon (degrees) to OSGB36 National Grid easting/northing (m).
 */
export function wgs84ToBNG(lat_deg, lon_deg) {
  const lat = lat_deg * Math.PI / 180;
  const lon = lon_deg * Math.PI / 180;
  // Inverse Helmert: WGS84 → OSGB36
  const xyz84 = geodeticToCartesian(lat, lon, 0, GRS80);
  const inv = {
    tx: -HELMERT_OSGB_TO_WGS84.tx, ty: -HELMERT_OSGB_TO_WGS84.ty, tz: -HELMERT_OSGB_TO_WGS84.tz,
    rx: -HELMERT_OSGB_TO_WGS84.rx, ry: -HELMERT_OSGB_TO_WGS84.ry, rz: -HELMERT_OSGB_TO_WGS84.rz,
    s:  -HELMERT_OSGB_TO_WGS84.s,
  };
  const xyz36 = helmert(xyz84, inv);
  const { lat: lat36, lon: lon36 } = cartesianToGeodetic(xyz36, AIRY);
  return tmProject(lat36, lon36, AIRY);
}

/**
 * Format OSGB36 easting/northing as a standard OS grid reference string.
 * e.g. (530000, 180000) → "TQ 30000 80000"
 */
export function toOSGridRef(E, N, digits = 6) {
  const letters = 'VWXYZQRSTULMNOPFGHJKABCDE';
  const e500 = Math.floor(E / 500000);
  const n500 = Math.floor(N / 500000);
  const gridSquare2 = 19 - n500 * 5 + e500;
  if (gridSquare2 < 0 || gridSquare2 >= 25) return `${E.toFixed(0)} ${N.toFixed(0)}`;
  const e100 = Math.floor((E % 500000) / 100000);
  const n100 = Math.floor((N % 500000) / 100000);
  const sq1 = letters[gridSquare2];
  const sq2 = letters[20 - n100 * 5 + e100];
  const step = 10 ** (5 - digits / 2);
  const eLocal = Math.floor((E % 100000) / step);
  const nLocal = Math.floor((N % 100000) / step);
  const pad = (v) => v.toString().padStart(digits / 2, '0');
  return `${sq1}${sq2} ${pad(eLocal)} ${pad(nLocal)}`;
}

/**
 * Convert a local site grid coordinate to WGS84.
 * origin: { E, N } — the site datum in BNG metres
 */
export function localToBNG(lx, ly, origin) {
  return { E: origin.E + lx, N: origin.N + ly };
}
