/**
 * Robust statistics: the second opinion in the ensemble.
 *
 * Isolation Forest is good at joint weirdness and blind to a single feature
 * that is merely extreme. A per-feature robust z-score covers exactly that gap,
 * and unlike a mean/σ z-score it does not get dragged around by the outliers it
 * is supposed to find — median and MAD have a 50% breakdown point.
 *
 *   z = 0.6745 · (x − median) / MAD
 *
 * (0.6745 makes MAD a consistent estimator of σ for normal data.)
 *
 * The per-event score is the mean of the three largest directional z-scores,
 * so an event needs a few genuinely extreme axes rather than one lucky spike.
 */

export function median(sorted) {
  const n = sorted.length;
  if (!n) return 0;
  const mid = n >> 1;
  return n % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function quantile(sorted, q) {
  const n = sorted.length;
  if (!n) return 0;
  const pos = (n - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Median and MAD for each column of a packed matrix. */
export function columnStats(matrix, n, D) {
  const med = new Float64Array(D);
  const mad = new Float64Array(D);
  const col = new Float64Array(n);
  for (let f = 0; f < D; f++) {
    for (let i = 0; i < n; i++) col[i] = matrix[i * D + f];
    const sorted = Float64Array.from(col).sort();
    const m = median(sorted);
    med[f] = m;
    for (let i = 0; i < n; i++) col[i] = Math.abs(matrix[i * D + f] - m);
    const devSorted = Float64Array.from(col).sort();
    let d = median(devSorted);
    if (d < 1e-9) {
      // Degenerate spread (e.g. a mostly-zero indicator). Fall back to the
      // inter-quartile range, then to a floor, so a rare 1 still scores.
      const iqr = quantile(sorted, 0.75) - quantile(sorted, 0.25);
      d = iqr > 1e-9 ? iqr / 1.349 : 0;
    }
    mad[f] = d;
  }
  return { med, mad };
}

/**
 * Directional robust z for one row.
 * `dirs[f]` is +1 when large values are suspicious, −1 when small ones are,
 * 0 when the feature is only meaningful jointly (encodings) and is skipped.
 */
export function robustZRow(matrix, i, D, stats, dirs, out) {
  const off = i * D;
  for (let f = 0; f < D; f++) {
    const dir = dirs[f];
    if (!dir) { out[f] = 0; continue; }
    const spread = stats.mad[f];
    if (spread < 1e-9) {
      // No spread at all: any deviation from the constant is fully anomalous.
      out[f] = matrix[off + f] === stats.med[f] ? 0 : 3.5 * dir * Math.sign(matrix[off + f] - stats.med[f]);
      continue;
    }
    out[f] = (0.6745 * (matrix[off + f] - stats.med[f])) / spread * dir;
  }
  return out;
}

/** Aggregate robust-z anomaly score for every row: mean of the top-3 z values. */
export function robustZScores(matrix, n, D, dirs) {
  const stats = columnStats(matrix, n, D);
  const scores = new Float64Array(n);
  const z = new Float64Array(D);
  for (let i = 0; i < n; i++) {
    robustZRow(matrix, i, D, stats, dirs, z);
    let a = -Infinity;
    let b = -Infinity;
    let c = -Infinity;
    for (let f = 0; f < D; f++) {
      const v = z[f];
      if (v > a) { c = b; b = a; a = v; }
      else if (v > b) { c = b; b = v; }
      else if (v > c) { c = v; }
    }
    const top = [a, b, c].filter((v) => Number.isFinite(v));
    scores[i] = top.length ? Math.max(0, top.reduce((s, v) => s + v, 0) / top.length) : 0;
  }
  return { scores, stats };
}

/**
 * Rank-normalise scores to [0,1] using average ranks for ties.
 * Rank space is what makes three detectors on wildly different scales
 * combinable without one silently dominating the sum.
 */
export function rankNormalise(scores) {
  const n = scores.length;
  if (!n) return new Float64Array(0);
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => scores[a] - scores[b]);
  const out = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && scores[idx[j + 1]] === scores[idx[i]]) j++;
    const avgRank = (i + j) / 2;
    const norm = n > 1 ? avgRank / (n - 1) : 0;
    for (let k = i; k <= j; k++) out[idx[k]] = norm;
    i = j + 1;
  }
  return out;
}

