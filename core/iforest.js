/**
 * Isolation Forest, implemented from first principles.
 *
 * Liu, Ting & Zhou (2008): anomalies are few and different, so random axis
 * splits isolate them in fewer partitions than normal points. The score is the
 * expected path length across an ensemble of random trees, normalised by the
 * average path length of an unsuccessful BST search, c(ψ).
 *
 *   s(x, ψ) = 2 ^ ( -E[h(x)] / c(ψ) )        s → 1 means anomalous
 *
 * Two additions over the textbook version, both needed for security telemetry:
 *
 *   - **Split-path attribution.** While scoring, the features that cut the
 *     point off early are recorded with weight 1/(depth+1), which gives a
 *     per-event "which axes isolated this" ranking for explanations.
 *   - **Degenerate-feature handling.** IAM features are full of constants
 *     within a subsample (`weekend` on a weekday batch). Split candidates are
 *     drawn only from features that actually vary in the node.
 *
 * Operates directly on a packed row-major Float64Array to keep 100k-event runs
 * allocation-free in the inner loop.
 */

import { makeRng } from './rng.js';

const EULER = 0.5772156649015329;

/** Average path length of an unsuccessful search in a BST of n nodes. */
export function cFactor(n) {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + EULER) - (2 * (n - 1)) / n;
}

function buildTree(matrix, D, rows, depth, maxDepth, rng) {
  const n = rows.length;
  if (depth >= maxDepth || n <= 1) {
    return { leaf: true, size: n, depth };
  }

  // Find features that actually vary in this node; splitting on a constant is
  // a wasted level and biases path lengths upward.
  let feature = -1;
  let min = 0;
  let max = 0;
  for (let attempt = 0; attempt < 12; attempt++) {
    const f = rng.int(D);
    let lo = Infinity;
    let hi = -Infinity;
    for (let k = 0; k < n; k++) {
      const v = matrix[rows[k] * D + f];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (hi - lo > 1e-12) { feature = f; min = lo; max = hi; break; }
  }
  if (feature === -1) return { leaf: true, size: n, depth };

  const split = min + rng() * (max - min);
  const left = [];
  const right = [];
  for (let k = 0; k < n; k++) {
    if (matrix[rows[k] * D + feature] < split) left.push(rows[k]);
    else right.push(rows[k]);
  }
  if (!left.length || !right.length) return { leaf: true, size: n, depth };

  return {
    leaf: false,
    feature,
    split,
    depth,
    left: buildTree(matrix, D, left, depth + 1, maxDepth, rng),
    right: buildTree(matrix, D, right, depth + 1, maxDepth, rng),
  };
}

/**
 * @param {Float64Array} matrix packed n × D
 * @param {number} n rows
 * @param {number} D columns
 * @param {object} opts { trees, sampleSize, seed }
 */
export function trainIsolationForest(matrix, n, D, opts = {}) {
  const trees = opts.trees ?? 120;
  // The subsample can never exceed the corpus. Clamping the other way round
  // (a floor of 8 on a 2-event file) walks the shuffle past the end of the
  // pool; typed arrays swallow those writes silently, which is exactly the
  // kind of accident that survives until it does not.
  const psi = Math.max(1, Math.min(opts.sampleSize ?? 256, n));
  const rng = makeRng(opts.seed ?? 7);
  const maxDepth = Math.ceil(Math.log2(Math.max(2, psi))) + 1;

  const forest = [];
  const pool = new Int32Array(n);
  for (let i = 0; i < n; i++) pool[i] = i;

  for (let t = 0; t < trees; t++) {
    // Partial Fisher–Yates: shuffle only the first psi slots.
    for (let i = 0; i < psi; i++) {
      const j = i + rng.int(n - i);
      const tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
    }
    const rows = Array.from(pool.subarray(0, psi));
    forest.push(buildTree(matrix, D, rows, 0, maxDepth, rng));
  }

  return { trees: forest, psi, D, c: cFactor(psi) };
}

/** Path length of one row through one tree, with early-split attribution. */
function pathLength(node, matrix, off, attribution) {
  let depth = 0;
  let cur = node;
  while (!cur.leaf) {
    if (attribution) attribution[cur.feature] += 1 / (depth + 1);
    cur = matrix[off + cur.feature] < cur.split ? cur.left : cur.right;
    depth++;
  }
  return depth + cFactor(cur.size);
}

/**
 * Score rows [from, to) into `out`.
 *
 * Scoring is the single most expensive step — n × trees tree descents — and on
 * a large corpus doing it in one call freezes the tab for tens of seconds with
 * no way to tell whether it is working or wedged. Exposing a range lets the
 * caller run it in slices and yield to the event loop between them.
 */
export function scoreIsolationForestRange(model, matrix, out, from, to) {
  const { trees, D, c } = model;
  const t = trees.length;
  for (let i = from; i < to; i++) {
    const off = i * D;
    let sum = 0;
    for (let k = 0; k < t; k++) sum += pathLength(trees[k], matrix, off, null);
    out[i] = Math.pow(2, -(sum / t) / c);
  }
  return out;
}

/** Anomaly score in (0,1] for every row. Higher = more isolated. */
export function scoreIsolationForest(model, matrix, n) {
  return scoreIsolationForestRange(model, matrix, new Float64Array(n), 0, n);
}

/**
 * Which features isolated row `i`, normalised to sum 1.
 * Returns an array of D weights aligned with the feature catalogue.
 */
export function attributeIsolation(model, matrix, i) {
  const { trees, D } = model;
  const attribution = new Float64Array(D);
  const off = i * D;
  for (const tree of trees) pathLength(tree, matrix, off, attribution);
  let sum = 0;
  for (let k = 0; k < D; k++) sum += attribution[k];
  if (sum > 0) for (let k = 0; k < D; k++) attribution[k] /= sum;
  return attribution;
}
