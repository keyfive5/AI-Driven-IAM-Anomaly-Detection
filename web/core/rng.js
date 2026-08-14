/**
 * Deterministic random number utilities.
 *
 * Every stochastic part of ARGUS (synthetic corpus generation, Isolation Forest
 * subsampling and split selection) draws from a seeded generator so that a given
 * seed always produces byte-identical results. Reproducibility is the whole
 * point: a benchmark number you cannot re-run is a rumour, not a measurement.
 */

/** mulberry32 — small, fast, statistically decent 32-bit PRNG. */
export function makeRng(seed = 1337) {
  let a = seed >>> 0;
  const rng = () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  rng.int = (maxExclusive) => Math.floor(rng() * maxExclusive);
  rng.range = (lo, hi) => lo + rng() * (hi - lo);
  rng.intRange = (lo, hi) => lo + Math.floor(rng() * (hi - lo + 1));
  rng.pick = (arr) => arr[Math.floor(rng() * arr.length)];
  rng.bool = (p = 0.5) => rng() < p;
  /** Box–Muller normal deviate. */
  rng.normal = (mean = 0, sd = 1) => {
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  /** Weighted pick. `weights[i]` need not be normalised. */
  rng.weighted = (items, weights) => {
    let total = 0;
    for (const w of weights) total += w;
    let r = rng() * total;
    for (let i = 0; i < items.length; i++) {
      r -= weights[i];
      if (r <= 0) return items[i];
    }
    return items[items.length - 1];
  };
  /** In-place Fisher–Yates shuffle. */
  rng.shuffle = (arr) => {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  };
  return rng;
}

/** Hash an arbitrary string into a 32-bit seed (FNV-1a). */
export function hashSeed(str) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
