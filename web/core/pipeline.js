/**
 * The analysis pipeline: raw events in, triaged incidents out.
 *
 *   normalise → features (streaming) → three detectors → rank fusion
 *            → rule floors → alerts → correlation → evaluation
 *
 * Everything runs in the browser on the analyst's machine. No upload, no
 * server, no dependency — an audit log full of principal names and source
 * addresses is exactly the kind of data that should never be posted to someone
 * else's compute in exchange for a demo.
 */

import { buildFeatures, FEATURE_KEYS, D } from './features.js';
import { trainIsolationForest, scoreIsolationForest, attributeIsolation } from './iforest.js';
import { robustZScores, rankNormalise, columnStats, robustZRow } from './stats.js';
import { evaluateRules } from './rules.js';
import { fuseTailMax, depthToRisk, applyRules, explainEvent, buildAlert, DEFAULT_WEIGHTS, FEATURE_DIRS, RISK_DECADES } from './score.js';
import { buildIncidents } from './incidents.js';
import { evaluateScores, ablation, campaignDetection } from './evaluate.js';
import { severityFromRisk } from './schema.js';

const IDX = Object.fromEntries(FEATURE_KEYS.map((k, i) => [k, i]));
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

export const DEFAULT_OPTIONS = {
  threshold: 60,          // ≈ the 8 oddest events per thousand, see score.js
  weights: { ...DEFAULT_WEIGHTS },
  trees: 120,
  sampleSize: 256,
  seed: 7,
  decades: RISK_DECADES,
  enabledRules: null,
  incidentGapMinutes: 45,
  maxAlerts: 4000,
};

/**
 * @param {Array} events normalised events, any order
 * @param {object} options see DEFAULT_OPTIONS
 * @param {(pct:number, msg:string)=>void} onProgress
 */
export async function analyse(events, options = {}, onProgress = () => {}) {
  const opts = { ...DEFAULT_OPTIONS, ...options, weights: { ...DEFAULT_WEIGHTS, ...(options.weights || {}) } };
  const t0 = performance.now();

  const data = events.slice().sort((a, b) => a.ts - b.ts);
  const n = data.length;
  if (!n) throw new Error('No events to analyse.');

  // --- 1. Features + rules, one streaming pass -----------------------------
  onProgress(6, 'Building behavioural baselines…');
  await tick();

  const ruleHits = new Array(n);
  const enabled = opts.enabledRules ? new Set(opts.enabledRules) : null;
  const { matrix, ctx, profiles, priors } = buildFeatures(data, {
    onEvent: (e, i, c) => { ruleHits[i] = evaluateRules(e, c, enabled); },
  });

  // --- 2. Isolation Forest -------------------------------------------------
  onProgress(30, `Growing ${opts.trees} isolation trees…`);
  await tick();
  const forest = trainIsolationForest(matrix, n, D, {
    trees: opts.trees,
    sampleSize: opts.sampleSize,
    seed: opts.seed,
  });
  const ifScores = scoreIsolationForest(forest, matrix, n);

  // --- 3. Robust z ---------------------------------------------------------
  onProgress(52, 'Measuring deviation from robust baselines…');
  await tick();
  const { scores: zScores, stats: colStats } = robustZScores(matrix, n, D, FEATURE_DIRS);

  // --- 4. Behavioural surprisal -------------------------------------------
  onProgress(66, 'Scoring behavioural surprisal…');
  await tick();
  const baseScores = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const off = i * D;
    baseScores[i] =
      0.5 * matrix[off + IDX.seq_surprise] +
      0.3 * matrix[off + IDX.action_surprise] +
      0.2 * matrix[off + IDX.hour_surprise];
  }

  // --- 5. Fusion -----------------------------------------------------------
  onProgress(74, 'Fusing detector opinions…');
  await tick();
  const ifRank = rankNormalise(ifScores);
  const zRank = rankNormalise(zScores);
  const baseRank = rankNormalise(baseScores);
  const fusedDepth = fuseTailMax(ifRank, zRank, baseRank, n, opts.weights);

  const modelRisk = new Float64Array(n);
  const risk = new Float64Array(n);
  const ruleRisk = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    modelRisk[i] = depthToRisk(fusedDepth[i], opts.decades);
    const hits = ruleHits[i] || [];
    risk[i] = applyRules(modelRisk[i], hits);
    ruleRisk[i] = hits.length ? applyRules(0, hits) : 0;
  }

  // --- 6. Alerts -----------------------------------------------------------
  onProgress(84, 'Explaining findings…');
  await tick();
  const candidates = [];
  for (let i = 0; i < n; i++) {
    const hits = ruleHits[i] || [];
    const escalating = hits.some((h) => h.severity >= 4);
    if (risk[i] >= opts.threshold || escalating) candidates.push(i);
  }
  candidates.sort((a, b) => risk[b] - risk[a]);
  const selected = candidates.slice(0, opts.maxAlerts).sort((a, b) => a - b);

  const zRow = new Float64Array(D);
  const alerts = selected.map((i) => {
    robustZRow(matrix, i, D, colStats, FEATURE_DIRS, zRow);
    const attribution = attributeIsolation(forest, matrix, i);
    const hits = ruleHits[i] || [];
    const explanation = explainEvent({
      event: data[i], ctx: ctx[i], matrix, index: i,
      zRow: Float64Array.from(zRow), attribution, hits,
    });
    return buildAlert({
      event: data[i],
      index: i,
      risk: risk[i],
      hits,
      explanation,
      detectors: {
        iforest: ifRank[i],
        robustz: zRank[i],
        baseline: baseRank[i],
        blended: 1 - Math.pow(10, -fusedDepth[i]),
        modelRisk: modelRisk[i],
        ruleRisk: ruleRisk[i],
      },
    });
  });

  // --- 7. Correlation ------------------------------------------------------
  onProgress(92, 'Correlating alerts into incidents…');
  await tick();
  const incidents = buildIncidents(alerts, { gap: opts.incidentGapMinutes * 60_000 });

  // --- 8. Summaries and evaluation ----------------------------------------
  onProgress(97, 'Compiling report…');
  await tick();
  const identities = summariseIdentities(data, risk, alerts, profiles);
  const summary = summarise(data, risk, alerts, incidents, ruleHits);
  const labels = data.map((e) => e.label || 0);
  const hasLabels = labels.some((v) => v === 1);

  let evaluation = null;
  if (hasLabels) {
    const riskArr = Array.from(risk);
    evaluation = {
      threshold: opts.threshold,
      overall: evaluateScores(riskArr, labels, opts.threshold),
      ablation: ablation([
        { name: 'Isolation Forest alone', note: 'Joint outlier score only', scores: Array.from(ifScores) },
        { name: 'Robust z alone', note: 'Per-feature extremeness only', scores: Array.from(zScores) },
        { name: 'Behavioural model alone', note: 'Self-baseline surprisal only', scores: Array.from(baseScores) },
        { name: 'Rules alone', note: 'Signatures, no learning', scores: Array.from(ruleRisk) },
        { name: 'Ensemble (models only)', note: 'Rank fusion, no rules', scores: Array.from(modelRisk) },
        { name: 'ARGUS (ensemble + rules)', note: 'Shipping configuration', scores: riskArr },
      ], labels),
      campaigns: campaignDetection(data, risk, opts.threshold, options.campaigns || [], { modelRisk, ruleRisk }),
    };
  }

  onProgress(100, 'Done');
  return {
    events: data,
    matrix,
    ctx,
    profiles,
    priors,
    forest,
    colStats,
    risk,
    modelRisk,
    ruleRisk,
    fusedDepth,
    detectorScores: { iforest: ifScores, robustz: zScores, baseline: baseScores },
    detectorRanks: { iforest: ifRank, robustz: zRank, baseline: baseRank },
    ruleHits,
    alerts,
    incidents,
    identities,
    summary,
    evaluation,
    options: opts,
    elapsedMs: Math.round(performance.now() - t0),
  };
}

function summariseIdentities(events, risk, alerts, profiles) {
  const map = new Map();
  for (let i = 0; i < events.length; i++) {
    const e = events[i];
    let rec = map.get(e.actor);
    if (!rec) {
      rec = {
        actor: e.actor,
        actorType: e.actorType,
        events: 0,
        failures: 0,
        riskSum: 0,
        maxRisk: 0,
        firstSeen: e.ts,
        lastSeen: e.ts,
        ips: new Set(),
        countries: new Set(),
        actions: new Map(),
        alerts: 0,
        criticalAlerts: 0,
        labelled: 0,
        mfaPresent: 0,
        mfaAbsent: 0,
      };
      map.set(e.actor, rec);
    }
    rec.events++;
    if (e.mfa === true) rec.mfaPresent++;
    else if (e.mfa === false) rec.mfaAbsent++;
    rec.riskSum += risk[i];
    if (risk[i] > rec.maxRisk) rec.maxRisk = risk[i];
    if (e.outcome === 'failure') rec.failures++;
    rec.firstSeen = Math.min(rec.firstSeen, e.ts);
    rec.lastSeen = Math.max(rec.lastSeen, e.ts);
    rec.ips.add(e.ip);
    if (e.country) rec.countries.add(e.country);
    rec.actions.set(e.action, (rec.actions.get(e.action) || 0) + 1);
    if (e.label === 1) rec.labelled++;
  }
  for (const a of alerts) {
    const rec = map.get(a.actor);
    if (!rec) continue;
    rec.alerts++;
    if (a.severity === 'critical' || a.severity === 'high') rec.criticalAlerts++;
  }
  return [...map.values()].map((r) => {
    const profile = profiles.get(r.actor);
    return {
      actor: r.actor,
      actorType: r.actorType,
      events: r.events,
      failures: r.failures,
      failureRate: r.events ? r.failures / r.events : 0,
      meanRisk: r.events ? r.riskSum / r.events : 0,
      maxRisk: r.maxRisk,
      severity: severityFromRisk(r.maxRisk),
      firstSeen: r.firstSeen,
      lastSeen: r.lastSeen,
      ipCount: r.ips.size,
      countries: [...r.countries],
      alerts: r.alerts,
      criticalAlerts: r.criticalAlerts,
      labelled: r.labelled,
      // null when the source never reports MFA state (CloudTrail data events,
      // most CyberArk exports) — absence of evidence, not evidence of absence.
      mfaCoverage: r.mfaPresent + r.mfaAbsent ? r.mfaPresent / (r.mfaPresent + r.mfaAbsent) : null,
      topActions: [...r.actions.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6),
      distinctActions: r.actions.size,
      hourHistogram: profile ? Array.from(profile.hours) : new Array(24).fill(0),
    };
  }).sort((a, b) => b.maxRisk - a.maxRisk);
}

function summarise(events, risk, alerts, incidents, ruleHits) {
  const bySeverity = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
  for (const a of alerts) bySeverity[a.severity]++;

  const byTactic = new Map();
  for (const a of alerts) byTactic.set(a.tactic, (byTactic.get(a.tactic) || 0) + 1);

  const byRule = new Map();
  for (const hits of ruleHits) {
    if (!hits) continue;
    for (const h of hits) {
      const rec = byRule.get(h.id) || { id: h.id, name: h.name, tactic: h.tactic, count: 0, severity: h.severity };
      rec.count++;
      byRule.set(h.id, rec);
    }
  }

  const start = events[0].ts;
  const end = events[events.length - 1].ts;
  const span = Math.max(1, end - start);
  const buckets = 72;
  const timeline = Array.from({ length: buckets }, (_, i) => ({
    t: start + (span * i) / buckets,
    events: 0,
    alerts: 0,
    maxRisk: 0,
  }));
  for (let i = 0; i < events.length; i++) {
    const b = Math.min(buckets - 1, Math.floor(((events[i].ts - start) / span) * buckets));
    timeline[b].events++;
    if (risk[i] > timeline[b].maxRisk) timeline[b].maxRisk = risk[i];
  }
  for (const a of alerts) {
    const b = Math.min(buckets - 1, Math.floor(((a.ts - start) / span) * buckets));
    timeline[b].alerts++;
  }

  const services = new Map();
  for (const e of events) services.set(e.service, (services.get(e.service) || 0) + 1);

  return {
    events: events.length,
    identities: new Set(events.map((e) => e.actor)).size,
    alerts: alerts.length,
    incidents: incidents.length,
    alertRate: events.length ? (alerts.length / events.length) * 1000 : 0,
    bySeverity,
    byTactic: [...byTactic.entries()].sort((a, b) => b[1] - a[1]),
    byRule: [...byRule.values()].sort((a, b) => b.count - a.count),
    timeline,
    window: [start, end],
    services: [...services.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12),
    failureRate: events.filter((e) => e.outcome === 'failure').length / events.length,
    countries: [...new Set(events.map((e) => e.country).filter(Boolean))],
  };
}
