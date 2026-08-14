/**
 * Reproducible benchmark. Produces the numbers quoted in the README.
 *
 *   node web/tests/benchmark.mjs [seeds...]
 *
 * The corpus window is pinned to a fixed end date so the output does not drift
 * with the day you happen to run it — the default synthetic estate in the app
 * ends "today", which is right for a demo and wrong for a citation.
 */

import { generateCorpus, CAMPAIGN_IDS } from '../core/generate.js';
import { analyse } from '../core/pipeline.js';

const END = Date.UTC(2026, 7, 14);            // pinned: 14 Aug 2026
const SEEDS = process.argv.slice(2).map(Number).filter(Boolean);
const seeds = SEEDS.length ? SEEDS : [20260813, 77, 991, 5150, 31337];

const pad = (s, n) => String(s).padEnd(n);
const rpad = (s, n) => String(s).padStart(n);
const f3 = (v) => (v === null || v === undefined ? '  –  ' : v.toFixed(3));
const pc = (v) => (v === null || v === undefined ? '  –  ' : `${(v * 100).toFixed(1)}%`);

const agg = new Map();
const campaignAgg = new Map();
let totalEvents = 0;
let totalMalicious = 0;
let totalAlerts = 0;
let totalMs = 0;

for (const seed of seeds) {
  const { events, meta } = generateCorpus({ seed, days: 14, users: 40, campaigns: CAMPAIGN_IDS, endTs: END });
  const result = await analyse(events, { campaigns: meta.campaigns }, () => {});
  const ev = result.evaluation;

  totalEvents += events.length;
  totalMalicious += events.filter((e) => e.label === 1).length;
  totalAlerts += result.alerts.length;
  totalMs += result.elapsedMs;

  for (const row of ev.ablation) {
    const rec = agg.get(row.name) || { auc: 0, ap: 0, f1: 0, p: 0, r: 0, rate: 0, n: 0 };
    rec.auc += row.auc ?? 0;
    rec.ap += row.ap ?? 0;
    rec.f1 += row.f1 ?? 0;
    rec.p += row.precision ?? 0;
    rec.r += row.recall ?? 0;
    rec.rate += row.alertsPerThousand ?? 0;
    rec.n++;
    agg.set(row.name, rec);
  }

  for (const c of ev.campaigns) {
    const rec = campaignAgg.get(c.id) || { name: c.name, difficulty: c.difficulty, runs: 0, detected: 0, coverage: 0, latency: 0, rulesOnly: 0 };
    rec.runs++;
    if (c.detected) rec.detected++;
    rec.coverage += c.coverage;
    rec.latency += c.latencyMs ?? 0;
    if (c.attribution === 'rules only') rec.rulesOnly++;
    campaignAgg.set(c.id, rec);
  }

  const cm = ev.overall.confusion;
  console.log(`seed ${rpad(seed, 9)}  ${rpad(events.length, 6)} events  ` +
    `${rpad(result.alerts.length, 4)} alerts (${(result.summary.alertRate).toFixed(1)}/1k)  ` +
    `AUC ${f3(ev.overall.auc)}  AP ${f3(ev.overall.ap)}  ` +
    `P ${pc(cm.precision)}  R ${pc(cm.recall)}  ${rpad(result.elapsedMs, 5)} ms`);
}

console.log(`\nCorpus: ${seeds.length} seeds · ${totalEvents.toLocaleString()} events · ` +
  `${totalMalicious} malicious (${((totalMalicious / totalEvents) * 100).toFixed(2)}%) · ` +
  `${(totalMs / seeds.length).toFixed(0)} ms mean analysis time\n`);

console.log('DETECTOR ABLATION — mean over seeds, each variant at its own best-F1 threshold');
console.log(`${pad('configuration', 30)} ${rpad('AUC', 6)} ${rpad('AP', 6)} ${rpad('F1', 6)} ${rpad('prec', 7)} ${rpad('recall', 7)} ${rpad('alerts/1k', 10)}`);
console.log('─'.repeat(76));
for (const [name, r] of agg) {
  console.log(`${pad(name, 30)} ${rpad(f3(r.auc / r.n), 6)} ${rpad(f3(r.ap / r.n), 6)} ` +
    `${rpad(f3(r.f1 / r.n), 6)} ${rpad(pc(r.p / r.n), 7)} ${rpad(pc(r.r / r.n), 7)} ${rpad((r.rate / r.n).toFixed(1), 10)}`);
}

console.log('\nCAMPAIGN DETECTION — at the default threshold of 60');
console.log(`${pad('campaign', 24)} ${pad('difficulty', 10)} ${rpad('detected', 9)} ${rpad('coverage', 9)} ${rpad('latency', 9)} rules-only`);
console.log('─'.repeat(76));
for (const [id, r] of campaignAgg) {
  console.log(`${pad(id, 24)} ${pad(r.difficulty, 10)} ${rpad(`${r.detected}/${r.runs}`, 9)} ` +
    `${rpad(pc(r.coverage / r.runs), 9)} ${rpad(`${(r.latency / r.runs / 1000).toFixed(0)}s`, 9)} ${r.rulesOnly}/${r.runs}`);
}
