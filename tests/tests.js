/**
 * Test suite. Runs in the browser (open tests.html) or under Node
 * (`node web/tests/run.mjs`) — no test framework, no install step.
 *
 * These cover the parts where a silent bug would quietly corrupt every number
 * the console displays: parser fidelity, the no-lookahead guarantee, the
 * Isolation Forest's ordering, the risk scale's calibration, and the metric
 * implementations themselves (checked against hand-computed values).
 */

import { makeRng } from '../core/rng.js';
import { ingest, parseCloudTrail, parseOkta, parseEntraSignIn, parseCsv, detectFormat } from '../core/parse.js';
import { isPrivateIp, sensitivityOf, isWriteAction, haversineKm, severityFromRisk } from '../core/schema.js';
import { buildFeatures, FEATURE_KEYS, D } from '../core/features.js';
import { trainIsolationForest, scoreIsolationForest, cFactor } from '../core/iforest.js';
import { columnStats, rankNormalise, robustZScores } from '../core/stats.js';
import { rocAuc, prCurve, confusionAt } from '../core/evaluate.js';
import { tailDepth, depthToRisk, fuseTailMax, applyRules } from '../core/score.js';
import { evaluateRules } from '../core/rules.js';
import { generateCorpus } from '../core/generate.js';
import { analyse } from '../core/pipeline.js';
import { buildIncidents } from '../core/incidents.js';

const tests = [];
const test = (name, fn) => tests.push({ name, fn });

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion failed');
}
function near(actual, expected, tolerance, message) {
  assert(Math.abs(actual - expected) <= tolerance,
    `${message || 'value'}: expected ${expected} ±${tolerance}, got ${actual}`);
}

/* ------------------------------------------------------------------ rng --- */

test('seeded rng is reproducible and bounded', () => {
  const a = makeRng(42);
  const b = makeRng(42);
  for (let i = 0; i < 500; i++) {
    const v = a();
    assert(v === b(), 'same seed must yield the same stream');
    assert(v >= 0 && v < 1, 'values stay in [0,1)');
  }
  assert(makeRng(43)() !== makeRng(42)(), 'different seeds diverge');
});

test('weighted pick respects weights', () => {
  const rng = makeRng(9);
  let heavy = 0;
  for (let i = 0; i < 4000; i++) if (rng.weighted(['a', 'b'], [9, 1]) === 'a') heavy++;
  near(heavy / 4000, 0.9, 0.03, 'weighted share');
});

/* --------------------------------------------------------------- schema --- */

test('private address classification', () => {
  for (const ip of ['10.0.0.1', '192.168.1.1', '172.16.5.4', '127.0.0.1', '100.64.0.1']) {
    assert(isPrivateIp(ip), `${ip} should be private`);
  }
  for (const ip of ['8.8.8.8', '203.0.113.5', '172.32.0.1', '', null]) {
    assert(!isPrivateIp(ip), `${ip} should not be private`);
  }
});

test('write detection and sensitivity ordering', () => {
  assert(isWriteAction('iam:CreateUser'), 'CreateUser is a write');
  assert(!isWriteAction('iam:ListUsers'), 'ListUsers is a read');
  assert(!isWriteAction('s3:GetObject'), 'GetObject is a read');
  assert(sensitivityOf('cloudtrail:StopLogging') > sensitivityOf('s3:GetObject'),
    'disabling audit logs outranks reading an object');
  assert(sensitivityOf('iam:CreateAccessKey') > sensitivityOf('iam:ChangePassword'),
    'minting a key outranks a password change');
});

test('haversine distance is sane', () => {
  near(haversineKm(43.65, -79.38, 1.35, 103.82), 15000, 800, 'Toronto→Singapore');
  near(haversineKm(43.65, -79.38, 43.65, -79.38), 0, 0.001, 'same point');
});

test('severity thresholds', () => {
  assert(severityFromRisk(90) === 'critical');
  assert(severityFromRisk(75) === 'high');
  assert(severityFromRisk(55) === 'medium');
  assert(severityFromRisk(10) === 'info');
});

/* --------------------------------------------------------------- parsers -- */

test('CloudTrail parsing extracts identity, outcome and MFA', () => {
  const e = parseCloudTrail({
    eventTime: '2026-08-03T21:25:09Z',
    eventSource: 'iam.amazonaws.com',
    eventName: 'CreateUser',
    awsRegion: 'us-east-1',
    sourceIPAddress: '203.0.113.9',
    userAgent: 'aws-cli/2.13.5',
    userIdentity: { type: 'IAMUser', userName: 'Mary', sessionContext: { attributes: { mfaAuthenticated: 'false' } } },
    requestParameters: { userName: 'Richard' },
    errorCode: 'AccessDenied',
  });
  assert(e.actor === 'Mary', 'actor from userName');
  assert(e.action === 'iam:CreateUser', `action normalised, got ${e.action}`);
  assert(e.outcome === 'failure', 'errorCode implies failure');
  assert(e.mfa === false, 'mfaAuthenticated string parsed as boolean');
  assert(e.resource === 'Richard', 'resource from request parameters');
  assert(e.country === 'United States', 'region mapped to coarse geography');
});

test('Okta event types map onto canonical operations', () => {
  const mk = (eventType) => parseOkta({
    published: '2026-08-03T10:00:00Z',
    eventType,
    actor: { alternateId: 'a@b.c', type: 'User' },
    client: { ipAddress: '1.2.3.4', geographicalContext: {} },
    outcome: { result: 'SUCCESS' },
  });
  assert(mk('user.session.start').action === 'okta:SignIn', 'session start → SignIn');
  assert(mk('user.mfa.factor.deactivate').action === 'okta:DeactivateMFADevice', 'MFA removal canonicalised');
  assert(mk('user.account.privilege.grant').action === 'okta:GrantPrivilege', 'privilege grant canonicalised');
});

test('Entra sign-in failure codes and geography', () => {
  const e = parseEntraSignIn({
    createdDateTime: '2026-08-03T02:00:00Z',
    userPrincipalName: 'a@b.c',
    status: { errorCode: 50126 },
    location: { city: 'Lagos', countryOrRegion: 'NG', geoCoordinates: { latitude: 6.52, longitude: 3.38 } },
    authenticationRequirement: 'singleFactorAuthentication',
  });
  assert(e.outcome === 'failure', 'non-zero errorCode is a failure');
  assert(e.country === 'NG' && e.lat === 6.52, 'geography preserved');
  assert(e.mfa === false, 'single factor recorded');
});

test('format sniffing picks the right parser', () => {
  assert(detectFormat([{ eventVersion: '1.08', eventTime: 'x' }]).format === 'aws');
  assert(detectFormat([{ eventType: 'user.session.start', actor: {}, client: {} }]).format === 'okta');
  assert(detectFormat([{ userPrincipalName: 'a@b', createdDateTime: 'x' }]).format === 'entra');
  assert(detectFormat([{ protoPayload: {} }]).format === 'gcp');
  assert(detectFormat([{ what: 1 }]).format === 'generic');
});

test('CSV reader handles quotes, commas and newlines', () => {
  const rows = parseCsv('a,b\n1,"x, y"\n2,"line\nbreak"\n3,"say ""hi"""');
  assert(rows.length === 3, `expected 3 rows, got ${rows.length}`);
  assert(rows[0].b === 'x, y', 'embedded comma');
  assert(rows[1].b === 'line\nbreak', 'embedded newline');
  assert(rows[2].b === 'say "hi"', 'escaped quotes');
});

test('ingest sorts by time and survives malformed records', () => {
  const { events, skipped } = ingest(JSON.stringify([
    { time: '2026-08-03T10:00:00Z', user: 'b', event: 'Two' },
    { time: 'not-a-date', user: 'x', event: 'Bad' },
    { time: '2026-08-03T09:00:00Z', user: 'a', event: 'One' },
  ]));
  assert(events.length === 2, `2 valid events, got ${events.length}`);
  assert(skipped === 1, 'the undated record is skipped, not fatal');
  assert(events[0].ts < events[1].ts, 'output is time-ordered');
});

/* -------------------------------------------------------------- features -- */

test('features never look ahead', () => {
  // Truncating the stream must not change the features of earlier events:
  // if it does, some statistic is being computed over the whole corpus.
  const { events } = generateCorpus({ seed: 5, days: 3, users: 6, campaigns: [] });
  const cut = Math.floor(events.length / 2);
  const full = buildFeatures(events);
  const partial = buildFeatures(events.slice(0, cut));

  // Corpus-level rarity priors are documented as population statistics and do
  // change with the sample; every other feature must be identical.
  const priorFeatures = new Set(['action_rarity', 'ip_rarity', 'actor_rarity']);
  const checked = FEATURE_KEYS.map((k, i) => [k, i]).filter(([k]) => !priorFeatures.has(k));

  for (let i = 0; i < cut; i++) {
    for (const [key, f] of checked) {
      const a = full.matrix[i * D + f];
      const b = partial.matrix[i * D + f];
      assert(Math.abs(a - b) < 1e-9,
        `feature "${key}" of event ${i} changed when later events were removed (${a} vs ${b})`);
    }
  }
});

test('novelty features fire on first use and then settle', () => {
  const base = { actor: 'u', actorType: 'Human', ip: '9.9.9.9', region: 'r', country: 'X',
    userAgent: 'ua', outcome: 'success', resource: 'res', service: 's', mfa: true, lat: 0, lon: 0, label: 0 };
  const events = [
    { ...base, id: '1', ts: 1000, action: 's:A' },
    { ...base, id: '2', ts: 2000, action: 's:A' },
    { ...base, id: '3', ts: 3000, action: 's:B' },
  ];
  const { matrix } = buildFeatures(events);
  const idx = FEATURE_KEYS.indexOf('action_new');
  assert(matrix[0 * D + idx] === 0, 'the very first event cannot be "new" — there is no baseline yet');
  assert(matrix[1 * D + idx] === 0, 'a repeat is not new');
  assert(matrix[2 * D + idx] === 1, 'a genuinely unseen operation is new');
});

test('travel velocity only uses authentication events', () => {
  const base = { actor: 'u', actorType: 'Human', region: 'r', userAgent: 'ua',
    outcome: 'success', resource: 'r', service: 's', mfa: true, label: 0 };
  const idx = FEATURE_KEYS.indexOf('travel_kmh');
  const far = { lat: 1.35, lon: 103.82, country: 'SG', ip: '5.5.5.5' };
  const home = { lat: 43.65, lon: -79.38, country: 'CA', ip: '1.1.1.1' };

  const apiOnly = buildFeatures([
    { ...base, ...home, id: '1', ts: 0, action: 's:GetThing' },
    { ...base, ...far, id: '2', ts: 600000, action: 's:GetThing' },
  ]);
  assert(apiOnly.matrix[1 * D + idx] === 0,
    'two API calls from different egress addresses are not a person teleporting');

  const signIns = buildFeatures([
    { ...base, ...home, id: '1', ts: 0, action: 'signin:ConsoleLogin' },
    { ...base, ...far, id: '2', ts: 600000, action: 'signin:ConsoleLogin' },
  ]);
  assert(signIns.matrix[1 * D + idx] > 0, 'two sign-ins 15,000 km apart in 10 minutes must register');
});

/* -------------------------------------------------------- isolation forest */

test('c(n) matches the published values', () => {
  near(cFactor(2), 1, 1e-9, 'c(2)');
  near(cFactor(256), 10.244, 0.01, 'c(256)');
  assert(cFactor(1) === 0, 'c(1) is 0');
});

test('isolation forest ranks planted outliers above the cluster', () => {
  const rng = makeRng(3);
  const n = 600;
  const dims = 4;
  const matrix = new Float64Array(n * dims);
  for (let i = 0; i < n; i++) {
    const outlier = i >= n - 6;
    for (let f = 0; f < dims; f++) {
      matrix[i * dims + f] = outlier ? rng.normal(9, 0.3) : rng.normal(0, 1);
    }
  }
  const model = trainIsolationForest(matrix, n, dims, { trees: 100, sampleSize: 256, seed: 11 });
  const scores = scoreIsolationForest(model, matrix, n);
  const ranked = [...scores.keys()].sort((a, b) => scores[b] - scores[a]).slice(0, 6);
  assert(ranked.every((i) => i >= n - 6), `top 6 scores should be the planted outliers, got ${ranked}`);
  assert(scores.every((s) => s > 0 && s <= 1), 'scores stay in (0,1]');
});

test('isolation forest is deterministic for a fixed seed', () => {
  const rng = makeRng(1);
  const n = 200;
  const matrix = new Float64Array(n * 3);
  for (let i = 0; i < n * 3; i++) matrix[i] = rng.normal(0, 1);
  const a = scoreIsolationForest(trainIsolationForest(matrix, n, 3, { seed: 4 }), matrix, n);
  const b = scoreIsolationForest(trainIsolationForest(matrix, n, 3, { seed: 4 }), matrix, n);
  for (let i = 0; i < n; i++) assert(a[i] === b[i], 'same seed, same scores');
});

/* ---------------------------------------------------------------- stats --- */

test('median and MAD resist contamination', () => {
  const n = 101;
  const matrix = new Float64Array(n);
  for (let i = 0; i < 100; i++) matrix[i] = 5;
  matrix[100] = 100000;
  const { med, mad } = columnStats(matrix, n, 1);
  assert(med[0] === 5, `median should ignore the outlier, got ${med[0]}`);
  assert(mad[0] < 1, 'MAD stays small despite a huge outlier');
});

test('rank normalisation is uniform and ties share a rank', () => {
  const scores = Float64Array.from([5, 1, 3, 3, 9]);
  const ranks = rankNormalise(scores);
  assert(ranks[1] === 0, 'smallest maps to 0');
  assert(ranks[4] === 1, 'largest maps to 1');
  assert(ranks[2] === ranks[3], 'ties receive the same rank');
});

/* ----------------------------------------------------------- risk scale --- */

test('risk scale is calibrated to tail percentiles', () => {
  const n = 100000;
  near(depthToRisk(tailDepth(0.90, n)), 28.6, 0.5, 'top 10% ≈ 29');
  near(depthToRisk(tailDepth(0.99, n)), 57.1, 0.5, 'top 1% ≈ 57');
  near(depthToRisk(tailDepth(0.999, n)), 85.7, 0.5, 'top 0.1% ≈ 86');
  assert(depthToRisk(tailDepth(0.5, n)) < 10, 'the median event is not risky');
  assert(depthToRisk(tailDepth(1, 20)) <= 100, 'risk is capped at 100');
});

test('fusion takes the deepest tail, not the average', () => {
  const one = Float64Array.from([0.999]);
  const dull = Float64Array.from([0.5]);
  const weights = { iforest: 1, robustz: 1, baseline: 1 };
  const fused = fuseTailMax(one, dull, dull, 1, weights);
  near(fused[0], tailDepth(0.999, 1), 1e-9,
    'one confident detector must not be diluted by two ambivalent ones');
});

test('rules raise a floor and never lower a score', () => {
  const hits = [{ id: 'r', risk: 90, severity: 5 }];
  assert(applyRules(20, hits) >= 90, 'a rule lifts a low model score to its floor');
  assert(applyRules(97, hits) >= 97, 'a rule never drags a high model score down');
  assert(applyRules(40, []) === 40, 'no rules, no change');
  assert(applyRules(95, [{ risk: 90 }, { risk: 80 }, { risk: 70 }]) <= 100, 'risk stays capped');
});

/* -------------------------------------------------------------- metrics --- */

test('ROC AUC matches hand-computed values', () => {
  assert(rocAuc([1, 2, 3, 4], [0, 0, 1, 1]) === 1, 'perfect separation is 1.0');
  assert(rocAuc([4, 3, 2, 1], [0, 0, 1, 1]) === 0, 'perfectly inverted is 0.0');
  near(rocAuc([1, 2, 3, 4], [0, 1, 0, 1]), 0.75, 1e-9, 'interleaved case');
  assert(rocAuc([1, 1, 1, 1], [0, 0, 1, 1]) === 0.5, 'all ties is a coin flip');
  assert(rocAuc([1, 2], [0, 0]) === null, 'undefined without both classes');
});

test('precision–recall and confusion matrix agree', () => {
  const scores = [90, 80, 70, 60, 50, 40];
  const labels = [1, 1, 0, 1, 0, 0];
  const pr = prCurve(scores, labels);
  near(pr.ap, (1 + 1 + 0.75) / 3, 1e-9, 'average precision');
  const cm = confusionAt(scores, labels, 60);
  assert(cm.tp === 3 && cm.fp === 1 && cm.fn === 0 && cm.tn === 2, 'confusion counts');
  near(cm.precision, 0.75, 1e-9, 'precision');
  near(cm.recall, 1, 1e-9, 'recall');
  near(cm.alertsPerThousand, (4 / 6) * 1000, 1e-9, 'alert budget');
});

/* ---------------------------------------------------------------- rules --- */

test('routine administration does not alert, the same act at 03:00 does', () => {
  const routine = {
    action: 'iam:CreateUser', actor: 'admin', resource: 'newhire', mfa: true,
    outcome: 'success', ip: '10.0.0.5', country: 'Canada', actorType: 'Human',
  };
  const calmCtx = {
    hour: 11, newAction: false, newIp: false, newCountry: false, travelKmh: 0, mfaDropOff: false,
    dormantRevival: false, burst5m: 1, failStreak: 0, sensitivity: 0.85, isWrite: true,
    isDiscovery: false, actionCountForActor: 30, actorEventsSoFar: 400, priorSensitiveOps: 40,
    windowActions: [], distinctActions1h: 3, discoveryRatio: 0, burst1h: 5, burst1m: 1,
  };
  assert(evaluateRules(routine, calmCtx).length === 0,
    'an identity administrator onboarding someone at 11:00 is not an incident');

  const oddCtx = { ...calmCtx, hour: 3, newIp: true, mfaDropOff: true };
  const hits = evaluateRules({ ...routine, mfa: false, ip: '203.0.113.9' }, oddCtx);
  assert(hits.length > 0, 'the same call at 03:00 from a new address must alert');
});

test('audit-log tampering alerts unconditionally', () => {
  const ctx = {
    hour: 11, newAction: false, newIp: false, newCountry: false, travelKmh: 0, mfaDropOff: false,
    dormantRevival: false, burst5m: 1, failStreak: 0, sensitivity: 1, isWrite: true, isDiscovery: false,
    actionCountForActor: 50, actorEventsSoFar: 900, priorSensitiveOps: 90, windowActions: [],
    distinctActions1h: 2, discoveryRatio: 0, burst1h: 3, burst1m: 1,
  };
  const hits = evaluateRules({
    action: 'cloudtrail:StopLogging', actor: 'admin', resource: 'org-trail',
    mfa: true, outcome: 'success', ip: '10.0.0.5', actorType: 'Human',
  }, ctx);
  assert(hits.some((h) => h.id === 'logging_tamper'),
    'disabling the audit trail is wrong however routine it looks');
});

/* ----------------------------------------------------------- correlation -- */

test('incidents split on the configured time gap', () => {
  const mk = (ts, risk) => ({
    id: `A${ts}`, actor: 'u', actorType: 'Human', ts, risk, severity: 'high',
    tactic: 'Discovery', rules: [], ip: '1.1.1.1', country: null, campaign: null,
    label: 0, headline: 'x', factors: [],
  });
  const alerts = [mk(0, 70), mk(60_000, 70), mk(10 * 60_000, 70)];
  assert(buildIncidents(alerts, { gap: 5 * 60_000 }).length === 2, 'a 10-minute gap splits at 5');
  assert(buildIncidents(alerts, { gap: 30 * 60_000 }).length === 1, 'one incident at a 30-minute gap');
});

/* ------------------------------------------------------------- pipeline --- */

test('end to end: campaigns are detected and the alert budget stays sane', async () => {
  const { events, meta } = generateCorpus({ seed: 20260813, days: 10, users: 25 });
  assert(events.length > 3000, `expected a substantial corpus, got ${events.length}`);
  const result = await analyse(events, { campaigns: meta.campaigns }, () => {});

  assert(result.alerts.length > 0, 'something must be raised');
  assert(result.summary.alertRate < 40,
    `alert budget must stay reviewable, got ${result.summary.alertRate.toFixed(1)} per 1k`);
  assert(result.evaluation.overall.auc > 0.9,
    `AUC should be strong on labelled data, got ${result.evaluation.overall.auc}`);

  const detected = result.evaluation.campaigns.filter((c) => c.detected).length;
  assert(detected === result.evaluation.campaigns.length,
    `every injected campaign should be caught, got ${detected}/${result.evaluation.campaigns.length}`);

  for (const alert of result.alerts) {
    assert(alert.risk >= 0 && alert.risk <= 100, 'risk stays in range');
    assert(alert.headline && alert.headline.length > 0, 'every alert explains itself');
    assert(alert.incidentId, 'every alert belongs to an incident');
  }
});

test('analysis is reproducible for a fixed seed', async () => {
  const { events, meta } = generateCorpus({ seed: 4242, days: 6, users: 15 });
  const a = await analyse(events, { campaigns: meta.campaigns }, () => {});
  const b = await analyse(events, { campaigns: meta.campaigns }, () => {});
  assert(a.alerts.length === b.alerts.length, 'same corpus, same alert count');
  for (let i = 0; i < a.alerts.length; i++) {
    assert(a.alerts[i].risk === b.alerts[i].risk, `alert ${i} scored differently between runs`);
  }
});

test('an unlabelled corpus produces no evaluation instead of fake metrics', async () => {
  const { events } = generateCorpus({ seed: 7, days: 4, users: 10, campaigns: [] });
  const result = await analyse(events, {}, () => {});
  assert(result.evaluation === null,
    'without ground truth the system must decline to report precision, not invent it');
});

/* ------------------------------------------------------------------ run --- */

export async function runTests(report = () => {}) {
  let passed = 0;
  const failures = [];
  for (const t of tests) {
    const started = performance.now();
    try {
      await t.fn();
      passed++;
      report({ name: t.name, ok: true, ms: performance.now() - started });
    } catch (err) {
      failures.push({ name: t.name, error: err });
      report({ name: t.name, ok: false, ms: performance.now() - started, error: err.message });
    }
  }
  return { total: tests.length, passed, failures };
}

export { tests };
