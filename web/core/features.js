/**
 * Streaming feature extraction and behavioural baselining.
 *
 * Design rule: **no lookahead**. Every feature for event *i* is computed from
 * events 0..i-1 only, exactly as a detector watching the stream would see it.
 * The original pipeline computed group statistics over the whole DataFrame,
 * which leaks the future into the present and inflates every metric that gets
 * reported afterwards.
 *
 * The one exception is deliberately marked: corpus-level *rarity* priors
 * (how common an operation is across the estate) are computed in a first pass.
 * That is a population prior, not per-event lookahead — it is the equivalent of
 * a detection engineer knowing that `cloudtrail:StopLogging` is rare
 * everywhere. It is applied identically to every event.
 *
 * Output is a packed Float64Array (n × D) plus a per-event context object that
 * the rule engine and the UI both read.
 */

import { isPrivateIp, sensitivityOf, isWriteAction, isDiscoveryAction, clientClass, haversineKm } from './schema.js';

const MIN = 60_000;
const HOUR = 3_600_000;

/**
 * Authentication events, the only ones that carry a *person's* location.
 *
 * Travel velocity is computed across sign-ins only. An API call's source
 * address tells you where the request came from, which for a role session,
 * an SDK on a build box, or a VPN egress is not where the human is — treating
 * every event as a location fix generates impossible-travel alerts every time
 * someone's laptop and their CI job act in the same minute.
 */
const AUTH_RX = /(signin:|ConsoleLogin|SignIn|Logon|user\.session\.start|okta:user\.authentication)/i;
const isAuthEvent = (action) => AUTH_RX.test(action);

/**
 * Feature catalogue. `dir: 1` means "larger is more suspicious", which is what
 * lets explanations say *why* a value contributed rather than just that it did.
 */
export const FEATURES = [
  { key: 'hour_sin', label: 'Hour (sin)', dir: 0, group: 'temporal', desc: 'Cyclical encoding of time of day.' },
  { key: 'hour_cos', label: 'Hour (cos)', dir: 0, group: 'temporal', desc: 'Cyclical encoding of time of day.' },
  { key: 'off_hours', label: 'Outside business hours', dir: 1, group: 'temporal', desc: 'Event fell outside 07:00–19:00 UTC.' },
  { key: 'weekend', label: 'Weekend', dir: 1, group: 'temporal', desc: 'Event fell on a Saturday or Sunday.' },
  { key: 'hour_surprise', label: 'Unusual hour for identity', dir: 1, group: 'temporal', desc: 'Surprisal of this hour against the identity’s own history.' },
  { key: 'log_gap', label: 'Gap since previous event', dir: 0, group: 'temporal', desc: 'log seconds since this identity was last seen.' },
  { key: 'dormancy', label: 'Dormant then active', dir: 1, group: 'temporal', desc: 'Identity silent for over a week before this event.' },

  { key: 'burst_1m', label: 'Events in last minute', dir: 1, group: 'volume', desc: 'Short-horizon activity spike.' },
  { key: 'burst_5m', label: 'Events in last 5 minutes', dir: 1, group: 'volume', desc: 'Medium-horizon activity spike.' },
  { key: 'burst_1h', label: 'Events in last hour', dir: 1, group: 'volume', desc: 'Sustained activity volume.' },
  { key: 'distinct_actions_1h', label: 'Distinct operations / hour', dir: 1, group: 'volume', desc: 'Breadth of API surface touched.' },
  { key: 'distinct_resources_1h', label: 'Distinct resources / hour', dir: 1, group: 'volume', desc: 'Breadth of objects touched.' },
  { key: 'discovery_ratio_1h', label: 'Enumeration ratio', dir: 1, group: 'volume', desc: 'Share of recent calls that are List/Describe/Get.' },

  { key: 'is_failure', label: 'Failed call', dir: 1, group: 'outcome', desc: 'The API call was denied or errored.' },
  { key: 'failure_rate_1h', label: 'Failure rate (1h)', dir: 1, group: 'outcome', desc: 'Share of this identity’s recent calls that failed.' },
  { key: 'fail_streak', label: 'Consecutive failures', dir: 1, group: 'outcome', desc: 'Run length of back-to-back failures.' },

  { key: 'action_new', label: 'First use of operation', dir: 1, group: 'novelty', desc: 'This identity has never called this operation before.' },
  { key: 'action_surprise', label: 'Operation rare for identity', dir: 1, group: 'novelty', desc: 'Surprisal of the operation against the identity’s own history.' },
  { key: 'action_rarity', label: 'Operation rare estate-wide', dir: 1, group: 'novelty', desc: 'Surprisal of the operation across all identities.' },
  { key: 'seq_surprise', label: 'Unexpected next step', dir: 1, group: 'novelty', desc: 'Markov surprisal of this operation given the previous one.' },
  { key: 'ip_new', label: 'New source address', dir: 1, group: 'novelty', desc: 'First time this identity used this IP.' },
  { key: 'ip_rarity', label: 'Rare source address', dir: 1, group: 'novelty', desc: 'How rarely this IP appears estate-wide.' },
  { key: 'country_new', label: 'New country', dir: 1, group: 'novelty', desc: 'First time this identity appeared from this country.' },
  { key: 'region_new', label: 'New region', dir: 1, group: 'novelty', desc: 'First time this identity used this cloud region.' },
  { key: 'ua_new', label: 'New client', dir: 1, group: 'novelty', desc: 'First time this identity used this user agent.' },
  { key: 'client_shift', label: 'Client class change', dir: 1, group: 'novelty', desc: 'Console user suddenly on a script, or vice versa.' },
  { key: 'resource_new', label: 'New resource', dir: 1, group: 'novelty', desc: 'First time this identity touched this resource.' },

  { key: 'travel_kmh', label: 'Implied travel speed', dir: 1, group: 'geo', desc: 'Speed required to move between consecutive login locations.' },
  { key: 'private_ip', label: 'Internal address', dir: -1, group: 'geo', desc: 'Source is an RFC1918 / internal address.' },

  { key: 'sensitivity', label: 'Operation sensitivity', dir: 1, group: 'privilege', desc: 'Blast radius of the operation if misused.' },
  { key: 'is_write', label: 'Mutating call', dir: 1, group: 'privilege', desc: 'The operation changes state.' },
  { key: 'no_mfa_priv', label: 'Privileged without MFA', dir: 1, group: 'privilege', desc: 'Sensitive operation on a session with no MFA.' },
  { key: 'actor_rarity', label: 'Rare identity', dir: 1, group: 'privilege', desc: 'How little this identity appears in the corpus.' },
  { key: 'experience', label: 'Identity history depth', dir: -1, group: 'privilege', desc: 'log of events seen from this identity so far.' },
];

export const FEATURE_KEYS = FEATURES.map((f) => f.key);
export const D = FEATURES.length;
const IDX = Object.fromEntries(FEATURE_KEYS.map((k, i) => [k, i]));

/** Live behavioural profile for one principal. */
class Profile {
  constructor(actor) {
    this.actor = actor;
    this.count = 0;
    this.firstTs = null;
    this.lastTs = null;
    this.actions = new Map();
    this.transitions = new Map();
    this.prevAction = null;
    this.ips = new Map();
    this.countries = new Map();
    this.regions = new Map();
    this.uas = new Map();
    this.resources = new Map();
    this.clientClasses = new Map();
    this.hours = new Int32Array(24);
    this.window = [];          // recent events within 1h: {ts, action, resource, failed, discovery}
    this.failStreak = 0;
    this.lastGeo = null;       // {lat, lon, ts}
    this.failures = 0;
    this.sensitiveOps = 0;
    this.mfaPresent = 0;       // events where a second factor was presented
    this.mfaAbsent = 0;
  }

  /**
   * True when *this* session lacks MFA and the identity normally has it.
   *
   * An identity that never uses MFA is a standing posture problem, reported on
   * its profile — not an anomaly to re-raise on every event it generates. The
   * anomaly is the drop-off.
   */
  mfaDropOff(mfa) {
    if (mfa !== false) return false;
    const total = this.mfaPresent + this.mfaAbsent;
    if (total < 5 || this.mfaPresent === 0) return false;
    return this.mfaAbsent / total < 0.2;
  }

  prune(now) {
    const cutoff = now - HOUR;
    let i = 0;
    while (i < this.window.length && this.window[i].ts < cutoff) i++;
    if (i) this.window.splice(0, i);
  }
}

const bump = (map, key) => {
  const v = (map.get(key) || 0) + 1;
  map.set(key, v);
  return v;
};

/**
 * Build the feature matrix.
 *
 * @param {Array} events sorted ascending by ts
 * @param {object} opts
 * @param {(e, i, profile, ctx) => void} opts.onEvent hook invoked after the
 *        feature row for event i is written, with the *pre-update* profile
 *        state; the rule engine uses it so both see identical context.
 * @returns {{matrix: Float64Array, ctx: Array, profiles: Map}}
 */
export function buildFeatures(events, opts = {}) {
  const n = events.length;
  const matrix = new Float64Array(n * D);
  const ctxArr = new Array(n);
  const profiles = new Map();

  // ---- Pass 1: population priors ------------------------------------------
  const actionCount = new Map();
  const ipCount = new Map();
  const actorCount = new Map();
  for (const e of events) {
    bump(actionCount, e.action);
    bump(ipCount, e.ip);
    bump(actorCount, e.actor);
  }
  const total = Math.max(1, n);
  const surprisal = (count) => -Math.log((count + 0.5) / (total + 1));

  // ---- Pass 2: streaming ---------------------------------------------------
  for (let i = 0; i < n; i++) {
    const e = events[i];
    let p = profiles.get(e.actor);
    if (!p) {
      p = new Profile(e.actor);
      profiles.set(e.actor, p);
    }
    p.prune(e.ts);

    const off = i * D;
    const date = new Date(e.ts);
    const hour = date.getUTCHours();
    const dow = date.getUTCDay();

    const gapMs = p.lastTs === null ? null : e.ts - p.lastTs;
    const gapSec = gapMs === null ? 86400 : Math.max(0, gapMs / 1000);

    // Window aggregates over the identity's last hour (past only).
    let burst1m = 0;
    let burst5m = 0;
    let failures1h = 0;
    let discovery1h = 0;
    const actSet = new Set();
    const resSet = new Set();
    for (const w of p.window) {
      if (e.ts - w.ts <= MIN) burst1m++;
      if (e.ts - w.ts <= 5 * MIN) burst5m++;
      if (w.failed) failures1h++;
      if (w.discovery) discovery1h++;
      actSet.add(w.action);
      resSet.add(w.resource);
    }
    const burst1h = p.window.length;
    const failureRate = burst1h ? failures1h / burst1h : 0;
    const discoveryRatio = burst1h ? discovery1h / burst1h : 0;

    const actionSeen = p.actions.get(e.action) || 0;
    const ipSeen = p.ips.get(e.ip) || 0;
    const countrySeen = e.country ? p.countries.get(e.country) || 0 : 1;
    const regionSeen = p.regions.get(e.region) || 0;
    const uaSeen = p.uas.get(e.userAgent) || 0;
    const resSeen = p.resources.get(e.resource) || 0;
    const cclass = clientClass(e.userAgent);
    const cclassSeen = p.clientClasses.get(cclass) || 0;

    // Markov surprisal of action | previous action, Laplace-smoothed.
    let seqSurprise = 0;
    if (p.prevAction) {
      const row = p.transitions.get(p.prevAction);
      const rowTotal = row ? row.total : 0;
      const c = row ? row.map.get(e.action) || 0 : 0;
      const vocab = Math.max(8, p.actions.size + 1);
      seqSurprise = -Math.log((c + 0.35) / (rowTotal + 0.35 * vocab));
    }

    const hourSurprise = p.count
      ? -Math.log((p.hours[hour] + 0.5) / (p.count + 12))
      : 0;

    // Implied travel speed since the identity's previous *authenticated* location.
    const authEvent = isAuthEvent(e.action);
    let travelKmh = 0;
    if (authEvent && e.lat !== null && e.lon !== null && p.lastGeo && e.ip !== p.lastGeo.ip) {
      const dt = (e.ts - p.lastGeo.ts) / HOUR;
      if (dt > 0.001) {
        const km = haversineKm(p.lastGeo.lat, p.lastGeo.lon, e.lat, e.lon);
        if (km > 150) travelKmh = km / dt;
      }
    }

    const sensitivity = sensitivityOf(e.action);
    const write = isWriteAction(e.action) ? 1 : 0;
    const discovery = isDiscoveryAction(e.action);
    const failed = e.outcome === 'failure' ? 1 : 0;
    const dormancy = gapMs !== null && gapMs > 7 * 86_400_000 ? 1 : 0;
    const isNew = p.count === 0;

    const row = matrix;
    row[off + IDX.hour_sin] = Math.sin((2 * Math.PI * hour) / 24);
    row[off + IDX.hour_cos] = Math.cos((2 * Math.PI * hour) / 24);
    row[off + IDX.off_hours] = hour < 7 || hour >= 19 ? 1 : 0;
    row[off + IDX.weekend] = dow === 0 || dow === 6 ? 1 : 0;
    row[off + IDX.hour_surprise] = hourSurprise;
    row[off + IDX.log_gap] = Math.log1p(gapSec);
    row[off + IDX.dormancy] = dormancy;

    row[off + IDX.burst_1m] = Math.log1p(burst1m);
    row[off + IDX.burst_5m] = Math.log1p(burst5m);
    row[off + IDX.burst_1h] = Math.log1p(burst1h);
    row[off + IDX.distinct_actions_1h] = Math.log1p(actSet.size);
    row[off + IDX.distinct_resources_1h] = Math.log1p(resSet.size);
    row[off + IDX.discovery_ratio_1h] = discoveryRatio;

    row[off + IDX.is_failure] = failed;
    row[off + IDX.failure_rate_1h] = failureRate;
    row[off + IDX.fail_streak] = Math.log1p(p.failStreak);

    row[off + IDX.action_new] = actionSeen === 0 && !isNew ? 1 : 0;
    row[off + IDX.action_surprise] = p.count
      ? -Math.log((actionSeen + 0.5) / (p.count + 0.5 * Math.max(8, p.actions.size + 1)))
      : 0;
    row[off + IDX.action_rarity] = surprisal(actionCount.get(e.action) || 0);
    row[off + IDX.seq_surprise] = seqSurprise;
    row[off + IDX.ip_new] = ipSeen === 0 && !isNew ? 1 : 0;
    row[off + IDX.ip_rarity] = surprisal(ipCount.get(e.ip) || 0);
    row[off + IDX.country_new] = countrySeen === 0 && !isNew ? 1 : 0;
    row[off + IDX.region_new] = regionSeen === 0 && !isNew ? 1 : 0;
    row[off + IDX.ua_new] = uaSeen === 0 && !isNew ? 1 : 0;
    row[off + IDX.client_shift] = cclassSeen === 0 && !isNew && p.clientClasses.size > 0 ? 1 : 0;
    row[off + IDX.resource_new] = resSeen === 0 && !isNew ? 1 : 0;

    row[off + IDX.travel_kmh] = Math.log1p(travelKmh);
    row[off + IDX.private_ip] = isPrivateIp(e.ip) ? 1 : 0;

    row[off + IDX.sensitivity] = sensitivity;
    row[off + IDX.is_write] = write;
    row[off + IDX.no_mfa_priv] = e.mfa === false && sensitivity >= 0.6 ? 1 : 0;
    row[off + IDX.actor_rarity] = surprisal(actorCount.get(e.actor) || 0);
    row[off + IDX.experience] = Math.log1p(p.count);

    const ctx = {
      hour,
      dow,
      gapSec,
      burst1m,
      burst5m,
      burst1h,
      distinctActions1h: actSet.size,
      distinctResources1h: resSet.size,
      discoveryRatio,
      failures1h,
      failureRate,
      failStreak: p.failStreak,
      newAction: actionSeen === 0 && !isNew,
      newIp: ipSeen === 0 && !isNew,
      newCountry: countrySeen === 0 && !isNew,
      newUa: uaSeen === 0 && !isNew,
      travelKmh,
      seqSurprise,
      hourSurprise,
      sensitivity,
      isWrite: !!write,
      isDiscovery: discovery,
      firstEverForActor: isNew,
      actorEventsSoFar: p.count,
      actionCountForActor: actionSeen,
      priorSensitiveOps: p.sensitiveOps,
      dormantRevival: dormancy === 1,
      mfaDropOff: p.mfaDropOff(e.mfa),
      mfaCoverage: p.mfaPresent + p.mfaAbsent ? p.mfaPresent / (p.mfaPresent + p.mfaAbsent) : null,
      prevAction: p.prevAction,
      clientClass: cclass,
      windowActions: p.window.slice(-40).map((w) => w.action),
      profile: p,
    };
    ctxArr[i] = ctx;

    if (opts.onEvent) opts.onEvent(e, i, ctx);

    // ---- Update state (strictly after scoring) ------------------------------
    if (p.prevAction) {
      let tr = p.transitions.get(p.prevAction);
      if (!tr) { tr = { map: new Map(), total: 0 }; p.transitions.set(p.prevAction, tr); }
      tr.map.set(e.action, (tr.map.get(e.action) || 0) + 1);
      tr.total++;
    }
    bump(p.actions, e.action);
    bump(p.ips, e.ip);
    if (e.country) bump(p.countries, e.country);
    bump(p.regions, e.region);
    bump(p.uas, e.userAgent);
    bump(p.resources, e.resource);
    bump(p.clientClasses, cclass);
    p.hours[hour]++;
    p.count++;
    p.prevAction = e.action;
    p.lastTs = e.ts;
    if (p.firstTs === null) p.firstTs = e.ts;
    p.failStreak = failed ? p.failStreak + 1 : 0;
    if (failed) p.failures++;
    if (e.mfa === true) p.mfaPresent++;
    else if (e.mfa === false) p.mfaAbsent++;
    if (sensitivity >= 0.7) p.sensitiveOps++;
    if (authEvent && e.lat !== null && e.lon !== null) p.lastGeo = { lat: e.lat, lon: e.lon, ts: e.ts, ip: e.ip };
    p.window.push({ ts: e.ts, action: e.action, resource: e.resource, failed: !!failed, discovery });
  }

  return { matrix, ctx: ctxArr, profiles, priors: { actionCount, ipCount, actorCount, total } };
}

/** Read one row of the packed matrix as a plain object (for explanations/UI). */
export function rowToObject(matrix, i) {
  const off = i * D;
  const out = {};
  for (let k = 0; k < D; k++) out[FEATURE_KEYS[k]] = matrix[off + k];
  return out;
}
