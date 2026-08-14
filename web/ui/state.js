/**
 * Application state: the loaded corpus, the last analysis, and the analyst's
 * own tuning (thresholds, detector weights, per-identity suppressions).
 *
 * Suppressions are applied *after* scoring rather than by re-running the
 * engine, so switching a noisy rule off for one identity is instant and
 * reversible — which is how tuning actually happens during a shift.
 */

import { analyse, DEFAULT_OPTIONS } from '../core/pipeline.js';
import { generateCorpus, CAMPAIGN_IDS } from '../core/generate.js';
import { buildIncidents } from '../core/incidents.js';
import { applyRules } from '../core/score.js';
import { severityFromRisk } from '../core/schema.js';

const STORAGE_KEY = 'argus.settings';

/**
 * Bump whenever the *meaning* of a saved setting changes, not just its default.
 *
 * Learned the hard way: detector weights used to be mixing proportions for a
 * weighted mean and are now trust factors on tail depth. A browser holding the
 * old values silently ran a badly mis-scaled engine and every metric on screen
 * was wrong, with nothing to indicate why. Stored settings are an API.
 */
// v3: the persisted corpus size is what trapped users behind a stuck progress
// bar, so stored generator settings from before the size guard are discarded
// rather than silently re-applied on the next load.
const SETTINGS_VERSION = 3;

/**
 * Sized so the first analysis finishes quickly on a modest laptop or phone.
 * The estate is still large enough for baselines to mean something; the Data
 * tab raises it for anyone who wants a heavier run, and it is what the
 * progress overlay's escape hatch resets to.
 */
export const DEFAULT_GENERATOR = { seed: 20260813, days: 10, users: 30 };

export const state = {
  view: 'overview',
  events: [],
  meta: null,
  result: null,
  running: false,
  error: null,
  options: { ...DEFAULT_OPTIONS },
  generator: { ...DEFAULT_GENERATOR, campaigns: [...CAMPAIGN_IDS] },
  suppressions: [],
  selection: { alertId: null, incidentId: null, identity: null },
  filters: {
    severity: new Set(),
    tactic: null,
    query: '',
    minRisk: 0,
    onlyTruth: false,
  },
};

const listeners = new Set();
export function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }
export function notify() { for (const fn of listeners) fn(state); }

/* ------------------------------------------------------------ persistence -- */

export function loadSettings() {
  try {
    // Clear anything written by an incompatible earlier build.
    localStorage.removeItem('argus.settings.v1');
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (saved.version !== SETTINGS_VERSION) {
      localStorage.removeItem(STORAGE_KEY);
      return;
    }
    if (saved.options) Object.assign(state.options, saved.options);
    if (saved.options?.weights) state.options.weights = { ...DEFAULT_OPTIONS.weights, ...saved.options.weights };
    if (saved.generator) Object.assign(state.generator, saved.generator);
    if (Array.isArray(saved.suppressions)) state.suppressions = saved.suppressions;
  } catch {
    /* corrupt settings are not worth a crash — fall back to defaults */
  }
}

export function saveSettings() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      version: SETTINGS_VERSION,
      options: state.options,
      generator: state.generator,
      suppressions: state.suppressions,
    }));
  } catch { /* private mode / quota — tuning simply will not persist */ }
}

/* ---------------------------------------------------------------- loading -- */

export function loadSynthetic() {
  const { events, meta } = generateCorpus({ ...state.generator });
  state.events = events;
  state.meta = meta;
  state.selection = { alertId: null, incidentId: null, identity: null };
  return { events, meta };
}

/**
 * Above this the analysis is minutes of arithmetic and hundreds of megabytes
 * in the tab. A real export can be far larger, so take the most recent window
 * and say so — silently analysing a slice, or silently wedging the browser,
 * are both worse than a clear statement.
 */
export const MAX_EVENTS = 200_000;

export function loadEvents(events, meta) {
  let loaded = events;
  let truncated = 0;
  if (events.length > MAX_EVENTS) {
    truncated = events.length - MAX_EVENTS;
    loaded = events.slice(-MAX_EVENTS);   // events arrive time-ordered
  }
  state.events = loaded;
  state.meta = { ...meta, truncated };
  state.selection = { alertId: null, incidentId: null, identity: null };
  return { loaded: loaded.length, truncated };
}

/* --------------------------------------------------------------- analysis -- */

export async function run(onProgress, signal) {
  if (!state.events.length) throw new Error('Load a dataset first.');
  state.running = true;
  state.error = null;
  notify();
  try {
    state.result = await analyse(state.events, {
      ...state.options,
      signal,
      campaigns: state.meta?.campaigns || [],
    }, onProgress);
    applySuppressions();
  } catch (err) {
    state.error = err;
    throw err;
  } finally {
    state.running = false;
    notify();
  }
  return state.result;
}

/* ----------------------------------------------------------- suppressions -- */

export function isSuppressed(ruleId, actor) {
  return state.suppressions.some((s) => s.rule === ruleId && (s.actor === '*' || s.actor === actor));
}

export function addSuppression(ruleId, actor, ruleName) {
  if (isSuppressed(ruleId, actor)) return 0;
  state.suppressions.push({ rule: ruleId, actor, name: ruleName });
  const before = state.result?.alerts.length ?? 0;
  applySuppressions();
  saveSettings();
  return before - (state.result?.alerts.length ?? 0);
}

export function removeSuppression(index) {
  state.suppressions.splice(index, 1);
  applySuppressions();
  saveSettings();
}

/**
 * Recompute alerts and incidents from the unmodified scoring output, minus
 * whatever the analyst has muted. An alert whose only reason was a muted rule
 * drops out unless the models alone still clear the threshold.
 */
export function applySuppressions() {
  const result = state.result;
  if (!result) return;
  const threshold = state.options.threshold;

  const alerts = [];
  for (const base of result.alertsRaw ?? result.alerts) {
    const kept = base.rules.filter((r) => !isSuppressed(r.id, base.actor));
    if (kept.length === base.rules.length && !state.suppressions.length) {
      alerts.push(base);
      continue;
    }
    const modelRisk = base.detectors.modelRisk;
    const risk = applyRules(modelRisk, kept);
    const escalating = kept.some((r) => r.severity >= 4);
    if (risk < threshold && !escalating) continue;
    alerts.push({
      ...base,
      rules: kept,
      risk: Math.round(risk * 10) / 10,
      severity: severityFromRisk(risk),
      headline: kept.length ? kept.slice().sort((a, b) => b.risk - a.risk)[0].name : base.headlineModel || base.headline,
      suppressedCount: base.rules.length - kept.length,
    });
  }

  if (!result.alertsRaw) result.alertsRaw = result.alerts;
  result.alerts = alerts;
  result.incidents = buildIncidents(alerts, { gap: state.options.incidentGapMinutes * 60_000 });
  result.summary.alerts = alerts.length;
  result.summary.incidents = result.incidents.length;
  result.summary.alertRate = result.summary.events ? (alerts.length / result.summary.events) * 1000 : 0;
  const bySeverity = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
  for (const a of alerts) bySeverity[a.severity]++;
  result.summary.bySeverity = bySeverity;
}

/* ------------------------------------------------------------------ views -- */

export function setView(view) {
  if (state.view === view) return;
  state.view = view;
  notify();
}

export function select(patch) {
  Object.assign(state.selection, patch);
  notify();
}

export function alertById(id) {
  return state.result?.alerts.find((a) => a.id === id) || null;
}
