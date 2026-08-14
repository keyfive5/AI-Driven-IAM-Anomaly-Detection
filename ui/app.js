/**
 * Shell: routing, the run lifecycle, and wiring the views to state.
 * Views are pure `render(state) → html` plus a `mount(root, ctx)` for events,
 * which keeps every screen re-derivable from state alone.
 */

import { state, subscribe, loadSettings, saveSettings, loadSynthetic, run, setView, DEFAULT_GENERATOR } from './state.js';
import { $, num, duration, toast, esc } from './dom.js';
import * as overview from './views/overview.js';
import * as alerts from './views/alerts.js';
import * as incidents from './views/incidents.js';
import * as identities from './views/identities.js';
import * as explore from './views/explore.js';
import * as evaluate from './views/evaluate.js';
import * as data from './views/data.js';
import * as engine from './views/engine.js';

const VIEWS = { overview, alerts, incidents, identities, explore, evaluate, data, engine };

const stage = $('#stage');
const progress = $('#progress');
const progressBar = $('#progressBar');
const progressNote = $('#progressNote');
const progressTitle = $('#progressTitle');

const ctx = {
  rerender: () => renderView(),
  setView,
  runAnalysis: () => runAnalysis(),
};

function renderView() {
  const view = VIEWS[state.view] || overview;
  stage.innerHTML = view.render(state);
  view.mount?.(stage, ctx);
  stage.scrollTop = state.view === 'incidents' && state.selection.incidentId ? stage.scrollTop : 0;

  for (const el of document.querySelectorAll('.rail-item[data-view]')) {
    el.classList.toggle('is-active', el.dataset.view === state.view);
  }
  renderChrome();
}

function renderChrome() {
  const r = state.result;
  $('#badgeAlerts').textContent = r ? num(r.alerts.length) : '';
  $('#badgeIncidents').textContent = r ? num(r.incidents.length) : '';

  const meta = $('#datasetMeta');
  if (!r) {
    meta.innerHTML = '<span class="faint">no corpus loaded</span>';
    return;
  }
  const s = r.summary;
  const [start, end] = s.window;
  meta.innerHTML = `
    <span class="tag">${esc(state.meta?.name || 'synthetic estate')}</span>
    <span>${num(s.events)} events</span><span class="faint">·</span>
    <span>${num(s.identities)} identities</span><span class="faint">·</span>
    <span>${duration(end - start)}</span><span class="faint">·</span>
    <span>${num(s.alerts)} alerts</span><span class="faint">·</span>
    <span>${s.alertRate.toFixed(1)}/1k</span><span class="faint">·</span>
    <span>${num(r.elapsedMs)} ms</span>`;
}

/**
 * Run the pipeline behind the progress overlay.
 *
 * The overlay must never become a dead end. Analysis is arithmetic on the main
 * thread, so a slow device, a huge upload or an outright bug all present the
 * same way — a bar that stops moving — and the first version offered no way
 * out of that state and no clue what was happening. So:
 *
 *   - a watchdog reports which stage stalled, and after that offers an escape;
 *   - Cancel always dismisses the overlay and leaves the previous result usable;
 *   - failures are shown *in* the overlay with a retry, not as a toast that
 *     disappears while the page is still blank.
 */
let analysisRunning = false;

async function runAnalysis() {
  if (analysisRunning) return;
  analysisRunning = true;

  const started = performance.now();
  let lastStage = 'starting';
  let lastTick = performance.now();
  let cancelled = false;
  const controller = new AbortController();

  progress.hidden = false;
  progress.classList.remove('is-stalled', 'is-error');
  progressTitle.textContent = 'Analysing corpus…';
  progressBar.style.width = '0%';
  progressNote.textContent = 'preparing…';
  setEscape(null);

  // Two separate failure modes, both of which used to look identical:
  //   - a genuine stall (no progress at all), and
  //   - a run that is simply long, where the bar moves but the tab is
  //     unusable for a minute and the user has no idea how long is left.
  // Elapsed time is always shown past a few seconds, and an escape appears for
  // either condition. Chunked progress keeps resetting the stall timer, so a
  // long run must be caught on total elapsed time, not on quiet alone.
  // One click has to land somewhere usable. Halving 60 days × 200 identities
  // still leaves ~91,000 events and another long wait, so the escape resets to
  // the known-good default rather than stepping down.
  const offerSmaller = () => setEscape({
    label: 'Cancel and load the default corpus',
    action: () => {
      cancelled = true;
      controller.abort();          // stop the running pipeline, not just the dialog
      state.generator.days = DEFAULT_GENERATOR.days;
      state.generator.users = DEFAULT_GENERATOR.users;
      saveSettings();
      dismiss();
      loadSynthetic();
      // Let the aborted run unwind at its next checkpoint before starting the
      // replacement, so the two never share the main thread.
      setTimeout(() => runAnalysis(), 0);
    },
  });

  const watchdog = setInterval(() => {
    const now = performance.now();
    const quiet = now - lastTick;
    const elapsed = now - started;
    if (elapsed < 4000) return;

    const secs = Math.round(elapsed / 1000);
    if (quiet >= 8000) {
      progress.classList.add('is-stalled');
      progressTitle.textContent = 'Still working…';
      progressNote.textContent =
        `"${lastStage}" has not reported for ${Math.round(quiet / 1000)}s (${secs}s total).`;
      offerSmaller();
      return;
    }

    progressTitle.textContent = `Analysing ${num(state.events.length)} events — ${secs}s`;
    if (elapsed >= 12000) {
      progressNote.textContent =
        `${lastStage} · a corpus this size takes a while, and the tab stays busy until it finishes.`;
      offerSmaller();
    }
  }, 1000);

  const dismiss = () => {
    clearInterval(watchdog);
    progress.hidden = true;
    progress.classList.remove('is-stalled', 'is-error');
    setEscape(null);
    analysisRunning = false;
  };

  try {
    await run((pct, message) => {
      if (controller.signal.aborted) return;
      lastStage = message;
      lastTick = performance.now();
      progress.classList.remove('is-stalled');
      progressBar.style.width = `${pct}%`;
      progressNote.textContent = message;
      // Leave the title and escape alone once the watchdog has taken over, so
      // a long run does not flicker between "Analysing…" and its own status.
      if (lastTick - started < 4000) progressTitle.textContent = 'Analysing corpus…';
    }, controller.signal);
    if (cancelled) return;
    renderView();
    const s = state.result.summary;
    toast(`${num(s.alerts)} alerts across ${num(s.incidents)} incidents (${s.alertRate.toFixed(1)} per 1k events)`);
    dismiss();
  } catch (err) {
    clearInterval(watchdog);
    // A cancelled run is an expected outcome, not a failure to report.
    if (cancelled || err.name === 'AnalysisCancelled') { analysisRunning = false; return; }
    console.error(err);
    // Show the failure where the user is already looking.
    progress.classList.add('is-error');
    progressTitle.textContent = 'Analysis failed';
    progressNote.textContent = `${err.message} — stage: ${lastStage}`;
    setEscape({
      label: 'Dismiss',
      action: () => { dismiss(); if (!state.result) setView('data'); },
    });
    analysisRunning = false;
  }
}

/** Render (or clear) the overlay's escape-hatch button. */
function setEscape(escape) {
  const host = $('#progressEscape');
  if (!host) return;
  host.innerHTML = '';
  if (!escape) { host.hidden = true; return; }
  host.hidden = false;
  const btn = document.createElement('button');
  btn.className = 'btn btn-sm';
  btn.textContent = escape.label;
  btn.addEventListener('click', escape.action);
  host.appendChild(btn);
}

function bindChrome() {
  for (const el of document.querySelectorAll('.rail-item[data-view]')) {
    el.addEventListener('click', () => setView(el.dataset.view));
  }
  $('#btnRun').addEventListener('click', () => runAnalysis());
  $('#btnDocs').addEventListener('click', () => setView('engine'));

  window.addEventListener('hashchange', () => {
    const view = location.hash.replace('#', '');
    if (VIEWS[view]) setView(view);
  });

  document.addEventListener('keydown', (ev) => {
    if (ev.target.matches('input, textarea, select')) return;
    const map = { 1: 'overview', 2: 'alerts', 3: 'incidents', 4: 'identities', 5: 'explore', 6: 'evaluate', 7: 'data' };
    if (map[ev.key]) setView(map[ev.key]);
    if (ev.key === '/') {
      ev.preventDefault();
      setView('alerts');
      setTimeout(() => document.querySelector('#alertQuery')?.focus(), 30);
    }
  });
}

async function boot() {
  loadSettings();
  bindChrome();
  subscribe(() => {
    if (location.hash.replace('#', '') !== state.view) {
      history.replaceState(null, '', `#${state.view}`);
    }
    renderView();
  });

  const initial = location.hash.replace('#', '');
  if (VIEWS[initial]) state.view = initial;

  renderView();
  loadSynthetic();
  await runAnalysis();
  saveSettings();
}

boot();
