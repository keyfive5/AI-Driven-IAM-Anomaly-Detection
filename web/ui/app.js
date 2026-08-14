/**
 * Shell: routing, the run lifecycle, and wiring the views to state.
 * Views are pure `render(state) → html` plus a `mount(root, ctx)` for events,
 * which keeps every screen re-derivable from state alone.
 */

import { state, subscribe, loadSettings, saveSettings, run, cancelRun, setView, DEFAULT_GENERATOR } from './state.js';
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
  runAnalysis: (opts) => runAnalysis(opts),
};

function renderView() {
  const view = VIEWS[state.view] || overview;
  // Before the first result exists there is no dashboard to show, but the rail,
  // the Data tab and the engine documentation all work — so show a placeholder
  // for the data-driven views rather than locking the whole console.
  const needsResult = ['overview', 'alerts', 'incidents', 'identities', 'explore', 'evaluate'];
  if (!state.result && state.running && needsResult.includes(state.view)) {
    stage.innerHTML = bootingPlaceholder();
    renderChrome();
    for (const el of document.querySelectorAll('.rail-item[data-view]')) {
      el.classList.toggle('is-active', el.dataset.view === state.view);
    }
    return;
  }
  stage.innerHTML = view.render(state);
  view.mount?.(stage, ctx);
  stage.scrollTop = state.view === 'incidents' && state.selection.incidentId ? stage.scrollTop : 0;

  for (const el of document.querySelectorAll('.rail-item[data-view]')) {
    el.classList.toggle('is-active', el.dataset.view === state.view);
  }
  renderChrome();
}

function bootingPlaceholder() {
  return `<div class="view"><div class="panel booting">
    <div class="skeleton" style="width:34%"></div>
    <div class="skeleton" style="width:52%"></div>
    <div class="skeleton" style="width:22%"></div>
    <p class="dim" style="margin-top:20px">Building the first analysis. The console is already
      usable — <button class="btn btn-sm btn-ghost" data-goto="data">Data &amp; tuning</button>
      and <button class="btn btn-sm btn-ghost" data-goto="engine">Engine</button> work now, and the
      status bar above can cancel this run at any time.</p>
  </div></div>`;
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
 * Run the pipeline and report on it.
 *
 * The engine runs in a worker, so this function only ever touches the DOM:
 * the console stays interactive for the whole run, cancelling is immediate,
 * and the status strip never covers anything. The earlier design blocked the
 * page behind a modal for the duration, which meant a slow corpus and a broken
 * one were indistinguishable and neither left the analyst anywhere to go.
 */
let analysisRunning = false;

async function runAnalysis({ generate = false } = {}) {
  if (analysisRunning) return;
  analysisRunning = true;

  const started = performance.now();
  let lastStage = 'starting';
  let lastTick = performance.now();
  let cancelled = false;

  progress.hidden = false;
  progress.classList.remove('is-stalled', 'is-error');
  progressTitle.textContent = 'Analysing…';
  progressBar.style.width = '0%';
  progressNote.textContent = 'starting';

  const finish = () => {
    clearInterval(ticker);
    progress.hidden = true;
    progress.classList.remove('is-stalled', 'is-error');
    setActions([]);
    analysisRunning = false;
  };

  // Cancel is offered from the first second, not after a timeout. Stopping a
  // run the user no longer wants should never require waiting for a watchdog.
  const cancel = () => {
    cancelled = true;
    cancelRun();
    finish();
    toast('Analysis cancelled');
    renderView();
  };

  const resetToDefault = () => {
    cancelled = true;
    cancelRun();
    state.generator.days = DEFAULT_GENERATOR.days;
    state.generator.users = DEFAULT_GENERATOR.users;
    saveSettings();
    finish();
    runAnalysis({ generate: true });
  };

  setActions([{ label: 'Cancel', action: cancel }]);

  const ticker = setInterval(() => {
    if (cancelled) return;
    const now = performance.now();
    const elapsed = now - started;
    const quiet = now - lastTick;
    const secs = Math.round(elapsed / 1000);
    if (elapsed < 3000) return;

    const size = state.events.length
      ? `${num(state.events.length)} events — `
      : '';
    progressTitle.textContent = `Analysing ${size}${secs}s`;

    if (quiet >= 10000) {
      progress.classList.add('is-stalled');
      progressNote.textContent = `"${lastStage}" has not reported for ${Math.round(quiet / 1000)}s`;
    }
    // A long run is a size problem, so offer the size remedy alongside cancel.
    if (elapsed >= 10000) {
      setActions([
        { label: 'Use the default corpus', action: resetToDefault },
        { label: 'Cancel', action: cancel },
      ]);
    }
  }, 1000);

  try {
    await run((pct, message) => {
      if (cancelled) return;
      lastStage = message;
      lastTick = performance.now();
      progress.classList.remove('is-stalled');
      progressBar.style.width = `${pct}%`;
      progressNote.textContent = message;
    }, { generate });
    if (cancelled) return;
    renderView();
    const s = state.result.summary;
    toast(`${num(s.alerts)} alerts across ${num(s.incidents)} incidents (${s.alertRate.toFixed(1)} per 1k events)`);
    finish();
  } catch (err) {
    clearInterval(ticker);
    if (cancelled) { analysisRunning = false; return; }
    console.error(err);
    progress.classList.add('is-error');
    progressTitle.textContent = 'Analysis failed';
    progressNote.textContent = `${err.message} — stage: ${lastStage}`;
    setActions([
      { label: 'Use the default corpus', action: resetToDefault },
      { label: 'Dismiss', action: () => { finish(); renderView(); } },
    ]);
    analysisRunning = false;
  }
}

/** Render the status strip's action buttons. */
function setActions(actions) {
  const host = $('#progressEscape');
  if (!host) return;
  host.innerHTML = '';
  for (const a of actions) {
    const btn = document.createElement('button');
    btn.className = 'btn btn-sm';
    btn.textContent = a.label;
    btn.addEventListener('click', a.action);
    host.appendChild(btn);
  }
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
  // The worker generates the corpus as well as analysing it, so the shell
  // paints and stays interactive from the very first frame.
  await runAnalysis({ generate: true });
  saveSettings();
}

boot();
