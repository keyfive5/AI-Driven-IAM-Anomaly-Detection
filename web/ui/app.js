/**
 * Shell: routing, the run lifecycle, and wiring the views to state.
 * Views are pure `render(state) → html` plus a `mount(root, ctx)` for events,
 * which keeps every screen re-derivable from state alone.
 */

import { state, subscribe, notify, loadSettings, saveSettings, loadSynthetic, run, setView } from './state.js';
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

async function runAnalysis() {
  progress.hidden = false;
  progressTitle.textContent = 'Analysing corpus…';
  progressBar.style.width = '0%';
  try {
    await run((pct, message) => {
      progressBar.style.width = `${pct}%`;
      progressNote.textContent = message;
    });
    renderView();
    const s = state.result.summary;
    toast(`${num(s.alerts)} alerts across ${num(s.incidents)} incidents (${s.alertRate.toFixed(1)} per 1k events)`);
  } catch (err) {
    console.error(err);
    toast(`Analysis failed: ${err.message}`, 'err');
  } finally {
    progress.hidden = true;
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
  loadSynthetic();
  await runAnalysis();
  saveSettings();
}

boot();
