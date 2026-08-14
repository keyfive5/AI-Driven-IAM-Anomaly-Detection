import { esc, num, ts, tsFull, fixed, clip, riskCell, sevTag, on, download, copyText, toast } from '../dom.js';
import { barList } from '../charts.js';
import { state, select, addSuppression, alertById } from '../state.js';
import { TACTICS } from '../../core/schema.js';

const PAGE = 200;
let page = 1;
let sortKey = 'risk';

export function render(s) {
  const r = s.result;
  if (!r) return `<div class="view"><div class="panel faint">Run an analysis first.</div></div>`;

  const filtered = filterAlerts(r.alerts, s.filters);
  const sorted = sortAlerts(filtered, sortKey);
  const shown = sorted.slice(0, page * PAGE);
  const selected = s.selection.alertId ? alertById(s.selection.alertId) : null;

  return `<div class="view">
    <div class="view-head">
      <div>
        <h1>Alert triage</h1>
        <div class="sub">Every row states why it was raised. Sorted by risk, which is a tail
          percentile: 57 ≈ the oddest 1% of events, 86 ≈ the oddest 0.1%.</div>
      </div>
      <div class="row">
        <button class="btn btn-sm" data-export="csv">Export CSV</button>
        <button class="btn btn-sm" data-export="json">Export JSON</button>
      </div>
    </div>

    <div class="filters">
      <input type="search" id="alertQuery" placeholder="Search identity, operation, address, resource…"
        value="${esc(s.filters.query)}" autocomplete="off">
      ${['critical', 'high', 'medium', 'low'].map((sev) => `
        <button class="chip ${s.filters.severity.has(sev) ? 'is-on' : ''}" data-sev="${sev}">${sev}</button>`).join('')}
      <select id="tacticFilter">
        <option value="">All tactics</option>
        ${TACTICS.map((t) => `<option value="${esc(t)}" ${s.filters.tactic === t ? 'selected' : ''}>${esc(t)}</option>`).join('')}
      </select>
      <label class="switch"><input type="checkbox" id="onlyTruth" ${s.filters.onlyTruth ? 'checked' : ''}
        ${r.evaluation ? '' : 'disabled'}> <span>Ground truth only</span></label>
      <select id="sortKey">
        <option value="risk" ${sortKey === 'risk' ? 'selected' : ''}>Sort: risk</option>
        <option value="time" ${sortKey === 'time' ? 'selected' : ''}>Sort: newest</option>
        <option value="actor" ${sortKey === 'actor' ? 'selected' : ''}>Sort: identity</option>
      </select>
      <span class="small faint" style="margin-left:auto">${num(filtered.length)} of ${num(r.alerts.length)} alerts</span>
    </div>

    <div class="split">
      <div class="panel" style="padding:0;overflow:hidden">
        <div class="table-wrap" style="border:0;max-height:calc(100vh - 250px);overflow-y:auto">
          <table>
            <thead><tr>
              <th style="width:104px">Risk</th><th>Time</th><th>Identity</th><th>Operation</th>
              <th>Resource</th><th>Source</th><th>Tactic</th>
            </tr></thead>
            <tbody>
              ${shown.map((a) => `
                <tr class="clickable ${a.id === s.selection.alertId ? 'is-selected' : ''} ${a.label === 1 ? 'is-truth' : ''}"
                    data-alert="${esc(a.id)}">
                  <td>${riskCell(a.risk, a.severity)}</td>
                  <td class="mono small nowrap">${esc(ts(a.ts))}</td>
                  <td class="mono">${esc(clip(a.actor, 18))}</td>
                  <td class="mono small">${esc(clip(a.action, 30))}</td>
                  <td class="small dim">${esc(clip(a.resource, 26))}</td>
                  <td class="small dim nowrap">${esc(a.country || a.region || '–')}</td>
                  <td><span class="tag tag-tactic">${esc(a.tactic)}</span></td>
                </tr>`).join('') || `<tr><td colspan="7" class="faint small center" style="padding:34px">
                  No alerts match these filters.</td></tr>`}
            </tbody>
          </table>
        </div>
        ${sorted.length > shown.length ? `<div class="pager">
          <button class="btn btn-sm" id="loadMore">Show ${num(Math.min(PAGE, sorted.length - shown.length))} more</button>
          <span class="small faint">${num(shown.length)} / ${num(sorted.length)}</span>
        </div>` : ''}
      </div>

      <div class="panel detail">${selected ? detail(selected, s) : `
        <div class="detail-empty">Select an alert to see the evidence behind it.</div>`}</div>
    </div>
  </div>`;
}

function detail(a, s) {
  const raw = s.result.events[a.eventIndex];
  const d = a.detectors;
  return `<div>
    <div class="row" style="justify-content:space-between;align-items:flex-start">
      <div>
        <div class="gauge">
          <span class="gauge-num" style="color:var(--${a.severity})">${Math.round(a.risk)}</span>
          <span class="gauge-scale">risk / 100<br>${sevTag(a.severity)}</span>
        </div>
      </div>
      <div class="row" style="gap:6px">
        <button class="btn btn-sm btn-ghost" data-copy="${esc(a.id)}">Copy</button>
        ${a.incidentId ? `<button class="btn btn-sm btn-ghost" data-open-incident="${esc(a.incidentId)}">Incident</button>` : ''}
      </div>
    </div>

    <h2 style="margin:12px 0 4px;font-size:14px">${esc(a.headline)}</h2>
    <div class="small dim">${esc(a.actor)} · ${esc(a.action)} · ${esc(tsFull(a.ts))}</div>
    ${a.label === 1 ? `<div class="tag tag-truth mt">ground truth: ${esc(a.campaign)} · ${esc(a.stage || '')}</div>` : ''}

    ${a.rules.length ? `<h3 style="margin-top:16px">Detections</h3>
      ${a.rules.slice().sort((x, y) => y.risk - x.risk).map((rule) => `
        <div class="rule-card sev-${rule.severity}">
          <div class="rule-title">
            <span>${esc(rule.name)}</span>
            <span class="mono faint small">${esc(rule.technique)}</span>
          </div>
          <div class="rule-detail">${esc(rule.detail)}</div>
          <div class="row" style="margin-top:7px;gap:6px">
            <span class="tag tag-tactic">${esc(rule.tactic)}</span>
            <span class="tag">floor ${rule.risk}</span>
            <button class="btn btn-sm btn-ghost" data-suppress="${esc(rule.id)}" data-actor="${esc(a.actor)}"
              title="Mute this rule for this identity only">Mute for ${esc(clip(a.actor, 14))}</button>
          </div>
        </div>`).join('')}` : `
      <div class="note mt">No signature matched. This was raised by the models alone — the
        behaviour itself is out of character for this identity.</div>`}

    <h3 style="margin-top:16px">Why it scored</h3>
    <div>${a.factors.length ? a.factors.map((f) => `
      <div class="factor">
        <div class="factor-top">
          <span class="factor-name">${esc(f.label)}</span>
          <span class="factor-z">${f.z >= 0 ? '+' : ''}${fixed(f.z, 1)}σ</span>
        </div>
        <div class="factor-meter"><i style="width:${Math.min(100, f.strength * 42).toFixed(0)}%"></i></div>
        <div class="factor-text">${esc(f.text)}</div>
      </div>`).join('') : '<div class="faint small">Nothing individually extreme — this was flagged on the joint pattern.</div>'}
    </div>

    <h3 style="margin-top:16px">Detector agreement</h3>
    ${barList([
      { label: 'Isolation Forest', value: d.iforest, display: `${(d.iforest * 100).toFixed(1)}%` },
      { label: 'Robust z', value: d.robustz, display: `${(d.robustz * 100).toFixed(1)}%` },
      { label: 'Behavioural', value: d.baseline, display: `${(d.baseline * 100).toFixed(1)}%` },
      { label: 'Fused rank', value: d.blended, display: `${(d.blended * 100).toFixed(1)}%` },
    ])}
    <div class="small faint" style="margin-top:6px">Percentile within this corpus. Model-only risk
      ${Math.round(d.modelRisk)}${a.rules.length ? `, lifted to ${Math.round(a.risk)} by the rule floor` : ''}.</div>

    <h3 style="margin-top:16px">Event</h3>
    <dl class="kv">
      <dt>Identity</dt><dd>${esc(a.actor)} <span class="faint">(${esc(a.actorType)})</span></dd>
      <dt>Operation</dt><dd>${esc(a.action)}</dd>
      <dt>Resource</dt><dd>${esc(a.resource)}</dd>
      <dt>Outcome</dt><dd>${esc(a.outcome)}${a.outcome === 'failure' ? ` <span class="faint">${esc(raw?.errorCode || '')}</span>` : ''}</dd>
      <dt>Source</dt><dd>${esc(a.ip)}${a.city ? ` · ${esc(a.city)}` : ''}${a.country ? `, ${esc(a.country)}` : ''}</dd>
      <dt>Region</dt><dd>${esc(a.region)}</dd>
      <dt>MFA</dt><dd>${a.mfa === null ? 'not reported' : a.mfa ? 'present' : 'absent'}</dd>
      <dt>Client</dt><dd>${esc(clip(a.userAgent, 70))}</dd>
      <dt>Incident</dt><dd>${esc(a.incidentId || '–')}</dd>
    </dl>

    <details style="margin-top:12px">
      <summary class="small dim" style="cursor:pointer">Raw source record</summary>
      <pre class="code" style="margin-top:8px">${esc(JSON.stringify(raw?.raw ?? raw, null, 2)?.slice(0, 4000) || '–')}</pre>
    </details>
  </div>`;
}

export function filterAlerts(alerts, f) {
  const q = f.query.trim().toLowerCase();
  return alerts.filter((a) => {
    if (f.severity.size && !f.severity.has(a.severity)) return false;
    if (f.tactic && a.tactic !== f.tactic) return false;
    if (f.onlyTruth && a.label !== 1) return false;
    if (a.risk < f.minRisk) return false;
    if (q) {
      const hay = `${a.actor} ${a.action} ${a.resource} ${a.ip} ${a.country || ''} ${a.headline} ${a.tactic}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function sortAlerts(alerts, key) {
  const copy = alerts.slice();
  if (key === 'time') copy.sort((a, b) => b.ts - a.ts);
  else if (key === 'actor') copy.sort((a, b) => a.actor.localeCompare(b.actor) || b.risk - a.risk);
  else copy.sort((a, b) => b.risk - a.risk);
  return copy;
}

export function mount(root, ctx) {
  on(root, 'click', '[data-alert]', (ev, el) => select({ alertId: el.dataset.alert }));

  on(root, 'click', '[data-sev]', (ev, el) => {
    const sev = el.dataset.sev;
    if (state.filters.severity.has(sev)) state.filters.severity.delete(sev);
    else state.filters.severity.add(sev);
    page = 1;
    ctx.rerender();
  });

  on(root, 'click', '#loadMore', () => { page++; ctx.rerender(); });

  on(root, 'click', '[data-suppress]', (ev, el) => {
    const removed = addSuppression(el.dataset.suppress, el.dataset.actor, el.textContent);
    toast(removed > 0
      ? `Muted for ${el.dataset.actor} — ${removed} alert${removed === 1 ? '' : 's'} withdrawn`
      : 'Rule muted for this identity');
    state.selection.alertId = null;
    ctx.rerender();
  });

  on(root, 'click', '[data-copy]', (ev, el) => {
    const a = alertById(el.dataset.copy);
    if (a) copyText(JSON.stringify(a, (k, v) => (k === 'raw' ? undefined : v), 2));
  });

  on(root, 'click', '[data-open-incident]', (ev, el) => {
    select({ incidentId: el.dataset.openIncident });
    ctx.setView('incidents');
  });

  on(root, 'click', '[data-export]', (ev, el) => {
    const alerts = sortAlerts(filterAlerts(state.result.alerts, state.filters), sortKey);
    if (el.dataset.export === 'json') {
      download('argus-alerts.json', JSON.stringify(alerts.map(stripRaw), null, 2));
    } else {
      download('argus-alerts.csv', toCsv(alerts), 'text/csv');
    }
    toast(`Exported ${alerts.length} alerts`);
  });

  const query = root.querySelector('#alertQuery');
  if (query) {
    let timer;
    query.addEventListener('input', () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        state.filters.query = query.value;
        page = 1;
        ctx.rerender();
        const again = document.querySelector('#alertQuery');
        if (again) { again.focus(); again.setSelectionRange(again.value.length, again.value.length); }
      }, 220);
    });
  }

  const tactic = root.querySelector('#tacticFilter');
  if (tactic) tactic.addEventListener('change', () => {
    state.filters.tactic = tactic.value || null;
    page = 1;
    ctx.rerender();
  });

  const truth = root.querySelector('#onlyTruth');
  if (truth) truth.addEventListener('change', () => {
    state.filters.onlyTruth = truth.checked;
    page = 1;
    ctx.rerender();
  });

  const sort = root.querySelector('#sortKey');
  if (sort) sort.addEventListener('change', () => { sortKey = sort.value; ctx.rerender(); });
}

const stripRaw = (a) => ({ ...a, factors: a.factors.map(({ key, label, z, text }) => ({ key, label, z, text })) });

function toCsv(alerts) {
  const head = ['id', 'timestamp', 'risk', 'severity', 'identity', 'identity_type', 'operation', 'resource',
    'source_ip', 'country', 'outcome', 'tactic', 'technique', 'rules', 'headline', 'incident', 'ground_truth'];
  const rows = alerts.map((a) => [
    a.id, new Date(a.ts).toISOString(), a.risk, a.severity, a.actor, a.actorType, a.action, a.resource,
    a.ip, a.country || '', a.outcome, a.tactic, a.technique || '', a.rules.map((r) => r.id).join('|'),
    a.headline, a.incidentId || '', a.label === 1 ? 'malicious' : '',
  ]);
  const cell = (v) => {
    const str = String(v ?? '');
    return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
  };
  return [head, ...rows].map((r) => r.map(cell).join(',')).join('\n');
}
