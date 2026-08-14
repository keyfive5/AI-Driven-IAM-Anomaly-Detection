import { esc, num, pct, ts, duration, riskCell, sevTag, clip, on } from '../dom.js';
import { hourHeat, barList, sparkline } from '../charts.js';
import { state, select } from '../state.js';

export function render(s) {
  const r = s.result;
  if (!r) return `<div class="view"><div class="panel faint">Run an analysis first.</div></div>`;

  const selected = s.selection.identity
    ? r.identities.find((i) => i.actor === s.selection.identity)
    : null;

  return `<div class="view">
    <div class="view-head">
      <div>
        <h1>Identities</h1>
        <div class="sub">Each principal's own baseline — when they work, what they call, where from.
          This is the model the detectors compare every new event against.</div>
      </div>
      <input type="search" id="identityQuery" placeholder="Filter identities…" style="min-width:220px">
    </div>

    <div class="split">
      <div class="panel" style="padding:0">
        <div class="table-wrap" style="border:0;max-height:calc(100vh - 230px);overflow-y:auto">
          <table>
            <thead><tr>
              <th style="width:100px">Peak risk</th><th>Identity</th><th>Type</th>
              <th class="right">Events</th><th class="right">Alerts</th><th class="right">Denied</th>
              <th>Activity by hour</th>
            </tr></thead>
            <tbody id="identityRows">
              ${r.identities.map((i) => row(i, s)).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <div class="panel detail">${selected ? profile(selected, r) : `
        <div class="detail-empty">Select an identity to see its behavioural profile.</div>`}</div>
    </div>
  </div>`;
}

function row(i, s) {
  return `<tr class="clickable ${i.actor === s.selection.identity ? 'is-selected' : ''}"
      data-identity="${esc(i.actor)}" data-name="${esc(i.actor.toLowerCase())}">
    <td>${riskCell(i.maxRisk, i.severity)}</td>
    <td class="mono">${esc(clip(i.actor, 20))}</td>
    <td class="faint small">${esc(i.actorType)}</td>
    <td class="right mono small">${num(i.events)}</td>
    <td class="right mono small">${i.alerts ? `<strong>${num(i.alerts)}</strong>` : '<span class="faint">0</span>'}</td>
    <td class="right mono small">${pct(i.failureRate, 0)}</td>
    <td>${sparkline(i.hourHistogram, { width: 120, height: 18 })}</td>
  </tr>`;
}

function profile(i, r) {
  const alerts = r.alerts.filter((a) => a.actor === i.actor).sort((a, b) => b.risk - a.risk);
  return `<div>
    <div class="row" style="justify-content:space-between">
      <div>
        <h2 class="mono" style="font-size:15px">${esc(i.actor)}</h2>
        <div class="small faint">${esc(i.actorType)} · first seen ${esc(ts(i.firstSeen))}</div>
      </div>
      ${sevTag(i.severity)}
    </div>

    <div class="grid grid-3 mt" style="gap:9px">
      ${stat('Events', num(i.events))}
      ${stat('Alerts', num(i.alerts))}
      ${stat('Peak risk', Math.round(i.maxRisk))}
      ${stat('Operations', num(i.distinctActions))}
      ${stat('Addresses', num(i.ipCount))}
      ${stat('Denied', pct(i.failureRate, 0))}
    </div>

    ${i.mfaCoverage === null ? '' : i.mfaCoverage < 0.5 ? `
      <div class="panel mt" style="border-color:rgba(255,155,66,.35);padding:10px 12px">
        <div class="small"><strong style="color:var(--high)">Posture:</strong> only
          ${pct(i.mfaCoverage, 0)} of this identity's sessions presented a second factor.
          That is a standing hygiene finding, not an alert — ARGUS reports it here rather than
          raising it on every event, and alerts only when an identity that normally uses MFA
          stops.</div>
      </div>` : `<div class="small faint mt">MFA presented on ${pct(i.mfaCoverage, 0)} of sessions.</div>`}

    <h3 style="margin-top:16px">When this identity works</h3>
    ${hourHeat(i.hourHistogram)}

    <h3 style="margin-top:16px">What it normally calls</h3>
    ${barList(i.topActions.map(([action, count]) => ({ label: action, value: count })), { mono: true })}

    <h3 style="margin-top:16px">Where from</h3>
    <div class="row" style="gap:6px">
      ${i.countries.length ? i.countries.map((c) => `<span class="tag">${esc(c)}</span>`).join('')
        : '<span class="faint small">No geography in this log source.</span>'}
    </div>

    <h3 style="margin-top:16px">Alerts (${num(alerts.length)})</h3>
    ${alerts.length ? alerts.slice(0, 12).map((a) => `
      <div class="factor">
        <div class="factor-top">
          <span class="factor-name">${esc(a.headline)}</span>
          <span class="factor-z">${Math.round(a.risk)}</span>
        </div>
        <div class="factor-text mono">${esc(ts(a.ts))} · ${esc(clip(a.action, 34))}</div>
      </div>`).join('') : '<div class="faint small">Nothing raised for this identity.</div>'}
  </div>`;
}

function stat(label, value) {
  return `<div class="panel" style="padding:9px 11px;background:var(--bg-panel-2)">
    <div class="kpi-label">${esc(label)}</div>
    <div class="mono" style="font-size:17px">${value}</div>
  </div>`;
}

export function mount(root) {
  on(root, 'click', '[data-identity]', (ev, el) => select({ identity: el.dataset.identity }));

  const q = root.querySelector('#identityQuery');
  if (q) {
    q.addEventListener('input', () => {
      const term = q.value.trim().toLowerCase();
      for (const tr of root.querySelectorAll('#identityRows tr')) {
        tr.hidden = term ? !tr.dataset.name.includes(term) : false;
      }
    });
  }

  const selected = state.selection.identity;
  if (selected) {
    const node = root.querySelector(`tr[data-identity="${CSS.escape(selected)}"]`);
    if (node) node.scrollIntoView({ block: 'center' });
  }
}
