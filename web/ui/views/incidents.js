import { esc, num, ts, tsFull, duration, riskCell, sevTag, clip, on, download, copyText, toast } from '../dom.js';
import { chainStrip, SEV_COLOR } from '../charts.js';
import { state, select } from '../state.js';

export function render(s) {
  const r = s.result;
  if (!r) return `<div class="view"><div class="panel faint">Run an analysis first.</div></div>`;
  const open = s.selection.incidentId;

  return `<div class="view">
    <div class="view-head">
      <div>
        <h1>Incidents</h1>
        <div class="sub">Alerts grouped by identity and contiguous activity, then ordered along the
          kill chain. An intrusion is a shape, not a row — this is the view that shows the shape.</div>
      </div>
      <div class="row small faint">
        <span class="tag">${num(r.incidents.length)} incidents</span>
        <span class="tag">gap ${s.options.incidentGapMinutes} min</span>
        <button class="btn btn-sm" data-export-incidents>Export report</button>
      </div>
    </div>

    ${r.incidents.length ? r.incidents.map((inc) => card(inc, inc.id === open, r)).join('')
      : '<div class="panel faint">No incidents at this threshold.</div>'}
  </div>`;
}

function card(inc, isOpen, result) {
  return `<div class="incident" id="inc-${esc(inc.id)}">
    <div class="incident-head" data-toggle="${esc(inc.id)}">
      <span class="incident-rail" style="background:${SEV_COLOR[inc.severity]}"></span>
      <div style="flex:1;min-width:0">
        <div class="row" style="gap:9px">
          ${sevTag(inc.severity)}
          <strong class="mono">${esc(inc.actor)}</strong>
          <span class="tag">${esc(inc.id)}</span>
          <span class="faint small">${num(inc.alertCount)} alerts · ${duration(inc.durationMs)} · ${esc(ts(inc.start))}</span>
          ${inc.campaigns.length ? `<span class="tag tag-truth">ground truth: ${esc(inc.campaigns.join(', '))}</span>` : ''}
          ${inc.links.length ? `<span class="tag">linked: ${esc(inc.links[0].with.join(', '))}</span>` : ''}
        </div>
        <div class="small dim" style="margin-top:6px">${esc(inc.summary)}</div>
        <div style="margin-top:8px">${chainStrip(inc.tactics)}</div>
      </div>
      <div style="width:100px;flex:none">${riskCell(inc.risk, inc.severity)}</div>
      <span class="faint" style="flex:none">${isOpen ? '▾' : '▸'}</span>
    </div>
    ${isOpen ? body(inc, result) : ''}
  </div>`;
}

function body(inc, result) {
  const first = inc.alerts[0].ts;
  const span = Math.max(1, inc.end - first);
  return `<div class="incident-body">
    <div class="grid grid-2">
      <div>
        <h3>Sequence</h3>
        <div class="table-wrap">
          <table>
            <thead><tr><th style="width:88px">Risk</th><th>+time</th><th>Operation</th><th>Detection</th></tr></thead>
            <tbody>
              ${inc.alerts.slice().sort((a, b) => a.ts - b.ts).map((a) => `
                <tr class="${a.label === 1 ? 'is-truth' : ''}">
                  <td>${riskCell(a.risk, a.severity)}</td>
                  <td class="mono small nowrap">+${duration(a.ts - first)}</td>
                  <td class="mono small">${esc(clip(a.action, 30))}<div class="faint small">${esc(clip(a.resource, 34))}</div></td>
                  <td class="small">${esc(a.headline)}
                    <div class="faint small">${esc(a.tactic)}${a.technique ? ` · ${esc(a.technique)}` : ''}</div></td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <h3>Context</h3>
        <dl class="kv">
          <dt>Identity</dt><dd>${esc(inc.actor)} (${esc(inc.actorType)})</dd>
          <dt>Window</dt><dd>${esc(tsFull(inc.start))}<br>→ ${esc(tsFull(inc.end))}</dd>
          <dt>Addresses</dt><dd>${inc.ips.map((i) => esc(i)).join('<br>')}</dd>
          <dt>Countries</dt><dd>${inc.countries.length ? inc.countries.map(esc).join(', ') : 'not reported'}</dd>
          <dt>Rules</dt><dd>${inc.rules.length ? inc.rules.map(esc).join(', ') : 'model-only'}</dd>
          <dt>Peak risk</dt><dd>${Math.round(inc.maxRisk)} (incident ${Math.round(inc.risk)} after corroboration)</dd>
        </dl>

        ${inc.links.length ? `<h3 style="margin-top:14px">Linked activity</h3>
          ${inc.links.map((l) => `<div class="note small">${esc(l.note)} Related: ${esc(l.with.join(', '))}.</div>`).join('')}` : ''}

        <h3 style="margin-top:14px">Response</h3>
        <div class="row">
          <button class="btn btn-sm" data-copy-incident="${esc(inc.id)}">Copy ticket text</button>
          <button class="btn btn-sm btn-ghost" data-focus-identity="${esc(inc.actor)}">Identity profile</button>
        </div>
        <div class="small faint mt">${esc(recommendation(inc))}</div>
      </div>
    </div>
  </div>`;
}

/** Deterministic next-step guidance from the tactics present. */
function recommendation(inc) {
  const t = new Set(inc.tactics);
  const steps = [];
  if (t.has('Credential Access') || t.has('Initial Access')) steps.push('revoke active sessions and reset credentials for this identity');
  if (t.has('Persistence')) steps.push('enumerate and remove access keys, login profiles and safe memberships created in this window');
  if (t.has('Privilege Escalation')) steps.push('diff attached policies against the last known-good baseline');
  if (t.has('Defense Evasion')) steps.push('confirm audit logging is running and backfill the gap from a second source');
  if (t.has('Exfiltration') || t.has('Collection')) steps.push('check egress and snapshot sharing for cross-account destinations');
  if (!steps.length) steps.push('confirm with the identity owner whether this activity was expected');
  return `Suggested next steps: ${steps.join('; ')}.`;
}

function ticketText(inc) {
  const lines = [
    `${inc.id} — ${inc.severity.toUpperCase()} — ${inc.actor}`,
    '',
    inc.summary,
    '',
    `Window: ${tsFull(inc.start)} → ${tsFull(inc.end)} (${duration(inc.durationMs)})`,
    `Source addresses: ${inc.ips.join(', ')}`,
    `Kill chain: ${inc.tactics.join(' → ')}`,
    `Incident risk: ${Math.round(inc.risk)}/100 (peak alert ${Math.round(inc.maxRisk)})`,
    '',
    'Timeline:',
    ...inc.alerts.slice().sort((a, b) => a.ts - b.ts).map((a) =>
      `  ${tsFull(a.ts)}  [${String(Math.round(a.risk)).padStart(3)}]  ${a.action}  ${a.resource}\n      ${a.headline}`),
    '',
    recommendation(inc),
  ];
  return lines.join('\n');
}

export function mount(root, ctx) {
  on(root, 'click', '[data-toggle]', (ev, el) => {
    if (ev.target.closest('button')) return;
    const id = el.dataset.toggle;
    select({ incidentId: state.selection.incidentId === id ? null : id });
  });

  on(root, 'click', '[data-copy-incident]', (ev, el) => {
    const inc = state.result.incidents.find((i) => i.id === el.dataset.copyIncident);
    if (inc) copyText(ticketText(inc));
  });

  on(root, 'click', '[data-focus-identity]', (ev, el) => {
    select({ identity: el.dataset.focusIdentity });
    ctx.setView('identities');
  });

  on(root, 'click', '[data-export-incidents]', () => {
    const text = state.result.incidents.map(ticketText).join('\n\n' + '─'.repeat(72) + '\n\n');
    download('argus-incident-report.txt', text, 'text/plain');
    toast(`Exported ${state.result.incidents.length} incidents`);
  });

  const open = state.selection.incidentId;
  if (open) {
    const node = root.querySelector(`#inc-${CSS.escape(open)}`);
    if (node) node.scrollIntoView({ block: 'center' });
  }
}
