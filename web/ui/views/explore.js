import { esc, num, ts, clip, riskCell, on, download, toast } from '../dom.js';
import { severityFromRisk } from '../../core/schema.js';
import { state } from '../state.js';

/**
 * Raw event explorer with a small query language, because the question during
 * an investigation is never "show me everything" — it is "show me everything
 * this principal did from that address after 02:00".
 *
 *   actor:jchen  action:iam:  ip:203.  country:Singapore  outcome:failure
 *   risk>70  service:s3  resource:prod  free text
 */

const PAGE = 250;
let page = 1;
let query = '';

export function render(s) {
  const r = s.result;
  if (!r) return `<div class="view"><div class="panel faint">Run an analysis first.</div></div>`;

  const rows = filterEvents(r, query);
  const shown = rows.slice(0, page * PAGE);

  return `<div class="view">
    <div class="view-head">
      <div>
        <h1>Event explorer</h1>
        <div class="sub">Every normalised event with its score, whether or not it was alerted on.
          Filters: <code class="mono">actor:</code> <code class="mono">action:</code>
          <code class="mono">ip:</code> <code class="mono">country:</code> <code class="mono">service:</code>
          <code class="mono">outcome:</code> <code class="mono">risk&gt;70</code></div>
      </div>
      <button class="btn btn-sm" id="exportEvents">Export filtered CSV</button>
    </div>

    <div class="filters">
      <input type="search" id="eventQuery" value="${esc(query)}" placeholder="e.g. actor:jchen risk>60"
        autocomplete="off" spellcheck="false">
      <span class="small faint" style="margin-left:auto">${num(rows.length)} of ${num(r.events.length)} events</span>
    </div>

    <div class="panel" style="padding:0">
      <div class="table-wrap" style="border:0">
        <table>
          <thead><tr>
            <th style="width:96px">Risk</th><th>Time</th><th>Identity</th><th>Operation</th>
            <th>Resource</th><th>Source</th><th>Outcome</th><th>Client</th>
          </tr></thead>
          <tbody>
            ${shown.map(({ e, risk }) => `
              <tr class="${e.label === 1 ? 'is-truth' : ''}">
                <td>${riskCell(risk, severityFromRisk(risk))}</td>
                <td class="mono small nowrap">${esc(ts(e.ts))}</td>
                <td class="mono small">${esc(clip(e.actor, 18))}</td>
                <td class="mono small">${esc(clip(e.action, 32))}</td>
                <td class="small dim">${esc(clip(e.resource, 28))}</td>
                <td class="small dim nowrap">${esc(e.ip)}${e.country ? `<div class="faint">${esc(e.country)}</div>` : ''}</td>
                <td class="small ${e.outcome === 'failure' ? '' : 'faint'}">${esc(e.outcome)}</td>
                <td class="small faint">${esc(clip(e.userAgent, 26))}</td>
              </tr>`).join('') || '<tr><td colspan="8" class="faint small center" style="padding:34px">No events match.</td></tr>'}
          </tbody>
        </table>
      </div>
      ${rows.length > shown.length ? `<div class="pager">
        <button class="btn btn-sm" id="moreEvents">Show ${num(Math.min(PAGE, rows.length - shown.length))} more</button>
        <span class="small faint">${num(shown.length)} / ${num(rows.length)}</span></div>` : ''}
    </div>
  </div>`;
}

function filterEvents(result, raw) {
  const terms = raw.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const out = [];
  for (let i = 0; i < result.events.length; i++) {
    const e = result.events[i];
    const risk = result.risk[i];
    if (matches(e, risk, terms)) out.push({ e, risk });
  }
  out.sort((a, b) => b.risk - a.risk);
  return out;
}

function matches(e, risk, terms) {
  for (const term of terms) {
    const gt = term.match(/^risk>(\d+(?:\.\d+)?)$/);
    if (gt) { if (risk <= Number(gt[1])) return false; continue; }
    const lt = term.match(/^risk<(\d+(?:\.\d+)?)$/);
    if (lt) { if (risk >= Number(lt[1])) return false; continue; }

    const kv = term.match(/^(actor|action|ip|country|service|outcome|resource|region|type):(.*)$/);
    if (kv) {
      const [, field, value] = kv;
      const map = {
        actor: e.actor, action: e.action, ip: e.ip, country: e.country || '',
        service: e.service, outcome: e.outcome, resource: e.resource, region: e.region, type: e.actorType,
      };
      if (!String(map[field] ?? '').toLowerCase().includes(value)) return false;
      continue;
    }

    const hay = `${e.actor} ${e.action} ${e.resource} ${e.ip} ${e.country || ''} ${e.userAgent}`.toLowerCase();
    if (!hay.includes(term)) return false;
  }
  return true;
}

export function mount(root, ctx) {
  const input = root.querySelector('#eventQuery');
  if (input) {
    let timer;
    input.addEventListener('input', () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        query = input.value;
        page = 1;
        ctx.rerender();
        const again = document.querySelector('#eventQuery');
        if (again) { again.focus(); again.setSelectionRange(again.value.length, again.value.length); }
      }, 250);
    });
  }

  on(root, 'click', '#moreEvents', () => { page++; ctx.rerender(); });

  on(root, 'click', '#exportEvents', () => {
    const rows = filterEvents(state.result, query);
    const head = ['timestamp', 'risk', 'identity', 'identity_type', 'operation', 'resource', 'ip', 'country', 'region', 'outcome', 'client', 'ground_truth'];
    const body = rows.map(({ e, risk }) => [
      new Date(e.ts).toISOString(), risk.toFixed(1), e.actor, e.actorType, e.action, e.resource,
      e.ip, e.country || '', e.region, e.outcome, e.userAgent, e.label === 1 ? 'malicious' : '',
    ]);
    const cell = (v) => {
      const str = String(v ?? '');
      return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
    };
    download('argus-events.csv', [head, ...body].map((r) => r.map(cell).join(',')).join('\n'), 'text/csv');
    toast(`Exported ${rows.length} events`);
  });
}
