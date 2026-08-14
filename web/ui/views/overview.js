import { esc, num, ts, duration, riskCell, sevTag, on } from '../dom.js';
import { timelineChart, donut, barList, chainStrip, SEV_COLOR } from '../charts.js';
import { setView, select } from '../state.js';

export function render(state) {
  const r = state.result;
  if (!r) return empty();
  const s = r.summary;
  const [start, end] = s.window;
  const critical = s.bySeverity.critical + s.bySeverity.high;
  const truth = r.evaluation ? r.alerts.filter((a) => a.label === 1).length : null;

  return `<div class="view">
    <div class="view-head">
      <div>
        <h1>Operations overview</h1>
        <div class="sub">${num(s.events)} events from ${num(s.identities)} identities across
          ${duration(end - start)}, scored in ${num(r.elapsedMs)} ms in this browser tab.</div>
      </div>
      <div class="row small faint">
        <span class="tag">threshold ${state.options.threshold}</span>
        <span class="tag">seed ${state.options.seed}</span>
        <span class="tag">${state.options.trees} trees</span>
      </div>
    </div>

    <div class="grid grid-kpi">
      ${kpi('Events analysed', num(s.events), `${num(s.failureRate * 100, 1)}% denied`)}
      ${kpi('Identities', num(s.identities), `${num(s.countries.length)} countries seen`)}
      ${kpi('Alerts', num(s.alerts), `${s.alertRate.toFixed(1)} per 1,000 events`, s.alertRate > 20 ? 'warn' : 'acc')}
      ${kpi('Incidents', num(s.incidents), 'correlated by identity + time')}
      ${kpi('Critical / high', num(critical), 'need an analyst today', critical ? 'crit' : 'ok')}
      ${truth === null
        ? kpi('Ground truth', '–', 'unlabelled corpus')
        : kpi('True positives', num(truth), `of ${num(r.alerts.length)} alerts raised`, 'crit')}
    </div>

    ${thinCorpusWarning(s)}

    <div class="panel mt">
      <div class="panel-head">
        <h2>Activity and alerting over time</h2>
        <span class="panel-note">area = event volume · bars = alerts, coloured by peak risk</span>
      </div>
      ${timelineChart(s.timeline, { height: 160 })}
    </div>

    <div class="grid grid-2 mt">
      <div class="panel">
        <div class="panel-head"><h2>Alert severity</h2><span class="panel-note">risk 85+ = critical</span></div>
        <div class="row" style="gap:22px;align-items:center">
          ${donut(['critical', 'high', 'medium', 'low', 'info'].map((k) => ({
            label: k, value: s.bySeverity[k], colour: SEV_COLOR[k],
          })), { size: 160 })}
          <div class="legend" style="flex-direction:column;gap:7px">
            ${['critical', 'high', 'medium', 'low'].map((k) => `
              <span><i style="background:${SEV_COLOR[k]}"></i>${k} — ${num(s.bySeverity[k])}</span>`).join('')}
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head"><h2>Kill-chain coverage</h2><span class="panel-note">MITRE ATT&amp;CK tactic per alert</span></div>
        ${s.byTactic.length
          ? barList(s.byTactic.slice(0, 8).map(([t, c]) => ({ label: t, value: c })))
          : '<div class="faint small">No alerts raised.</div>'}
      </div>
    </div>

    <div class="grid grid-2 mt">
      <div class="panel">
        <div class="panel-head">
          <h2>Highest-risk incidents</h2>
          <button class="btn btn-sm btn-ghost" data-goto="incidents">Open queue →</button>
        </div>
        ${r.incidents.length ? r.incidents.slice(0, 5).map((inc) => `
          <div class="incident" style="margin-bottom:9px" data-incident="${esc(inc.id)}">
            <div class="incident-head" style="padding:10px 12px">
              <span class="incident-rail" style="background:${SEV_COLOR[inc.severity]}"></span>
              <div style="flex:1;min-width:0">
                <div class="row" style="gap:8px">
                  ${sevTag(inc.severity)}
                  <strong class="mono" style="font-size:12.5px">${esc(inc.actor)}</strong>
                  <span class="faint small">${esc(inc.id)} · ${num(inc.alertCount)} alerts · ${duration(inc.durationMs)}</span>
                  ${inc.campaigns.length ? `<span class="tag tag-truth">ground truth: ${esc(inc.campaigns.join(', '))}</span>` : ''}
                </div>
                <div class="small dim" style="margin-top:5px">${esc(inc.topAlert.headline)}</div>
                <div style="margin-top:6px">${chainStrip(inc.tactics)}</div>
              </div>
              <div style="width:92px">${riskCell(inc.risk, inc.severity)}</div>
            </div>
          </div>`).join('')
          : '<div class="faint small">Nothing correlated — the estate looks quiet at this threshold.</div>'}
      </div>

      <div class="panel">
        <div class="panel-head">
          <h2>Identities to look at first</h2>
          <button class="btn btn-sm btn-ghost" data-goto="identities">All identities →</button>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Identity</th><th>Type</th><th class="right">Events</th><th class="right">Alerts</th><th style="width:110px">Peak risk</th></tr></thead>
            <tbody>
              ${r.identities.filter((i) => i.alerts > 0).slice(0, 8).map((i) => `
                <tr class="clickable" data-identity="${esc(i.actor)}">
                  <td class="mono">${esc(i.actor)}</td>
                  <td class="faint small">${esc(i.actorType)}</td>
                  <td class="right mono">${num(i.events)}</td>
                  <td class="right mono">${num(i.alerts)}</td>
                  <td>${riskCell(i.maxRisk, i.severity)}</td>
                </tr>`).join('') || '<tr><td colspan="5" class="faint small">No identity raised an alert.</td></tr>'}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="panel mt">
      <div class="panel-head">
        <h2>Detection rules that fired</h2>
        <span class="panel-note">counted across every event, before threshold</span>
      </div>
      ${s.byRule.length ? `<div class="table-wrap"><table>
        <thead><tr><th>Rule</th><th>Tactic</th><th class="right">Firings</th><th style="width:220px"></th></tr></thead>
        <tbody>${s.byRule.slice(0, 12).map((rule) => `
          <tr><td>${esc(rule.name)}</td>
            <td><span class="tag tag-tactic">${esc(rule.tactic)}</span></td>
            <td class="right mono">${num(rule.count)}</td>
            <td><span class="bar-track"><i style="width:${(rule.count / s.byRule[0].count) * 100}%"></i></span></td>
          </tr>`).join('')}</tbody></table></div>`
        : '<div class="faint small">No rule matched. Everything raised came from the models alone.</div>'}
    </div>
  </div>`;
}

/**
 * Behavioural detection needs history. On a small export every operation is
 * "the first time this identity has done this", so the alert rate says more
 * about the sample than the estate — say so rather than letting the number
 * be read as a finding.
 */
function thinCorpusWarning(s) {
  const perIdentity = s.events / Math.max(1, s.identities);
  if (s.events >= 400 && perIdentity >= 20) return '';
  return `<div class="panel mt" style="border-color:rgba(255,155,66,.35)">
    <div class="row" style="gap:10px;align-items:flex-start">
      <span style="color:var(--high);font-size:15px">⚠</span>
      <div class="small dim">
        <strong style="color:var(--high)">Thin corpus — read the alert rate with care.</strong>
        ${num(s.events)} events across ${num(s.identities)} identities is
        ${perIdentity.toFixed(0)} events each, which is not enough history to establish what normal
        looks like for anyone. Almost everything registers as novel, so scores here mostly reflect
        how sensitive each operation is rather than how out of character it was. Load a few thousand
        events, or generate a synthetic estate, before judging the detection quality.
      </div>
    </div>
  </div>`;
}

function kpi(label, value, sub, tone = '') {
  return `<div class="panel kpi">
    <div class="kpi-label">${esc(label)}</div>
    <div class="kpi-value ${tone}">${value}</div>
    <div class="kpi-sub">${esc(sub)}</div>
  </div>`;
}

function empty() {
  return `<div class="view"><div class="panel" style="text-align:center;padding:56px 24px">
    <h1 style="margin-bottom:8px">No corpus loaded</h1>
    <p class="dim">Generate a labelled synthetic estate, or drop in your own CloudTrail, Entra ID,
      Okta, GCP or CyberArk export. Everything is processed locally.</p>
    <button class="btn btn-primary mt" data-goto="data">Load data →</button>
  </div></div>`;
}

export function mount(root) {
  on(root, 'click', '[data-goto]', (ev, el) => setView(el.dataset.goto));
  on(root, 'click', '[data-incident]', (ev, el) => {
    select({ incidentId: el.dataset.incident });
    setView('incidents');
  });
  on(root, 'click', '[data-identity]', (ev, el) => {
    select({ identity: el.dataset.identity });
    setView('identities');
  });
}
