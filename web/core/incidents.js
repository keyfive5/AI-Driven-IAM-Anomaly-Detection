/**
 * Alert correlation.
 *
 * A single alert is a data point; an intrusion is a *shape*. Analysts drown
 * because tools ship them 400 independent rows describing one afternoon. ARGUS
 * groups alerts into incidents along the axis attackers actually travel —
 * one identity, one contiguous stretch of time — then orders the members by
 * ATT&CK tactic so the story reads in kill-chain order rather than by score.
 *
 * Incidents are also linked when they share a source address, which is what
 * catches an operator pivoting from a compromised identity onto a second one.
 */

import { TACTIC_ORDER } from './schema.js';
import { severityFromRisk } from './schema.js';

const DEFAULT_GAP = 45 * 60_000;

export function buildIncidents(alerts, opts = {}) {
  const gap = opts.gap ?? DEFAULT_GAP;
  const byActor = new Map();
  for (const a of alerts) {
    if (!byActor.has(a.actor)) byActor.set(a.actor, []);
    byActor.get(a.actor).push(a);
  }

  const incidents = [];
  for (const [actor, list] of byActor) {
    list.sort((x, y) => x.ts - y.ts);
    let current = null;
    for (const a of list) {
      if (!current || a.ts - current.end > gap) {
        current = { actor, alerts: [a], start: a.ts, end: a.ts };
        incidents.push(current);
      } else {
        current.alerts.push(a);
        current.end = a.ts;
      }
    }
  }

  const finished = incidents.map((inc, i) => finalise(inc, i));
  finished.sort((a, b) => b.risk - a.risk);
  linkByAddress(finished);
  return finished;
}

function finalise(inc, i) {
  const alerts = inc.alerts;
  const maxRisk = Math.max(...alerts.map((a) => a.risk));
  const tactics = [...new Set(alerts.map((a) => a.tactic))]
    .sort((a, b) => (TACTIC_ORDER[a] ?? 99) - (TACTIC_ORDER[b] ?? 99));
  const rules = [...new Set(alerts.flatMap((a) => a.rules.map((r) => r.id)))];
  const ips = [...new Set(alerts.map((a) => a.ip))];
  const countries = [...new Set(alerts.map((a) => a.country).filter(Boolean))];

  // Corroboration bonus: breadth across the kill chain matters more than
  // repetition of the same alert, so distinct tactics are weighted heavier.
  const breadth = Math.min(12, 3 * (tactics.length - 1));
  const volume = Math.min(6, 2 * Math.log2(alerts.length + 1));
  const risk = Math.min(100, maxRisk + breadth + volume);

  const id = `INC-${String(i + 1).padStart(4, '0')}`;
  for (const a of alerts) a.incidentId = id;

  return {
    id,
    actor: inc.actor,
    actorType: alerts[0].actorType,
    start: inc.start,
    end: inc.end,
    durationMs: inc.end - inc.start,
    alerts,
    alertCount: alerts.length,
    maxRisk,
    risk: Math.round(risk * 10) / 10,
    severity: severityFromRisk(risk),
    tactics,
    rules,
    ips,
    countries,
    topAlert: alerts.slice().sort((a, b) => b.risk - a.risk)[0],
    labelled: alerts.some((a) => a.label === 1),
    campaigns: [...new Set(alerts.map((a) => a.campaign).filter(Boolean))],
    links: [],
    summary: summarise(inc.actor, alerts, tactics),
  };
}

/** One-paragraph narrative an analyst can paste into a ticket. */
function summarise(actor, alerts, tactics) {
  const first = alerts[0];
  const last = alerts[alerts.length - 1];
  const mins = Math.max(1, Math.round((last.ts - first.ts) / 60000));
  const ruleNames = [...new Set(alerts.flatMap((a) => a.rules.map((r) => r.name)))];
  const where = first.country ? ` from ${first.country}` : '';
  const chain = tactics.join(' → ');
  const lead = `${actor} produced ${alerts.length} correlated alert${alerts.length === 1 ? '' : 's'} over ${mins} minute${mins === 1 ? '' : 's'}${where}.`;
  const body = ruleNames.length
    ? ` Detections fired: ${ruleNames.slice(0, 4).join('; ')}${ruleNames.length > 4 ? `; +${ruleNames.length - 4} more` : ''}.`
    : ' No signature matched — this was flagged purely on deviation from the identity\'s own baseline.';
  return `${lead}${body} Kill-chain coverage: ${chain}.`;
}

/** Join incidents that share a non-private source address. */
function linkByAddress(incidents) {
  const byIp = new Map();
  for (const inc of incidents) {
    for (const ip of inc.ips) {
      if (!ip || ip === '0.0.0.0') continue;
      if (!byIp.has(ip)) byIp.set(ip, []);
      byIp.get(ip).push(inc);
    }
  }
  for (const [ip, list] of byIp) {
    if (list.length < 2) continue;
    const actors = new Set(list.map((i) => i.actor));
    if (actors.size < 2) continue;
    for (const inc of list) {
      inc.links.push({
        type: 'shared-address',
        ip,
        with: list.filter((o) => o !== inc).map((o) => o.id),
        note: `${actors.size} identities acted from ${ip}.`,
      });
    }
  }
}
