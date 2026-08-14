/**
 * Hand-rolled SVG charts.
 *
 * No charting library: every visual here is a few dozen lines of path
 * arithmetic, which keeps the whole console dependency-free and means the
 * rendering can be read and audited alongside the detection logic.
 * Charts are emitted as strings and use a 1000-unit viewBox so they scale to
 * any container width.
 */

import { esc, num } from './dom.js';

const W = 1000;
const SEV_COLOR = {
  critical: 'var(--critical)',
  high: 'var(--high)',
  medium: 'var(--medium)',
  low: 'var(--low)',
  info: 'var(--info)',
};

const path = (points) => points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');

/**
 * Event volume over time with alert counts overlaid.
 * Volume is an area (context); alerts are bars (the thing you look at).
 */
export function timelineChart(timeline, opts = {}) {
  const H = opts.height ?? 150;
  const pad = { l: 8, r: 8, t: 12, b: 20 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;
  const n = timeline.length;
  if (!n) return '';

  const maxEvents = Math.max(1, ...timeline.map((t) => t.events));
  const maxAlerts = Math.max(1, ...timeline.map((t) => t.alerts));
  const x = (i) => pad.l + (innerW * i) / Math.max(1, n - 1);
  const yE = (v) => pad.t + innerH - (innerH * v) / maxEvents;

  const areaPts = timeline.map((t, i) => [x(i), yE(t.events)]);
  const area = `${path(areaPts)} L${x(n - 1).toFixed(1)} ${(pad.t + innerH).toFixed(1)} L${x(0).toFixed(1)} ${(pad.t + innerH).toFixed(1)} Z`;

  const barW = Math.max(1.5, innerW / n - 1.5);
  const bars = timeline.map((t, i) => {
    if (!t.alerts) return '';
    const h = Math.max(2, (innerH * 0.75 * t.alerts) / maxAlerts);
    const colour = t.maxRisk >= 85 ? SEV_COLOR.critical : t.maxRisk >= 70 ? SEV_COLOR.high : SEV_COLOR.medium;
    return `<rect x="${(x(i) - barW / 2).toFixed(1)}" y="${(pad.t + innerH - h).toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" fill="${colour}" opacity=".9" rx="1"><title>${esc(new Date(t.t).toISOString().slice(0, 16).replace('T', ' '))} — ${t.alerts} alerts, ${t.events} events</title></rect>`;
  }).join('');

  const first = new Date(timeline[0].t);
  const last = new Date(timeline[n - 1].t);
  const mid = new Date(timeline[Math.floor(n / 2)].t);
  const label = (d) => d.toISOString().slice(5, 16).replace('T', ' ');

  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Event volume and alerts over time">
    <defs><linearGradient id="tlFill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="var(--accent)" stop-opacity=".28"/>
      <stop offset="100%" stop-color="var(--accent)" stop-opacity="0"/>
    </linearGradient></defs>
    <line class="grid-line" x1="${pad.l}" y1="${pad.t + innerH / 2}" x2="${W - pad.r}" y2="${pad.t + innerH / 2}"/>
    <path d="${area}" fill="url(#tlFill)"/>
    <path d="${path(areaPts)}" fill="none" stroke="var(--accent)" stroke-width="1.4" opacity=".8"/>
    ${bars}
    <line class="axis" x1="${pad.l}" y1="${pad.t + innerH}" x2="${W - pad.r}" y2="${pad.t + innerH}"/>
    <text x="${pad.l}" y="${H - 5}">${esc(label(first))}</text>
    <text x="${W / 2}" y="${H - 5}" text-anchor="middle">${esc(label(mid))}</text>
    <text x="${W - pad.r}" y="${H - 5}" text-anchor="end">${esc(label(last))}</text>
    <text x="${pad.l}" y="${pad.t - 2}">peak ${num(maxEvents)} events / bucket</text>
  </svg>`;
}

/** Generic XY line chart used for ROC and precision–recall. */
export function lineChart(series, opts = {}) {
  const H = opts.height ?? 260;
  const pad = { l: 44, r: 14, t: 14, b: 34 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;
  const x = (v) => pad.l + innerW * v;
  const y = (v) => pad.t + innerH * (1 - v);

  const grid = [0, 0.25, 0.5, 0.75, 1].map((g) => `
    <line class="grid-line" x1="${pad.l}" y1="${y(g)}" x2="${W - pad.r}" y2="${y(g)}"/>
    <text x="${pad.l - 7}" y="${y(g) + 3}" text-anchor="end">${g.toFixed(2)}</text>
    <line class="grid-line" x1="${x(g)}" y1="${pad.t}" x2="${x(g)}" y2="${pad.t + innerH}"/>
    <text x="${x(g)}" y="${H - 16}" text-anchor="middle">${g.toFixed(2)}</text>`).join('');

  const lines = series.map((s) => {
    const pts = s.points.map((p) => [x(Math.max(0, Math.min(1, p[0]))), y(Math.max(0, Math.min(1, p[1])))]);
    if (!pts.length) return '';
    return `<path d="${path(pts)}" fill="none" stroke="${s.colour}" stroke-width="${s.width ?? 1.8}"
      stroke-dasharray="${s.dash || ''}" opacity="${s.opacity ?? 1}"/>`;
  }).join('');

  const marker = opts.marker
    ? `<circle cx="${x(opts.marker[0])}" cy="${y(opts.marker[1])}" r="4" fill="var(--accent)" stroke="var(--bg)" stroke-width="1.5"/>`
    : '';

  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.label || 'chart')}">
    ${grid}
    ${opts.diagonal ? `<path d="M${x(0)} ${y(0)} L${x(1)} ${y(1)}" stroke="var(--line)" stroke-width="1" stroke-dasharray="4 4" fill="none"/>` : ''}
    ${lines}
    ${marker}
    <text x="${pad.l + innerW / 2}" y="${H - 2}" text-anchor="middle">${esc(opts.xLabel || '')}</text>
    <text x="10" y="${pad.t + innerH / 2}" text-anchor="middle" transform="rotate(-90 10 ${pad.t + innerH / 2})">${esc(opts.yLabel || '')}</text>
  </svg>`;
}

/** Severity ring. Reads as one glance: how much of the queue is on fire. */
export function donut(parts, opts = {}) {
  const size = opts.size ?? 168;
  const r = size / 2 - 14;
  const cx = size / 2;
  const cy = size / 2;
  const total = parts.reduce((s, p) => s + p.value, 0);
  if (!total) {
    return `<svg class="chart" viewBox="0 0 ${size} ${size}" style="max-width:${size}px">
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--bg-raise)" stroke-width="14"/>
      <text x="${cx}" y="${cy + 4}" text-anchor="middle" style="font-size:11px">no alerts</text></svg>`;
  }
  const circumference = 2 * Math.PI * r;
  let offset = 0;
  const arcs = parts.filter((p) => p.value > 0).map((p) => {
    const frac = p.value / total;
    const dash = `${(frac * circumference).toFixed(2)} ${(circumference * (1 - frac)).toFixed(2)}`;
    const seg = `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${p.colour}" stroke-width="14"
      stroke-dasharray="${dash}" stroke-dashoffset="${(-offset * circumference).toFixed(2)}"
      transform="rotate(-90 ${cx} ${cy})"><title>${esc(p.label)}: ${p.value}</title></circle>`;
    offset += frac;
    return seg;
  }).join('');

  return `<svg class="chart" viewBox="0 0 ${size} ${size}" style="max-width:${size}px" role="img" aria-label="Alert severity distribution">
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--bg-raise)" stroke-width="14"/>
    ${arcs}
    <text x="${cx}" y="${cy - 2}" text-anchor="middle" style="font-size:21px;fill:var(--text)">${num(total)}</text>
    <text x="${cx}" y="${cy + 14}" text-anchor="middle" style="font-size:9px">alerts</text>
  </svg>`;
}

/** 24-cell activity-by-hour strip for one identity. */
export function hourHeat(hist, opts = {}) {
  const max = Math.max(1, ...hist);
  const cells = hist.map((v, h) => {
    const a = v / max;
    const bg = v ? `rgba(53,208,232,${(0.12 + a * 0.78).toFixed(3)})` : 'var(--bg-raise)';
    const business = h >= 7 && h < 19;
    return `<i style="background:${bg}${business ? '' : ';box-shadow:inset 0 0 0 1px rgba(255,155,66,.22)'}" title="${h}:00 UTC — ${v} events"></i>`;
  }).join('');
  return `<div class="heat" aria-label="Activity by hour of day">${cells}</div>
    ${opts.legend === false ? '' : '<div class="small faint" style="margin-top:5px">00 → 23 UTC · outlined cells fall outside business hours</div>'}`;
}

/** Horizontal bar list (HTML, not SVG — it needs to wrap and select text). */
export function barList(items, opts = {}) {
  const max = Math.max(1, ...items.map((i) => i.value));
  return `<div class="bars">${items.map((i) => `
    <div class="bar-row">
      <span class="${opts.mono ? 'mono' : ''}" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(i.label)}">${esc(i.label)}</span>
      <span class="bar-track"><i style="width:${((i.value / max) * 100).toFixed(1)}%${i.colour ? `;background:${i.colour}` : ''}"></i></span>
      <span class="bar-val">${esc(i.display ?? num(i.value))}</span>
    </div>`).join('')}</div>`;
}

/** Compact inline sparkline for table cells. */
export function sparkline(values, opts = {}) {
  const w = opts.width ?? 90;
  const h = opts.height ?? 20;
  if (!values.length) return '';
  const max = Math.max(1, ...values);
  const pts = values.map((v, i) => [(w * i) / Math.max(1, values.length - 1), h - (h - 2) * (v / max) - 1]);
  return `<svg class="chart" viewBox="0 0 ${w} ${h}" style="width:${w}px;height:${h}px">
    <path d="${path(pts)}" fill="none" stroke="${opts.colour || 'var(--accent)'}" stroke-width="1.2" opacity=".9"/>
  </svg>`;
}

/** Kill-chain strip for an incident: tactics in order with alert counts. */
export function chainStrip(tactics) {
  return `<div class="chain">${tactics.map((t, i) => `
    ${i ? '<span class="chain-arrow">→</span>' : ''}<span class="chain-step">${esc(t)}</span>`).join('')}</div>`;
}

export { SEV_COLOR };
