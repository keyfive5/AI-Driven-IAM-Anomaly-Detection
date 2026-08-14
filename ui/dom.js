/** Tiny DOM + formatting helpers. Views build HTML strings and delegate events. */

export function esc(value) {
  if (value === null || value === undefined) return '';
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** Event delegation: on(root, 'click', '.selector', handler). */
export function on(root, type, selector, handler) {
  root.addEventListener(type, (ev) => {
    const target = ev.target.closest(selector);
    if (target && root.contains(target)) handler(ev, target);
  });
}

export const num = (v, digits = 0) =>
  v === null || v === undefined || Number.isNaN(v) ? '–' : Number(v).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });

export const pct = (v, digits = 1) => (v === null || v === undefined || Number.isNaN(v) ? '–' : `${(v * 100).toFixed(digits)}%`);

export const fixed = (v, digits = 3) => (v === null || v === undefined || Number.isNaN(v) ? '–' : Number(v).toFixed(digits));

const DT = new Intl.DateTimeFormat(undefined, {
  month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
});

export const ts = (ms) => (ms ? DT.format(new Date(ms)) : '–');

export const tsFull = (ms) => (ms ? new Date(ms).toISOString().replace('T', ' ').replace('.000Z', 'Z') : '–');

export function duration(ms) {
  if (ms === null || ms === undefined) return '–';
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function relTime(ms) {
  const diff = Date.now() - ms;
  const mins = Math.round(diff / 60000);
  if (Math.abs(mins) < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (Math.abs(hours) < 48) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/** Truncate for table cells without breaking the layout. */
export const clip = (s, n = 40) => {
  const str = String(s ?? '');
  return str.length > n ? `${str.slice(0, n - 1)}…` : str;
};

export function riskCell(risk, severity) {
  return `<div class="risk-cell"><span class="risk-num">${Math.round(risk)}</span>
    <span class="risk-bar"><i class="risk-fill fill-${severity}" style="width:${Math.min(100, risk)}%"></i></span></div>`;
}

export function sevTag(severity) {
  return `<span class="tag sev sev-${severity}">${severity}</span>`;
}

export function download(filename, content, type = 'application/json') {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function toast(message, kind = '') {
  const stack = document.getElementById('toasts');
  if (!stack) return;
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  stack.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transition = 'opacity .3s';
    setTimeout(() => el.remove(), 320);
  }, 3600);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast('Copied to clipboard');
  } catch {
    toast('Clipboard blocked by the browser', 'err');
  }
}
