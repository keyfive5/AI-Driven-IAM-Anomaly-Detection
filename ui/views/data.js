import { esc, num, ts, duration, on, toast } from '../dom.js';
import { ingest, FORMAT_LABELS } from '../../core/parse.js';
import { CAMPAIGN_IDS, campaignInfo } from '../../core/generate.js';
import { RULES } from '../../core/rules.js';
import { state, loadSynthetic, loadEvents, saveSettings, removeSuppression, MAX_EVENTS } from '../state.js';

const SAMPLES = [
  { file: 'sample_aws_cloudtrail.json', name: 'AWS CloudTrail', note: 'Real-shaped management events: IAM, EC2, console sign-ins.' },
  { file: 'sample_entra_signins.json', name: 'Microsoft Entra ID', note: 'Sign-in logs with geography, MFA state and failure codes.' },
  { file: 'sample_okta_system_log.json', name: 'Okta System Log', note: 'Authentication and administrative events with geo context.' },
  { file: 'sample_cyberark_pam.json', name: 'CyberArk PAM', note: 'Vault checkouts, safe membership changes, PSM sessions.' },
  { file: 'sample_azure_activity.json', name: 'Azure Activity Log', note: 'Resource-plane administrative operations.' },
];

export function render(s) {
  const meta = s.meta;
  return `<div class="view">
    <div class="view-head">
      <div>
        <h1>Data &amp; tuning</h1>
        <div class="sub">Load a corpus and shape how the engine scores it. Files never leave this
          machine — parsing and detection both run in the page.</div>
      </div>
      ${meta ? `<div class="row small faint">
        <span class="tag">${esc(FORMAT_LABELS[meta.source] || meta.source)}</span>
        <span class="tag">${num(s.events.length)} events</span>
        ${meta.labelled ? '<span class="tag tag-truth">labelled</span>' : ''}
      </div>` : ''}
    </div>

    <div class="grid grid-2">
      <div class="panel">
        <div class="panel-head">
          <h2>Generate a labelled estate</h2>
          <span class="panel-note">reproducible from the seed</span>
        </div>
        <p class="small dim">Builds a synthetic organisation — engineers, analysts, contractors,
          service accounts — each with working hours, a home network and a repertoire of API calls,
          then injects multi-stage intrusions with per-event ground truth.</p>

        <div class="grid grid-3 mt" style="gap:10px">
          <label>Seed<input type="number" id="genSeed" value="${s.generator.seed}" style="width:100%"></label>
          <label>Days<input type="number" id="genDays" min="3" max="60" value="${s.generator.days}" style="width:100%"></label>
          <label>Identities<input type="number" id="genUsers" min="8" max="200" value="${s.generator.users}" style="width:100%"></label>
        </div>
        <div id="genEstimate">${estimateNote(s.generator.days, s.generator.users)}</div>

        <h3 style="margin-top:14px">Campaigns to inject</h3>
        <div class="switchlist">
          ${CAMPAIGN_IDS.map((id) => {
            const info = campaignInfo(id);
            return `<label class="switch">
              <input type="checkbox" data-campaign="${esc(id)}" ${s.generator.campaigns.includes(id) ? 'checked' : ''}>
              <span>${esc(info.name)}</span>
              <span class="desc">${esc(info.technique)} · ${esc(info.difficulty)}</span>
            </label>`;
          }).join('')}
        </div>

        <button class="btn btn-primary mt" id="btnGenerate">Generate and analyse</button>
      </div>

      <div class="stack">
        <div class="panel">
          <div class="panel-head"><h2>Load your own logs</h2><span class="panel-note">JSON · JSONL · CSV</span></div>
          <div class="dropzone" id="dropzone">
            <div style="font-size:22px">⬓</div>
            <div style="margin-top:6px">Drop a log export here, or click to choose a file</div>
            <div class="small faint" style="margin-top:6px">
              CloudTrail · Entra ID · Okta · GCP audit · Azure Activity · CyberArk · generic CSV
            </div>
          </div>
          <input type="file" id="fileInput" accept=".json,.jsonl,.ndjson,.csv,.txt,.log" hidden>
          <div class="small faint mt">The format is detected from the record shape. Unrecognised
            files fall back to a generic parser that maps common column names.</div>
        </div>

        <div class="panel">
          <div class="panel-head"><h2>Sample corpora</h2><span class="panel-note">bundled with the repo</span></div>
          <div class="stack" style="gap:8px">
            ${SAMPLES.map((sample) => `
              <button class="preset" data-sample="${esc(sample.file)}">
                <span class="preset-icon">▤</span>
                <span>
                  <strong style="font-size:12.5px">${esc(sample.name)}</strong>
                  <div class="small faint">${esc(sample.note)}</div>
                </span>
              </button>`).join('')}
          </div>
        </div>
      </div>
    </div>

    <div class="grid grid-2 mt">
      <div class="panel">
        <div class="panel-head"><h2>Engine settings</h2><span class="panel-note">applied on next run</span></div>

        <div class="slider-row">
          <label for="optThreshold">Alert threshold</label>
          <input type="range" id="optThreshold" min="20" max="95" value="${s.options.threshold}">
          <output>${s.options.threshold}</output>
        </div>
        <div class="small faint" style="margin:2px 0 12px">Risk is a tail percentile: 57 ≈ top 1% of
          events, 86 ≈ top 0.1%. Lower it to see more, raise it to protect the queue.</div>

        <h3>Detector weights</h3>
        ${weight('Isolation Forest', 'iforest', s.options.weights.iforest)}
        ${weight('Robust z', 'robustz', s.options.weights.robustz)}
        ${weight('Behavioural surprisal', 'baseline', s.options.weights.baseline)}

        <h3 style="margin-top:14px">Isolation Forest</h3>
        <div class="grid grid-3" style="gap:10px">
          <label>Trees<input type="number" id="optTrees" min="20" max="400" step="10" value="${s.options.trees}" style="width:100%"></label>
          <label>Subsample<input type="number" id="optSample" min="32" max="1024" step="32" value="${s.options.sampleSize}" style="width:100%"></label>
          <label>Seed<input type="number" id="optSeed" value="${s.options.seed}" style="width:100%"></label>
        </div>

        <div class="slider-row mt">
          <label for="optGap">Incident gap</label>
          <input type="range" id="optGap" min="5" max="240" step="5" value="${s.options.incidentGapMinutes}">
          <output>${s.options.incidentGapMinutes}m</output>
        </div>

        <button class="btn btn-primary mt" id="btnRerun">Apply and re-run</button>
        <button class="btn btn-ghost mt" id="btnReset">Reset to defaults</button>
      </div>

      <div class="stack">
        <div class="panel">
          <div class="panel-head">
            <h2>Muted detections</h2>
            <span class="panel-note">${num(s.suppressions.length)} active</span>
          </div>
          <p class="small dim">Muting a rule for one identity is how a real SOC stops a legitimate
            pattern from re-alerting daily. Mutes apply after scoring, so nothing is re-run and
            nothing is lost — the event is still in the explorer with its score.</p>
          ${s.suppressions.length ? `<div class="table-wrap mt"><table>
            <thead><tr><th>Rule</th><th>Identity</th><th></th></tr></thead>
            <tbody>${s.suppressions.map((sup, i) => `
              <tr><td class="small">${esc(RULES.find((r) => r.id === sup.rule)?.name || sup.rule)}</td>
                <td class="mono small">${esc(sup.actor)}</td>
                <td class="right"><button class="btn btn-sm btn-ghost" data-unmute="${i}">Restore</button></td></tr>`).join('')}
            </tbody></table></div>` : '<div class="faint small mt">Nothing muted.</div>'}
        </div>

        ${meta ? `<div class="panel">
          <div class="panel-head"><h2>Loaded corpus</h2></div>
          <dl class="kv">
            <dt>Source</dt><dd>${esc(FORMAT_LABELS[meta.source] || meta.source)}</dd>
            <dt>Events</dt><dd>${num(s.events.length)}</dd>
            <dt>Window</dt><dd>${esc(ts(s.events[0]?.ts))} → ${esc(ts(s.events[s.events.length - 1]?.ts))}</dd>
            <dt>Span</dt><dd>${esc(duration((s.events[s.events.length - 1]?.ts || 0) - (s.events[0]?.ts || 0)))}</dd>
            ${meta.seed !== undefined ? `<dt>Seed</dt><dd>${esc(meta.seed)}</dd>` : ''}
            ${meta.campaigns ? `<dt>Campaigns</dt><dd>${num(meta.campaigns.length)}</dd>` : ''}
            ${meta.skipped ? `<dt>Skipped</dt><dd>${num(meta.skipped)} unparseable records</dd>` : ''}
            ${meta.truncated ? `<dt>Truncated</dt><dd style="color:var(--high)">${num(meta.truncated)} older events dropped (${num(MAX_EVENTS)} cap)</dd>` : ''}
          </dl>
        </div>` : ''}
      </div>
    </div>
  </div>`;
}

/**
 * Projected corpus size, so the cost of a setting is visible before it is paid.
 *
 * These inputs are persisted, and a heavy choice therefore re-applies on every
 * reload: at the top of the range the estate is ~330,000 events, which is half
 * a minute of arithmetic on a fast desktop and minutes on a phone. Showing the
 * number turns a trap into a decision.
 */
const estimateEvents = (days, users) => Math.round(days * (468 + users * 26));

function estimateNote(days, users) {
  const n = estimateEvents(days, users);
  const heavy = n > 60000;
  const veryHeavy = n > 150000;
  return `<div class="small ${veryHeavy ? '' : 'faint'}" style="margin-top:8px${veryHeavy ? ';color:var(--high)' : ''}">
    ≈ <strong class="mono">${num(n)}</strong> events
    ${heavy ? `· ${veryHeavy ? 'this will take a while and use significant memory on a modest machine'
      : 'noticeably slower to analyse on a modest machine'}` : '· analyses in a second or two'}
  </div>`;
}

function weight(label, key, value) {
  return `<div class="slider-row">
    <label for="w_${key}">${esc(label)}</label>
    <input type="range" id="w_${key}" data-weight="${key}" min="0" max="1" step="0.05" value="${value}">
    <output>${value.toFixed(2)}</output>
  </div>`;
}

export function mount(root, ctx) {
  const gen = {
    genSeed: 'seed', genDays: 'days', genUsers: 'users',
  };
  const refreshEstimate = () => {
    const host = root.querySelector('#genEstimate');
    if (host) host.innerHTML = estimateNote(state.generator.days, state.generator.users);
  };
  for (const [id, key] of Object.entries(gen)) {
    const el = root.querySelector(`#${id}`);
    if (!el) continue;
    const apply = () => {
      const value = Number(el.value);
      if (!Number.isFinite(value)) return;
      // Clamp to the input's own bounds: typing past them otherwise persists a
      // setting the UI never offered.
      const min = Number(el.min);
      const max = Number(el.max);
      state.generator[key] = el.min !== '' && el.max !== ''
        ? Math.min(max, Math.max(min, value))
        : value;
      el.value = state.generator[key];
      saveSettings();
      refreshEstimate();
    };
    el.addEventListener('change', apply);
    el.addEventListener('input', apply);
  }

  on(root, 'change', '[data-campaign]', (ev, el) => {
    const id = el.dataset.campaign;
    const list = state.generator.campaigns;
    if (el.checked && !list.includes(id)) list.push(id);
    if (!el.checked) state.generator.campaigns = list.filter((c) => c !== id);
    saveSettings();
  });

  on(root, 'click', '#btnGenerate', async () => {
    loadSynthetic();
    await ctx.runAnalysis();
    ctx.setView('overview');
  });

  // --- settings ------------------------------------------------------------
  bindSlider(root, '#optThreshold', (v) => { state.options.threshold = v; });
  bindSlider(root, '#optGap', (v) => { state.options.incidentGapMinutes = v; }, (v) => `${v}m`);
  for (const el of root.querySelectorAll('[data-weight]')) {
    el.addEventListener('input', () => {
      state.options.weights[el.dataset.weight] = Number(el.value);
      const out = el.parentElement.querySelector('output');
      if (out) out.textContent = Number(el.value).toFixed(2);
      saveSettings();
    });
  }
  for (const [id, key] of Object.entries({ optTrees: 'trees', optSample: 'sampleSize', optSeed: 'seed' })) {
    const el = root.querySelector(`#${id}`);
    if (el) el.addEventListener('change', () => { state.options[key] = Number(el.value); saveSettings(); });
  }

  on(root, 'click', '#btnRerun', () => ctx.runAnalysis());
  on(root, 'click', '#btnReset', () => {
    localStorage.removeItem('argus.settings');
    location.reload();
  });
  on(root, 'click', '[data-unmute]', (ev, el) => {
    removeSuppression(Number(el.dataset.unmute));
    toast('Rule restored');
    ctx.rerender();
  });

  // --- file loading --------------------------------------------------------
  const drop = root.querySelector('#dropzone');
  const input = root.querySelector('#fileInput');
  if (drop && input) {
    drop.addEventListener('click', () => input.click());
    input.addEventListener('change', () => input.files[0] && readFile(input.files[0], ctx));
    ['dragenter', 'dragover'].forEach((type) => drop.addEventListener(type, (e) => {
      e.preventDefault();
      drop.classList.add('is-over');
    }));
    ['dragleave', 'drop'].forEach((type) => drop.addEventListener(type, (e) => {
      e.preventDefault();
      drop.classList.remove('is-over');
    }));
    drop.addEventListener('drop', (e) => {
      const file = e.dataTransfer?.files?.[0];
      if (file) readFile(file, ctx);
    });
  }

  on(root, 'click', '[data-sample]', async (ev, el) => {
    try {
      const res = await fetch(`data/${el.dataset.sample}`);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      await consume(await res.text(), el.dataset.sample, ctx);
    } catch (err) {
      toast(`Could not load sample: ${err.message}`, 'err');
    }
  });
}

function bindSlider(root, sel, apply, format = (v) => String(v)) {
  const el = root.querySelector(sel);
  if (!el) return;
  el.addEventListener('input', () => {
    apply(Number(el.value));
    const out = el.parentElement.querySelector('output');
    if (out) out.textContent = format(el.value);
    saveSettings();
  });
}

function readFile(file, ctx) {
  const reader = new FileReader();
  reader.onload = () => consume(String(reader.result), file.name, ctx);
  reader.onerror = () => toast('Could not read that file', 'err');
  reader.readAsText(file);
}

async function consume(text, name, ctx) {
  try {
    const { events, format, skipped, total } = ingest(text);
    if (!events.length) throw new Error('No events could be parsed from that file.');
    const { loaded, truncated } = loadEvents(events, {
      source: format,
      name,
      skipped,
      total,
      labelled: events.some((e) => e.label === 1),
      campaigns: [],
    });
    toast(`${name}: ${loaded} events parsed as ${FORMAT_LABELS[format] || format}${skipped ? ` (${skipped} skipped)` : ''}`);
    if (truncated) {
      toast(`File held ${num(events.length)} events — analysing the most recent ${num(loaded)}.`, 'err');
    }
    await ctx.runAnalysis();
    ctx.setView('overview');
  } catch (err) {
    toast(err.message, 'err');
  }
}
