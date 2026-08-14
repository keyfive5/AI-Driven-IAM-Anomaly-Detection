import { esc, num, pct, fixed, duration, on, toast, download } from '../dom.js';
import { lineChart } from '../charts.js';
import { confusionAt } from '../../core/evaluate.js';
import { state, setView } from '../state.js';

/**
 * The view the original project could not have produced: every number here is
 * recomputed from labelled data in front of the user, and the threshold slider
 * shows the trade-off rather than hiding it behind one flattering operating
 * point.
 */

let liveThreshold = null;

export function render(s) {
  const r = s.result;
  if (!r) return `<div class="view"><div class="panel faint">Run an analysis first.</div></div>`;
  if (!r.evaluation) return unlabelled();

  const ev = r.evaluation;
  const labels = r.events.map((e) => e.label || 0);
  const risks = Array.from(r.risk);
  const threshold = liveThreshold ?? s.options.threshold;
  const cm = confusionAt(risks, labels, threshold);
  const best = ev.overall.best || {};
  const positives = labels.reduce((a, b) => a + b, 0);

  const prPoints = ev.overall.pr.map((p) => [p.recall, p.precision]);
  const rocPoints = ev.overall.roc.map((p) => [p.fpr, p.tpr]);

  return `<div class="view">
    <div class="view-head">
      <div>
        <h1>Evaluation</h1>
        <div class="sub">Ground truth comes from the injected campaigns in this corpus
          (${num(positives)} malicious events out of ${num(labels.length)}, ${pct(positives / labels.length, 2)}).
          Every figure below is computed live from seed ${s.generator.seed} — nothing is quoted.</div>
      </div>
      <button class="btn btn-sm" id="exportEval">Export results JSON</button>
    </div>

    <div class="grid grid-kpi">
      ${kpi('ROC AUC', fixed(ev.overall.auc, 3), 'ranking quality, 0.5 = coin flip', 'acc')}
      ${kpi('Average precision', fixed(ev.overall.ap, 3), 'area under precision–recall')}
      ${kpi('Best F1', fixed(best.f1, 3), `at threshold ${Math.round(best.threshold ?? 0)}`)}
      ${kpi('Campaigns caught', `${ev.campaigns.filter((c) => c.detected).length}/${ev.campaigns.length}`,
        'multi-stage intrusions detected', 'ok')}
      ${kpi('Alert budget', `${cm.alertsPerThousand.toFixed(1)}`, 'alerts per 1,000 events')}
    </div>

    <div class="panel mt">
      <div class="panel-head">
        <h2>Operating point</h2>
        <span class="panel-note">move the threshold to trade recall against analyst workload</span>
      </div>
      <div class="slider-row">
        <label for="threshSlider">Decision threshold</label>
        <input type="range" id="threshSlider" min="20" max="99" step="1" value="${threshold}">
        <output>${threshold}</output>
      </div>

      <div class="grid grid-2 mt">
        <div>
          <div class="table-wrap">
            <table>
              <thead><tr><th></th><th class="right">Predicted malicious</th><th class="right">Predicted benign</th></tr></thead>
              <tbody>
                <tr><th style="position:static">Actually malicious</th>
                  <td class="right mono" style="color:var(--ok)">${num(cm.tp)}</td>
                  <td class="right mono" style="color:var(--critical)">${num(cm.fn)}</td></tr>
                <tr><th style="position:static">Actually benign</th>
                  <td class="right mono" style="color:var(--high)">${num(cm.fp)}</td>
                  <td class="right mono faint">${num(cm.tn)}</td></tr>
              </tbody>
            </table>
          </div>
          <div class="small faint mt">
            An analyst reviewing this queue opens ${num(cm.flagged)} events and finds
            ${num(cm.tp)} genuinely malicious ones.
          </div>
        </div>

        <div class="grid grid-3" style="gap:9px;align-content:start">
          ${stat('Precision', pct(cm.precision, 1))}
          ${stat('Recall', pct(cm.recall, 1))}
          ${stat('F1', fixed(cm.f1, 3))}
          ${stat('False positive rate', pct(cm.fpr, 2))}
          ${stat('Flagged', num(cm.flagged))}
          ${stat('Missed', num(cm.fn))}
        </div>
      </div>
    </div>

    <div class="grid grid-2 mt">
      <div class="panel">
        <div class="panel-head"><h2>Precision–recall</h2><span class="panel-note">AP ${fixed(ev.overall.ap, 3)}</span></div>
        ${lineChart([{ points: prPoints, colour: 'var(--accent)' }], {
          xLabel: 'recall', yLabel: 'precision', label: 'Precision-recall curve',
          marker: [cm.recall, cm.precision], height: 250,
        })}
        <div class="small faint">Marker = current threshold. A flat right-hand tail means extra
          recall is bought with a lot of false positives.</div>
      </div>

      <div class="panel">
        <div class="panel-head"><h2>ROC</h2><span class="panel-note">AUC ${fixed(ev.overall.auc, 3)}</span></div>
        ${lineChart([{ points: rocPoints, colour: 'var(--high)' }], {
          xLabel: 'false positive rate', yLabel: 'true positive rate', diagonal: true,
          marker: [cm.fpr, cm.recall], height: 250, label: 'ROC curve',
        })}
        <div class="small faint">ROC flatters rare-event detectors; precision–recall above is the
          honest picture when 0.5% of events are malicious.</div>
      </div>
    </div>

    <div class="panel mt">
      <div class="panel-head">
        <h2>Detector ablation</h2>
        <span class="panel-note">each variant at the threshold that maximises its own F1</span>
      </div>
      <div class="table-wrap"><table>
        <thead><tr><th>Configuration</th><th class="right">ROC AUC</th><th class="right">Avg precision</th>
          <th class="right">F1</th><th class="right">Precision</th><th class="right">Recall</th>
          <th class="right">Alerts / 1k</th></tr></thead>
        <tbody>
          ${ev.ablation.map((a) => {
            const shipping = a.name.startsWith('ARGUS');
            return `<tr style="${shipping ? 'background:rgba(53,208,232,.06)' : ''}">
              <td>${shipping ? '<strong>' : ''}${esc(a.name)}${shipping ? '</strong>' : ''}
                <div class="faint small">${esc(a.note)}</div></td>
              <td class="right mono">${fixed(a.auc, 3)}</td>
              <td class="right mono">${fixed(a.ap, 3)}</td>
              <td class="right mono">${fixed(a.f1, 3)}</td>
              <td class="right mono">${pct(a.precision, 1)}</td>
              <td class="right mono">${pct(a.recall, 1)}</td>
              <td class="right mono">${a.alertsPerThousand?.toFixed(1) ?? '–'}</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table></div>
      <div class="note mt small">${esc(ablationVerdict(ev.ablation))}</div>
    </div>

    <div class="panel mt">
      <div class="panel-head">
        <h2>Campaign detection</h2>
        <span class="panel-note">did each injected intrusion get caught, and how fast</span>
      </div>
      <div class="table-wrap"><table>
        <thead><tr><th>Campaign</th><th>Technique</th><th>Target</th><th class="right">Malicious events</th>
          <th class="right">Caught</th><th class="right">Time to detect</th><th>Found by</th><th>Status</th></tr></thead>
        <tbody>
          ${ev.campaigns.map((c) => `<tr>
            <td>${esc(c.name)}<div class="faint small mono">${esc(c.id)} · ${esc(c.difficulty)}</div></td>
            <td class="small mono">${esc(c.technique)}</td>
            <td class="mono small">${esc(c.actor)}</td>
            <td class="right mono">${num(c.totalMalicious)}</td>
            <td class="right mono">${num(c.caught)} <span class="faint">(${pct(c.coverage, 0)})</span></td>
            <td class="right mono">${c.detected ? duration(c.latencyMs) : '–'}</td>
            <td class="small">${attributionTag(c)}</td>
            <td>${c.detected
              ? '<span class="tag tag-benign">detected</span>'
              : '<span class="tag tag-truth">missed</span>'}</td>
          </tr>`).join('')}
        </tbody>
      </table></div>
      <div class="small faint mt">Time to detect is measured from the campaign's first event to the
        first one that cleared the threshold. Event-level coverage below 100% is expected and fine —
        a fifty-object exfiltration only needs to be caught once. "Found by" is the useful column:
        it shows which half of the system actually earned its keep on each intrusion.</div>
      ${rulesOnlyNote(ev.campaigns)}
    </div>

    <div class="panel mt">
      <h3>How to read these numbers honestly</h3>
      <p class="small dim">These metrics describe performance <em>on this synthetic corpus</em>. The
        generator was written alongside the detector, so it inevitably encodes some of the same
        assumptions about what "normal" looks like — treat the absolute values as an upper bound and
        the <em>ordering</em> of the ablation as the real result. Deliberately included benign
        confusers (a 03:00 on-call page, business travel, a quarterly access review, an onboarding
        batch, a release-day deploy storm) exist to stop precision looking better than it is.</p>
      <p class="small dim">The one number that transfers to a real estate is the alert budget:
        ${cm.alertsPerThousand.toFixed(1)} alerts per 1,000 events is a workload claim, and it holds
        regardless of whether the labels are right.</p>
    </div>
  </div>`;
}

function attributionTag(c) {
  if (!c.detected) return '<span class="faint">–</span>';
  const map = {
    both: '<span class="tag">models + rules</span>',
    'rules only': '<span class="tag tag-truth">rules only</span>',
    'models only': '<span class="tag tag-tactic">models only</span>',
    missed: '<span class="faint">–</span>',
  };
  return map[c.attribution] || '<span class="faint">–</span>';
}

/** Name the campaigns that only the rule pack caught — the ablation's blind spot. */
function rulesOnlyNote(campaigns) {
  const rulesOnly = campaigns.filter((c) => c.detected && c.attribution === 'rules only');
  const modelsOnly = campaigns.filter((c) => c.detected && c.attribution === 'models only');
  if (!rulesOnly.length && !modelsOnly.length) return '';
  const bits = [];
  if (rulesOnly.length) {
    bits.push(`<strong>${esc(rulesOnly.map((c) => c.name).join(', '))}</strong> would have been missed
      entirely without the rule pack — the behaviour was indistinguishable from this identity's
      normal work, and only knowing what the operation <em>means</em> catches it. This is the
      reason average precision alone is a poor way to judge the ensemble.`);
  }
  if (modelsOnly.length) {
    bits.push(`<strong>${esc(modelsOnly.map((c) => c.name).join(', '))}</strong> matched no signature
      and was found purely on deviation from baseline — the case rules cannot cover.`);
  }
  return `<div class="note mt small">${bits.join(' ')}</div>`;
}

function ablationVerdict(rows) {
  const shipping = rows.find((r) => r.name.startsWith('ARGUS'));
  const models = rows.find((r) => r.name.startsWith('Ensemble'));
  const rules = rows.find((r) => r.name.startsWith('Rules'));
  const best = rows.slice().sort((a, b) => (b.ap ?? 0) - (a.ap ?? 0))[0];
  if (!shipping || !models || !rules) return '';
  const parts = [];
  parts.push(`Rules alone reach ${fixed(rules.ap, 3)} average precision and models alone ${fixed(models.ap, 3)}; together they reach ${fixed(shipping.ap, 3)}.`);
  parts.push(best.name.startsWith('ARGUS')
    ? 'The shipping configuration is the strongest variant on this corpus.'
    : `Note that "${best.name}" scores higher on average precision here — the ensemble trades some of that for the rule coverage that catches signature-known attacks the models find unremarkable.`);
  return parts.join(' ');
}

function kpi(label, value, sub, tone = '') {
  return `<div class="panel kpi"><div class="kpi-label">${esc(label)}</div>
    <div class="kpi-value ${tone}">${value}</div><div class="kpi-sub">${esc(sub)}</div></div>`;
}

function stat(label, value) {
  return `<div class="panel" style="padding:9px 11px;background:var(--bg-panel-2)">
    <div class="kpi-label">${esc(label)}</div><div class="mono" style="font-size:17px">${value}</div></div>`;
}

function unlabelled() {
  return `<div class="view"><div class="panel" style="padding:40px 26px;text-align:center">
    <h1>No ground truth in this corpus</h1>
    <p class="dim" style="max-width:60ch;margin:8px auto 16px">Precision and recall need labels.
      The loaded events are real or unlabelled logs, so ARGUS scores them but cannot grade itself.
      Generate a synthetic estate with injected campaigns to measure detection quality, then bring
      the tuned settings back to the real corpus.</p>
    <button class="btn btn-primary" data-goto="data">Generate a labelled corpus →</button>
  </div></div>`;
}

export function mount(root, ctx) {
  on(root, 'click', '[data-goto]', (ev, el) => setView(el.dataset.goto));

  const slider = root.querySelector('#threshSlider');
  if (slider) {
    slider.addEventListener('input', () => {
      liveThreshold = Number(slider.value);
      ctx.rerender();
      const again = document.querySelector('#threshSlider');
      if (again) again.focus();
    });
  }

  on(root, 'click', '#exportEval', () => {
    const ev = state.result.evaluation;
    download('argus-evaluation.json', JSON.stringify({
      generatedAt: new Date().toISOString(),
      corpus: { events: state.result.summary.events, seed: state.generator.seed, days: state.generator.days },
      options: state.options,
      auc: ev.overall.auc,
      averagePrecision: ev.overall.ap,
      bestF1: ev.overall.best,
      operatingPoint: confusionAt(Array.from(state.result.risk), state.result.events.map((e) => e.label || 0), liveThreshold ?? state.options.threshold),
      ablation: ev.ablation,
      campaigns: ev.campaigns.map(({ id, name, technique, detected, coverage, latencyMs, totalMalicious, caught }) =>
        ({ id, name, technique, detected, coverage, latencyMs, totalMalicious, caught })),
    }, null, 2));
    toast('Evaluation exported');
  });
}
