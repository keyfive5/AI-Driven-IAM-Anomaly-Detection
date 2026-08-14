import { esc, num } from '../dom.js';
import { FEATURES } from '../../core/features.js';
import { RULES } from '../../core/rules.js';

/** In-app documentation. Same source of truth as the engine — the tables below
 *  are generated from the feature catalogue and rule pack themselves, so they
 *  cannot drift from what actually runs. */

export function render() {
  const groups = [...new Set(FEATURES.map((f) => f.group))];
  const bySeverity = [...RULES].sort((a, b) => b.severity - a.severity || b.risk - a.risk);

  return `<div class="view">
    <div class="view-head">
      <div>
        <h1>How the engine works</h1>
        <div class="sub">Everything below is read directly out of the running code — the feature
          catalogue and rule pack on this page are the same objects the detectors use.</div>
      </div>
    </div>

    <div class="panel">
      <h3>Pipeline</h3>
      <div class="row small mono" style="gap:6px;flex-wrap:wrap;margin-bottom:12px">
        ${['normalise', 'streaming features', 'Isolation Forest', 'robust z', 'behavioural surprisal',
          'rank fusion', 'rule floors', 'correlation', 'evaluation']
          .map((s, i) => `${i ? '<span class="chain-arrow">→</span>' : ''}<span class="chain-step">${esc(s)}</span>`).join('')}
      </div>
      <p class="small dim">Logs from any supported source are flattened to one flat event shape.
        A single streaming pass builds each identity's behavioural profile and emits
        ${FEATURES.length} features per event using <em>only</em> the events before it — no
        lookahead, so a score could genuinely have been produced live. Three detectors then score
        every event, their outputs are combined in rank space, and the rule pack applies risk floors
        for behaviour that is wrong regardless of how common it is.</p>
    </div>

    <div class="grid grid-3 mt">
      ${detector('Isolation Forest', 'joint weirdness',
        `Random axis-aligned splits isolate outliers in fewer partitions than normal points. Score is
         2^(−E[h(x)]/c(ψ)) over the ensemble. Written from scratch here, including the split-path
         attribution that tells each alert which features isolated it.`)}
      ${detector('Robust z', 'single-axis extremeness',
        `Median and MAD per feature, so the estimate is not dragged around by the very outliers it is
         meant to find. Event score is the mean of its three largest directional z-values.`)}
      ${detector('Behavioural surprisal', 'self-baseline',
        `A per-identity Markov chain over operations, plus operation and hour surprisal against that
         identity's own history. This replaces the original project's LSTM: it needs no training,
         updates online, and its output is directly interpretable as "how unexpected was this step".`)}
    </div>

    <div class="panel mt">
      <div class="panel-head"><h2>Risk scale</h2><span class="panel-note">tail percentile, not a probability</span></div>
      <p class="small dim">The fused rank is converted with risk = 100·log₁₀(1/p)/3.5, where p is the
        share of events at least as unusual. That makes the number mean the same thing on a
        2,000-event corpus and a 200,000-event one:</p>
      <div class="table-wrap" style="max-width:520px">
        <table><thead><tr><th>Risk</th><th>Meaning</th><th>Severity</th></tr></thead><tbody>
          <tr><td class="mono">29</td><td>top 10% of events</td><td><span class="tag sev sev-info">info</span></td></tr>
          <tr><td class="mono">43</td><td>top 5%</td><td><span class="tag sev sev-low">low</span></td></tr>
          <tr><td class="mono">57</td><td>top 1%</td><td><span class="tag sev sev-medium">medium</span></td></tr>
          <tr><td class="mono">71</td><td>top 0.3%</td><td><span class="tag sev sev-high">high</span></td></tr>
          <tr><td class="mono">86</td><td>top 0.1%</td><td><span class="tag sev sev-critical">critical</span></td></tr>
        </tbody></table>
      </div>
    </div>

    <div class="panel mt">
      <div class="panel-head">
        <h2>Feature catalogue</h2>
        <span class="panel-note">${FEATURES.length} features · direction shows what counts as suspicious</span>
      </div>
      ${groups.map((g) => `
        <h3 style="margin-top:14px">${esc(g)}</h3>
        <div class="table-wrap"><table>
          <thead><tr><th style="width:210px">Feature</th><th style="width:90px">Direction</th><th>What it measures</th></tr></thead>
          <tbody>${FEATURES.filter((f) => f.group === g).map((f) => `
            <tr><td class="mono small">${esc(f.label)}</td>
              <td class="small faint">${f.dir === 1 ? 'higher = worse' : f.dir === -1 ? 'lower = worse' : 'joint only'}</td>
              <td class="small dim">${esc(f.desc)}</td></tr>`).join('')}
          </tbody></table></div>`).join('')}
    </div>

    <div class="panel mt">
      <div class="panel-head">
        <h2>Rule pack</h2>
        <span class="panel-note">${RULES.length} rules · risk is a floor, never a cap</span>
      </div>
      <p class="small dim">Broad-surface rules fire on <em>circumstances</em>, not on the verb alone.
        An identity administrator creating a user at 11:00 from the office with MFA is doing their
        job; the same call at 03:00 from a new address is an incident. Only operations that are wrong
        under every circumstance fire unconditionally.</p>
      <div class="table-wrap"><table>
        <thead><tr><th>Rule</th><th>Tactic</th><th>Technique</th><th class="right">Severity</th><th class="right">Risk floor</th></tr></thead>
        <tbody>${bySeverity.map((r) => `
          <tr><td>${esc(r.name)}<div class="faint small mono">${esc(r.id)}</div></td>
            <td><span class="tag tag-tactic">${esc(r.tactic)}</span></td>
            <td class="mono small faint">${esc(r.technique)}</td>
            <td class="right mono">${r.severity}</td>
            <td class="right mono">${r.risk}</td></tr>`).join('')}
        </tbody></table></div>
    </div>

    <div class="panel mt">
      <div class="panel-head"><h2>What this rebuild changed</h2></div>
      <div class="table-wrap"><table>
        <thead><tr><th style="width:30%">Original (2025)</th><th>ARGUS</th></tr></thead>
        <tbody>
          ${[
            ['Metrics quoted in the write-up with no code that computes them',
             'Every metric recomputed live from labelled data, with the seed shown and the ablation published — including where a single detector beats the ensemble'],
            ['Group statistics computed over the whole DataFrame, so each event was scored using events from its own future',
             'Strictly streaming: event i is scored from events 0..i−1 only'],
            ['Synthetic generator emitted uniformly random events, making "anomaly" trivially separable',
             'Structured population with roles, working hours, task sequences and deliberate benign confusers'],
            ['Random Forest trained on labels produced by the Isolation Forest — it could only learn to imitate it',
             'Three genuinely independent detectors combined in rank space, plus a rule pack for known-bad'],
            ['An alert was a row in a table with a score',
             'Every alert states which features drove it, which rules fired, and what to do next; alerts correlate into incidents along the kill chain'],
            ['Tkinter desktop app, ~10 Python dependencies, TensorFlow for an LSTM that was never wired in',
             'Runs in any browser with zero dependencies and no build step; logs never leave the machine'],
          ].map(([a, b]) => `<tr><td class="small dim">${esc(a)}</td><td class="small">${esc(b)}</td></tr>`).join('')}
        </tbody>
      </table></div>
      <p class="small faint mt">The original code is preserved unchanged in <code>/src</code> at the
        repository root; nothing here overwrites it.</p>
    </div>
  </div>`;
}

function detector(name, tagline, body) {
  return `<div class="panel">
    <h2 style="font-size:13px">${esc(name)}</h2>
    <div class="small" style="color:var(--accent);margin:2px 0 8px">${esc(tagline)}</div>
    <p class="small dim" style="margin:0">${esc(body)}</p>
  </div>`;
}

export function mount() {}
