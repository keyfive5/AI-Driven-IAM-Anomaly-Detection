/**
 * Analysis worker.
 *
 * The engine is pure arithmetic over tens of thousands of events. Run on the
 * main thread it blocks the page for as long as it takes — and a blocked page
 * cannot repaint a progress bar, cannot respond to a cancel button, and cannot
 * let the analyst look at anything else while they wait. Chunking the loops
 * only softens that; it does not fix it, because the work still competes with
 * the interface for the one thread that draws it.
 *
 * So the pipeline runs here instead. The document stays interactive from the
 * first paint, progress messages arrive on time, and cancelling is a
 * `terminate()` rather than a request the busy thread may not notice.
 *
 * Only what the interface actually reads is posted back; the feature matrix,
 * forest and per-event context stay in the worker and die with it.
 */

import { analyse } from './pipeline.js';
import { generateCorpus } from './generate.js';

self.onmessage = async (ev) => {
  const { type, events, generator, options } = ev.data || {};
  if (type !== 'analyse') return;

  const post = (pct, message) => self.postMessage({ type: 'progress', pct, message });

  try {
    let corpus = events;
    let meta = ev.data.meta || null;

    // Generating in here keeps a second multi-second synchronous block off the
    // main thread, and avoids cloning the corpus across the boundary twice.
    if (!corpus && generator) {
      post(2, 'Generating synthetic estate…');
      const built = generateCorpus(generator);
      corpus = built.events;
      meta = built.meta;
    }
    if (!corpus?.length) throw new Error('No events to analyse.');

    const result = await analyse(corpus, {
      ...options,
      campaigns: meta?.campaigns || options?.campaigns || [],
    }, post);

    // `signal` is not structured-cloneable, and cancellation in worker mode is
    // termination rather than a token, so it never crosses the boundary.
    const { signal, ...cleanOptions } = result.options || {};

    self.postMessage({
      type: 'done',
      meta,
      result: {
        events: result.events,
        risk: result.risk,
        modelRisk: result.modelRisk,
        ruleRisk: result.ruleRisk,
        ruleHits: result.ruleHits,
        detectorRanks: result.detectorRanks,
        alerts: result.alerts,
        incidents: result.incidents,
        identities: result.identities,
        summary: result.summary,
        evaluation: result.evaluation,
        options: cleanOptions,
        elapsedMs: result.elapsedMs,
      },
    });
  } catch (err) {
    self.postMessage({ type: 'error', message: err?.message || String(err) });
  }
};
