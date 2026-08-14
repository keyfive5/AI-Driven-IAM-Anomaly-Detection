/**
 * Evaluation.
 *
 * The document shipped with the original project quotes precision 0.89 / recall
 * 0.92 for the hybrid model. Nothing in the repository computes those numbers,
 * and nothing could: the pipeline never held a labelled test set. This module
 * exists so every figure the interface displays is recomputed, in front of the
 * user, from labelled data, with the seed that produced it printed alongside.
 *
 * Reported here:
 *   - ROC AUC (Mann–Whitney form, so ties are handled exactly)
 *   - Average precision + the full PR curve
 *   - Operating-point confusion matrix at the live threshold
 *   - Alerts per 1,000 events — the metric that decides whether a SOC can
 *     actually run the thing
 *   - Time-to-detect per injected campaign
 *   - A detector ablation, because "the ensemble helps" is a claim, not a given
 */

/** ROC AUC via rank statistics; equals P(score(pos) > score(neg)). */
export function rocAuc(scores, labels) {
  const n = scores.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[a] - scores[b]);
  const ranks = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && scores[idx[j + 1]] === scores[idx[i]]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[idx[k]] = avg;
    i = j + 1;
  }
  let sumPos = 0;
  let nPos = 0;
  for (let k = 0; k < n; k++) {
    if (labels[k] === 1) { sumPos += ranks[k]; nPos++; }
  }
  const nNeg = n - nPos;
  if (!nPos || !nNeg) return null;
  return (sumPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

/**
 * Precision–recall curve. Returns points plus average precision.
 * Points are thinned for plotting but AP is computed on every threshold.
 */
export function prCurve(scores, labels, maxPoints = 240) {
  const n = scores.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const total = labels.reduce((s, v) => s + (v === 1 ? 1 : 0), 0);
  if (!total) return { points: [], ap: null, best: null };

  let tp = 0;
  let fp = 0;
  let prevRecall = 0;
  let ap = 0;
  let best = { f1: 0 };
  const raw = [];
  for (let k = 0; k < n; k++) {
    if (labels[order[k]] === 1) tp++; else fp++;
    const precision = tp / (tp + fp);
    const recall = tp / total;
    ap += (recall - prevRecall) * precision;
    prevRecall = recall;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    if (f1 > best.f1) {
      best = { f1, precision, recall, threshold: scores[order[k]], flagged: k + 1 };
    }
    raw.push({ recall, precision, threshold: scores[order[k]], flagged: k + 1 });
  }
  const step = Math.max(1, Math.floor(raw.length / maxPoints));
  const points = raw.filter((_, i) => i % step === 0 || i === raw.length - 1);
  return { points, ap, best, positives: total };
}

/** ROC curve points for plotting. */
export function rocCurve(scores, labels, maxPoints = 240) {
  const n = scores.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const pos = labels.reduce((s, v) => s + (v === 1 ? 1 : 0), 0);
  const neg = n - pos;
  if (!pos || !neg) return [];
  let tp = 0;
  let fp = 0;
  const raw = [{ fpr: 0, tpr: 0 }];
  for (let k = 0; k < n; k++) {
    if (labels[order[k]] === 1) tp++; else fp++;
    raw.push({ fpr: fp / neg, tpr: tp / pos });
  }
  const step = Math.max(1, Math.floor(raw.length / maxPoints));
  return raw.filter((_, i) => i % step === 0 || i === raw.length - 1);
}

/** Confusion matrix and derived rates at a fixed decision threshold. */
export function confusionAt(scores, labels, threshold) {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  for (let i = 0; i < scores.length; i++) {
    const flagged = scores[i] >= threshold;
    const positive = labels[i] === 1;
    if (flagged && positive) tp++;
    else if (flagged && !positive) fp++;
    else if (!flagged && positive) fn++;
    else tn++;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  const fpr = fp + tn ? fp / (fp + tn) : 0;
  return {
    tp, fp, fn, tn, precision, recall, f1, fpr,
    flagged: tp + fp,
    alertsPerThousand: scores.length ? ((tp + fp) / scores.length) * 1000 : 0,
  };
}

/**
 * Per-campaign detection outcome: was it caught, on which event, and how long
 * after the campaign began. Detection latency is the number an incident
 * responder actually cares about.
 */
export function campaignDetection(events, risks, threshold, campaigns, parts = null) {
  return campaigns.map((c) => {
    let firstTs = null;
    let detectedTs = null;
    let detectedIdx = null;
    let caught = 0;
    let totalMalicious = 0;
    let eventsBefore = 0;
    // Would the models have found this without the rule pack, and vice versa?
    // This is the question the ablation table cannot answer per campaign.
    let byModels = 0;
    let byRules = 0;
    for (let i = 0; i < events.length; i++) {
      const e = events[i];
      if (e.campaign !== c.id) continue;
      if (firstTs === null) firstTs = e.ts;
      if (e.label === 1) {
        totalMalicious++;
        if (risks[i] >= threshold) {
          caught++;
          if (detectedTs === null) { detectedTs = e.ts; detectedIdx = i; }
        }
        if (parts) {
          if (parts.modelRisk[i] >= threshold) byModels++;
          if (parts.ruleRisk[i] >= threshold) byRules++;
        }
        if (detectedTs === null) eventsBefore++;
      }
    }
    return {
      ...c,
      byModels,
      byRules,
      attribution: !parts ? null : byModels && byRules ? 'both' : byRules ? 'rules only' : byModels ? 'models only' : 'missed',
      detected: detectedTs !== null,
      detectedTs,
      detectedIdx,
      caught,
      totalMalicious,
      coverage: totalMalicious ? caught / totalMalicious : 0,
      latencyMs: detectedTs !== null && firstTs !== null ? detectedTs - firstTs : null,
      eventsBeforeDetection: detectedTs !== null ? eventsBefore : null,
    };
  });
}

/**
 * Full evaluation of one score vector against ground truth.
 */
export function evaluateScores(scores, labels, threshold) {
  const auc = rocAuc(scores, labels);
  const pr = prCurve(scores, labels);
  const cm = confusionAt(scores, labels, threshold);
  return { auc, ap: pr.ap, best: pr.best, pr: pr.points, roc: rocCurve(scores, labels), confusion: cm, positives: pr.positives };
}

/**
 * Detector ablation. Each variant is evaluated at the threshold that maximises
 * its own F1, so no variant is handicapped by a threshold tuned for another.
 */
export function ablation(variants, labels) {
  return variants.map((v) => {
    const auc = rocAuc(v.scores, labels);
    const pr = prCurve(v.scores, labels);
    const best = pr.best || { f1: 0, precision: 0, recall: 0, threshold: 0 };
    const cm = confusionAt(v.scores, labels, best.threshold ?? 0);
    return {
      name: v.name,
      note: v.note,
      auc,
      ap: pr.ap,
      f1: best.f1,
      precision: best.precision,
      recall: best.recall,
      threshold: best.threshold,
      alertsPerThousand: cm.alertsPerThousand,
    };
  });
}
