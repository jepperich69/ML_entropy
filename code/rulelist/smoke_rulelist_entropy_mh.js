#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

/*
Dependency-free smoke test for the Pub_ML_Entropy classifier idea.

It compares:
  1. Exact enumeration of short binary rule lists.
  2. Entropy-guided warm start from one-rule Boltzmann scores.
  3. Metropolis-Hastings polishing over feasible rule-list structures.

The point is not to be competitive yet. The point is to verify that the
research loop is concrete: exact small benchmark, entropy warm start,
Boltzmann/MH search, and held-out ML scoring.
*/

const CONFIG = {
  seed: 20260423,
  nTrain: 96,
  nTest: 256,
  nFeatures: 6,
  maxDepth: 3,
  regularization: 0.015,
  warmBeta: 35.0,
  mhBeta: 120.0,
  mhSteps: 5000,
};

function makeRng(seed) {
  let state = seed >>> 0;
  return function rand() {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function makeDataset(n, nFeatures, rng) {
  const X = [];
  const y = [];
  for (let i = 0; i < n; i += 1) {
    const row = [];
    for (let j = 0; j < nFeatures; j += 1) {
      const p = 0.35 + 0.08 * (j % 3);
      row.push(rng() < p ? 1 : 0);
    }

    // Ground truth is deliberately rule-list-like but not trivial.
    let label = 0;
    if (row[0] === 1 && row[1] === 1) label = 1;
    else if (row[2] === 1 && row[3] === 0) label = 1;
    else if (row[4] === 1 && row[5] === 1) label = 0;

    // Small label noise prevents the exact objective from being degenerate.
    if (rng() < 0.04) label = 1 - label;

    X.push(row);
    y.push(label);
  }
  return { X, y };
}

function makeAntecedents(nFeatures) {
  const literals = [];
  for (let feature = 0; feature < nFeatures; feature += 1) {
    literals.push([{ feature, value: 1 }]);
    literals.push([{ feature, value: 0 }]);
  }

  const antecedents = [...literals];
  for (let i = 0; i < literals.length; i += 1) {
    for (let j = i + 1; j < literals.length; j += 1) {
      const a = literals[i][0];
      const b = literals[j][0];
      if (a.feature === b.feature) continue;
      antecedents.push([a, b]);
    }
  }
  return antecedents.map((lits, idx) => ({
    id: idx,
    literals: lits,
    name: lits.map((lit) => `x${lit.feature}=${lit.value}`).join(" & "),
  }));
}

function covers(row, antecedent) {
  return antecedent.literals.every((lit) => row[lit.feature] === lit.value);
}

function majorityLabel(labels) {
  if (labels.length === 0) return 0;
  const positives = labels.reduce((sum, value) => sum + value, 0);
  return positives * 2 >= labels.length ? 1 : 0;
}

function fitPredictionsForOrder(order, antecedents, data) {
  const remaining = new Set(data.y.map((_, i) => i));
  const predictions = [];

  for (const ruleId of order) {
    const captured = [];
    for (const i of remaining) {
      if (covers(data.X[i], antecedents[ruleId])) captured.push(i);
    }
    const pred = majorityLabel(captured.map((i) => data.y[i]));
    predictions.push(pred);
    for (const i of captured) remaining.delete(i);
  }

  const defaultPred = majorityLabel([...remaining].map((i) => data.y[i]));
  return { predictions, defaultPred };
}

function predictOne(row, model, antecedents) {
  for (let k = 0; k < model.order.length; k += 1) {
    const ruleId = model.order[k];
    if (covers(row, antecedents[ruleId])) return model.predictions[k];
  }
  return model.defaultPred;
}

function evaluateOrder(order, antecedents, data, regularization) {
  const fitted = fitPredictionsForOrder(order, antecedents, data);
  const model = { order: [...order], ...fitted };
  let mistakes = 0;
  for (let i = 0; i < data.y.length; i += 1) {
    if (predictOne(data.X[i], model, antecedents) !== data.y[i]) mistakes += 1;
  }
  const error = mistakes / data.y.length;
  return {
    model,
    error,
    objective: error + regularization * order.length,
  };
}

function scoreModel(model, antecedents, data) {
  let correct = 0;
  for (let i = 0; i < data.y.length; i += 1) {
    if (predictOne(data.X[i], model, antecedents) === data.y[i]) correct += 1;
  }
  return correct / data.y.length;
}

function exactSearch(antecedents, data, maxDepth, regularization) {
  let best = evaluateOrder([], antecedents, data, regularization);
  let checked = 1;

  function visit(prefix, used) {
    if (prefix.length > 0) {
      const current = evaluateOrder(prefix, antecedents, data, regularization);
      checked += 1;
      if (current.objective < best.objective) best = current;
    }
    if (prefix.length === maxDepth) return;

    for (const ant of antecedents) {
      if (used.has(ant.id)) continue;
      used.add(ant.id);
      prefix.push(ant.id);
      visit(prefix, used);
      prefix.pop();
      used.delete(ant.id);
    }
  }

  visit([], new Set());
  return { ...best, checked };
}

function oneRuleScores(antecedents, data, regularization) {
  return antecedents.map((ant) => ({
    id: ant.id,
    objective: evaluateOrder([ant.id], antecedents, data, regularization).objective,
  }));
}

function entropyWarmStart(antecedents, data, maxDepth, regularization, beta) {
  const oneRule = oneRuleScores(antecedents, data, regularization);
  const maxUtility = Math.max(...oneRule.map((r) => -r.objective));
  const weights = new Map(
    oneRule.map((r) => [r.id, Math.exp(beta * (-r.objective - maxUtility))])
  );

  const order = [];
  const used = new Set();
  while (order.length < maxDepth) {
    let bestId = null;
    let bestObjective = Infinity;
    let bestScore = -Infinity;

    for (const ant of antecedents) {
      if (used.has(ant.id)) continue;
      const candidate = [...order, ant.id];
      const current = evaluateOrder(candidate, antecedents, data, regularization);
      const entropyScore = Math.log(weights.get(ant.id) + 1e-300);
      const score = -current.objective + 0.02 * entropyScore;
      if (score > bestScore || (score === bestScore && current.objective < bestObjective)) {
        bestId = ant.id;
        bestObjective = current.objective;
        bestScore = score;
      }
    }

    if (bestId === null) break;
    order.push(bestId);
    used.add(bestId);
  }

  return evaluateOrder(order, antecedents, data, regularization);
}

function randomUnusedRule(antecedents, used, rng) {
  const available = antecedents.filter((ant) => !used.has(ant.id));
  if (available.length === 0) return null;
  return available[Math.floor(rng() * available.length)].id;
}

function proposeMove(order, antecedents, maxDepth, rng) {
  const next = [...order];
  const used = new Set(next);
  const move = rng();

  if (move < 0.25 && next.length > 0) {
    const pos = Math.floor(rng() * next.length);
    next.splice(pos, 1);
    return next;
  }

  if (move < 0.50 && next.length < maxDepth) {
    const id = randomUnusedRule(antecedents, used, rng);
    if (id !== null) next.splice(Math.floor(rng() * (next.length + 1)), 0, id);
    return next;
  }

  if (move < 0.80 && next.length > 0) {
    const pos = Math.floor(rng() * next.length);
    used.delete(next[pos]);
    const id = randomUnusedRule(antecedents, used, rng);
    if (id !== null) next[pos] = id;
    return next;
  }

  if (next.length > 1) {
    const a = Math.floor(rng() * next.length);
    let b = Math.floor(rng() * next.length);
    if (a === b) b = (b + 1) % next.length;
    [next[a], next[b]] = [next[b], next[a]];
  }
  return next;
}

function mhPolish(initial, antecedents, data, maxDepth, regularization, beta, steps, rng) {
  let current = initial;
  let best = initial;
  let accepted = 0;

  for (let step = 0; step < steps; step += 1) {
    const proposalOrder = proposeMove(current.model.order, antecedents, maxDepth, rng);
    const proposal = evaluateOrder(proposalOrder, antecedents, data, regularization);
    const delta = proposal.objective - current.objective;
    if (delta <= 0 || rng() < Math.exp(-beta * delta)) {
      current = proposal;
      accepted += 1;
      if (current.objective < best.objective) best = current;
    }
  }

  return { ...best, accepted, steps };
}

function formatModel(model, antecedents) {
  const rows = [];
  for (let i = 0; i < model.order.length; i += 1) {
    rows.push(`if ${antecedents[model.order[i]].name} then ${model.predictions[i]}`);
  }
  rows.push(`else ${model.defaultPred}`);
  return rows.join("; ");
}

function main() {
  const rng = makeRng(CONFIG.seed);
  const train = makeDataset(CONFIG.nTrain, CONFIG.nFeatures, rng);
  const test = makeDataset(CONFIG.nTest, CONFIG.nFeatures, rng);
  const antecedents = makeAntecedents(CONFIG.nFeatures);

  const exactStart = Date.now();
  const exact = exactSearch(antecedents, train, CONFIG.maxDepth, CONFIG.regularization);
  const exactMs = Date.now() - exactStart;

  const warmStart = Date.now();
  const warm = entropyWarmStart(
    antecedents,
    train,
    CONFIG.maxDepth,
    CONFIG.regularization,
    CONFIG.warmBeta
  );
  const warmMs = Date.now() - warmStart;

  const mhStart = Date.now();
  const mh = mhPolish(
    warm,
    antecedents,
    train,
    CONFIG.maxDepth,
    CONFIG.regularization,
    CONFIG.mhBeta,
    CONFIG.mhSteps,
    rng
  );
  const mhMs = Date.now() - mhStart;

  const exactTestAccuracy = scoreModel(exact.model, antecedents, test);
  const warmTestAccuracy = scoreModel(warm.model, antecedents, test);
  const mhTestAccuracy = scoreModel(mh.model, antecedents, test);

  console.log("Pub_ML_Entropy rule-list smoke test");
  console.log("=====================================");
  console.log(`train rows: ${CONFIG.nTrain}, test rows: ${CONFIG.nTest}`);
  console.log(`features: ${CONFIG.nFeatures}, antecedents: ${antecedents.length}`);
  console.log(`max depth: ${CONFIG.maxDepth}, regularization: ${CONFIG.regularization}`);
  console.log("");
  console.log("Exact small benchmark");
  console.log(`  checked rule lists: ${exact.checked}`);
  console.log(`  train objective:    ${exact.objective.toFixed(4)}`);
  console.log(`  train error:        ${exact.error.toFixed(4)}`);
  console.log(`  test accuracy:      ${exactTestAccuracy.toFixed(4)}`);
  console.log(`  runtime:            ${exactMs} ms`);
  console.log(`  model:              ${formatModel(exact.model, antecedents)}`);
  console.log("");
  console.log("Entropy warm start");
  console.log(`  train objective:    ${warm.objective.toFixed(4)}`);
  console.log(`  train error:        ${warm.error.toFixed(4)}`);
  console.log(`  test accuracy:      ${warmTestAccuracy.toFixed(4)}`);
  console.log(`  runtime:            ${warmMs} ms`);
  console.log(`  model:              ${formatModel(warm.model, antecedents)}`);
  console.log("");
  console.log("MH/Boltzmann polish");
  console.log(`  train objective:    ${mh.objective.toFixed(4)}`);
  console.log(`  train error:        ${mh.error.toFixed(4)}`);
  console.log(`  test accuracy:      ${mhTestAccuracy.toFixed(4)}`);
  console.log(`  accepted moves:     ${mh.accepted}/${mh.steps}`);
  console.log(`  runtime:            ${mhMs} ms`);
  console.log(`  model:              ${formatModel(mh.model, antecedents)}`);
  console.log("");

  const gap = mh.objective - exact.objective;
  const resultsDir = path.join(__dirname, "results");
  fs.mkdirSync(resultsDir, { recursive: true });
  const csvPath = path.join(resultsDir, "smoke_results.csv");
  const rows = [
    [
      "dataset",
      "method",
      "train_objective",
      "train_error",
      "test_accuracy",
      "runtime_ms",
      "checked_or_steps",
      "accepted_moves",
      "model",
    ],
    [
      "synthetic_rulelist",
      "exact",
      exact.objective.toFixed(6),
      exact.error.toFixed(6),
      exactTestAccuracy.toFixed(6),
      String(exactMs),
      String(exact.checked),
      "",
      formatModel(exact.model, antecedents),
    ],
    [
      "synthetic_rulelist",
      "entropy_warm_start",
      warm.objective.toFixed(6),
      warm.error.toFixed(6),
      warmTestAccuracy.toFixed(6),
      String(warmMs),
      "",
      "",
      formatModel(warm.model, antecedents),
    ],
    [
      "synthetic_rulelist",
      "mh_polish",
      mh.objective.toFixed(6),
      mh.error.toFixed(6),
      mhTestAccuracy.toFixed(6),
      String(mhMs),
      String(mh.steps),
      String(mh.accepted),
      formatModel(mh.model, antecedents),
    ],
  ];
  fs.writeFileSync(
    csvPath,
    rows
      .map((row) => row.map((value) => `"${String(value).replace(/"/g, '""')}"`).join(","))
      .join("\n") + "\n",
    "utf8"
  );
  console.log(`CSV results:         ${csvPath}`);

  if (gap <= 1e-12) {
    console.log("SMOKE PASS: MH route matched the exact training objective.");
  } else if (gap <= 0.02) {
    console.log(`SMOKE PASS: MH route was near exact. objective gap=${gap.toFixed(4)}`);
  } else {
    console.log(`SMOKE CHECK: MH route ran but objective gap is ${gap.toFixed(4)}.`);
  }
}

main();
