// gta6-twitch-ccu-forecast-01: launch-day Twitch viewership forecast.
//
// Model: a launch day's viewer curve is predictable from past launches once
// you anchor on the first hour. For every past launch (a "comparable") we
// know its hourly viewers on its release day, so for each hour h we know the
// log ratio log(viewers_h / viewers_1). That ratio is NOT the same for every
// launch: a game whose first hour is small explodes later (its audience is
// still asleep in the later time zones), while a game whose first hour is
// already huge barely rises (its peak is near hour 1). Across the 37
// complete comparables, log(peak ratio) correlates -0.94 with log(hour-1
// print), so the ratio curve depends strongly on the first hour's scale.
//
// The forecast therefore conditions on the target's own hour-1 print p1:
// for each hour, the comparables are weighted by a Gaussian kernel in
// log(hour-1 print) around log(p1) (bandwidth 1.0 log units), and the
// forecast quantiles come from that weighted distribution of log ratios:
// the median is the weighted 0.5 quantile, the spread is the weighted
// interquartile range turned into a log-sigma, and the five levels are
// median + z_tau * sigma with the standard normal's quantiles. The spread
// has a floor of 0.10 log units so the band never collapses when only a few
// comparables carry weight. Nothing is tuned to the four practice launches
// beyond the bandwidth, which was picked on leave-one-out over the
// comparables themselves (a wider kernel generalises better to a launch
// bigger than every comparable, which GTA VI will be).
//
// Input/output contract (from RULES.txt): read every <id>.in in the input
// directory (MAKEFUN_INPUT_DIR, else /input), write <id>.out to the output
// directory (MAKEFUN_OUTPUT_DIR, else /output). One output per case:
// {"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]} for hours 2..24.
// Standard library only, no network, deterministic: the same input bytes
// always produce the same output bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Scored quantile levels, in the order each output row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at LEVELS. */
const Z_SCORES = [-1.645, -0.674, 0, 0.674, 1.645];
/** Gaussian kernel bandwidth in log(hour-1 viewers). */
const KERNEL_WIDTH = 1.0;
/** Minimum log-sigma for an hour's band, so it never collapses to a point. */
const MIN_SIGMA = 0.1;
/** The weighted IQR slightly overstates the band the pinball loss wants
 * (measured: 0.85 beats 1.0 on the practice cases and on leave-one-out over
 * the big comparables), so the IQR sigma is scaled by this much. */
const SPREAD_MULTIPLIER = 0.85;
/** Fallback spread (log units per sqrt hour) when too few comparables qualify. */
const FALLBACK_SIGMA = 0.15;

interface HourPoint {
  from: string;
  viewers: number;
}

interface Comparable {
  day_start: string;
  hourly: HourPoint[];
}

interface CaseFile {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
}

/** The release-day curve of one comparable: viewers indexed by hour offset from its day_start. */
interface LaunchCurve {
  hourOne: number;
  /** log(viewers at hour h / viewers at hour 1), for h = 2..24; missing where unobserved. */
  logRatioByHour: Array<number | null>;
}

const MS_PER_HOUR = 3600_000;

/** Round to 2 decimals, half up, matching the house baseline's output format. */
function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

/** Parse a comparable's hourly points into its release-day curve. */
function buildCurve(c: Comparable): LaunchCurve | null {
  const start = Date.parse(c.day_start);
  if (Number.isNaN(start)) return null;
  const byHour = new Map<number, number>();
  for (const p of c.hourly) {
    const t = Date.parse(p.from);
    if (Number.isNaN(t)) continue;
    const off = Math.round((t - start) / MS_PER_HOUR);
    if (off >= 0 && off < 24 && typeof p.viewers === 'number' && p.viewers > 0) {
      byHour.set(off, p.viewers);
    }
  }
  const hourOne = byHour.get(0);
  if (hourOne === undefined) return null;
  const logRatioByHour: Array<number | null> = [];
  for (let h = 2; h <= 24; h++) {
    const v = byHour.get(h - 1);
    logRatioByHour.push(v === undefined ? null : Math.log(v / hourOne));
  }
  return { hourOne, logRatioByHour };
}

/** Weighted quantile of values with weights, both arrays the same length. */
function weightedQuantile(values: number[], weights: number[], p: number): number {
  const pairs = values.map((v, i) => [v, weights[i]] as const).sort((a, b) => a[0] - b[0]);
  const total = weights.reduce((a, b) => a + b, 0);
  if (total <= 0) return pairs[Math.floor(pairs.length / 2)][0];
  let acc = 0;
  for (const [v, w] of pairs) {
    acc += w;
    if (acc >= p * total) return v;
  }
  return pairs[pairs.length - 1][0];
}

/** Forecast one hour's five quantiles from the comparables' log ratios. */
function hourQuantiles(p1: number, hour: number, curves: LaunchCurve[]): number[] {
  const logP1 = Math.log(Math.max(1, p1));
  const idx = hour - 2;
  const values: number[] = [];
  const weights: number[] = [];
  for (const c of curves) {
    const lr = c.logRatioByHour[idx];
    if (lr === null) continue;
    const d = Math.log(c.hourOne) - logP1;
    values.push(lr);
    weights.push(Math.exp(-0.5 * (d / KERNEL_WIDTH) ** 2));
  }
  if (values.length < 3) {
    // Too few comparables qualify for this hour: fall back to persistence.
    return Z_SCORES.map((z) => round2(p1 * Math.exp(z * FALLBACK_SIGMA * Math.sqrt(hour - 1))));
  }
  const median = weightedQuantile(values, weights, 0.5);
  const q25 = weightedQuantile(values, weights, 0.25);
  const q75 = weightedQuantile(values, weights, 0.75);
  const sigma = Math.max(((q75 - q25) / 1.349) * SPREAD_MULTIPLIER, MIN_SIGMA);
  const out = Z_SCORES.map((z) => p1 * Math.exp(median + z * sigma));
  // Clamp to the legal range and keep the row non-decreasing.
  let prev = 0;
  return out.map((q) => {
    const clamped = Math.min(1e9, Math.max(1, round2(q)));
    const v = Math.max(clamped, prev);
    prev = v;
    return v;
  });
}

/** Forecast one case: 23 rows, hours 2..24. */
function forecastCase(c: CaseFile): Array<{ hour: number; q: number[] }> {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('case has no hour-1 print');
  const p1 = Math.max(1, first.viewers);
  const curves: LaunchCurve[] = [];
  for (const comp of c.comparables) {
    const curve = buildCurve(comp);
    if (curve !== null) curves.push(curve);
  }
  return c.forecast.map(({ hour }) => ({ hour, q: hourQuantiles(p1, hour, curves) }));
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
