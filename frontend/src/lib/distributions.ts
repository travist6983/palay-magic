/**
 * Client-side distribution evaluation.
 *
 * The API ships distribution PARAMETERS, not a grid of probabilities, so the "line" input on the
 * deep-dive page recomputes `P(over)` locally on every keystroke instead of round-tripping to the
 * server (§8, D10). Every function here has to agree with backend/models/distributions.py to the
 * precision a user would notice; src/lib/distributions.test.ts pins the shared cases.
 */

import type { DistributionParams } from './types'

/** log Γ(x) — Lanczos approximation, g=7, n=9. Accurate to ~15 significant figures. */
export function logGamma(x: number): number {
  const g = 7
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ]
  if (x < 0.5) {
    // Reflection: Γ(x)Γ(1-x) = π / sin(πx)
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x)
  }
  x -= 1
  let a = c[0]
  const t = x + g + 0.5
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i)
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a)
}

/** Regularised lower incomplete gamma P(a, x), by series below the mean and continued fraction above. */
export function lowerGamma(a: number, x: number): number {
  if (x <= 0) return 0
  if (x < a + 1) {
    let sum = 1 / a
    let term = sum
    for (let n = 1; n < 500; n++) {
      term *= x / (a + n)
      sum += term
      if (Math.abs(term) < Math.abs(sum) * 1e-15) break
    }
    return sum * Math.exp(-x + a * Math.log(x) - logGamma(a))
  }
  // Lentz's algorithm on the continued fraction for Q(a, x).
  const tiny = 1e-300
  let b = x + 1 - a
  let c = 1 / tiny
  let d = 1 / b
  let h = d
  for (let i = 1; i < 500; i++) {
    const an = -i * (i - a)
    b += 2
    d = an * d + b
    if (Math.abs(d) < tiny) d = tiny
    c = b + an / c
    if (Math.abs(c) < tiny) c = tiny
    d = 1 / d
    const del = d * c
    h *= del
    if (Math.abs(del - 1) < 1e-15) break
  }
  return 1 - Math.exp(-x + a * Math.log(x) - logGamma(a)) * h
}

/** Regularised incomplete beta I_x(a, b), by continued fraction with the standard symmetry swap. */
export function incompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0
  if (x >= 1) return 1
  if (x > (a + 1) / (a + b + 2)) return 1 - incompleteBeta(1 - x, b, a)

  const front =
    Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x)) / a

  let f = 1
  let c = 1
  let d = 0
  for (let i = 0; i <= 300; i++) {
    const m = Math.floor(i / 2)
    let numerator: number
    if (i === 0) numerator = 1
    else if (i % 2 === 0) numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
    else numerator = (-((a + m) * (a + b + m)) * x) / ((a + 2 * m) * (a + 2 * m + 1))

    d = 1 + numerator * d
    if (Math.abs(d) < 1e-30) d = 1e-30
    d = 1 / d
    c = 1 + numerator / c
    if (Math.abs(c) < 1e-30) c = 1e-30
    f *= c * d
    if (Math.abs(1 - c * d) < 1e-15) break
  }
  return front * (f - 1)
}

/** Poisson CDF P(X ≤ k). */
export function poissonCdf(k: number, lambda: number): number {
  const kk = Math.floor(k)
  if (kk < 0) return 0
  if (lambda <= 0) return 1
  return 1 - lowerGamma(kk + 1, lambda)
}

/** Poisson PMF P(X = k). */
export function poissonPmf(k: number, lambda: number): number {
  if (k < 0 || !Number.isInteger(k)) return 0
  if (lambda <= 0) return k === 0 ? 1 : 0
  return Math.exp(-lambda + k * Math.log(lambda) - logGamma(k + 1))
}

/** Negative binomial CDF in the (r, p) parameterisation: P(X ≤ k) = I_p(r, k+1). */
export function nbinomCdf(k: number, r: number, p: number): number {
  const kk = Math.floor(k)
  if (kk < 0) return 0
  return incompleteBeta(p, r, kk + 1)
}

/** Negative binomial PMF. */
export function nbinomPmf(k: number, r: number, p: number): number {
  if (k < 0 || !Number.isInteger(k)) return 0
  return Math.exp(
    logGamma(k + r) - logGamma(r) - logGamma(k + 1) + r * Math.log(p) + k * Math.log1p(-p),
  )
}

/**
 * P(X > line) for any stored distribution.
 *
 * Prop lines carry a half point precisely so a push is impossible. At a whole number the
 * comparison stays strict: an exact match is a push, not a win.
 */
export function probOver(dist: DistributionParams, line: number): number {
  const p = dist.params
  switch (dist.family) {
    case 'negative_binomial':
      return clamp01(1 - nbinomCdf(Math.floor(line), p.r, p.p))
    case 'poisson':
      return clamp01(1 - poissonCdf(Math.floor(line), p.lam))
    case 'bernoulli':
      return line < 1 ? clamp01(p.p) : 0
    case 'empirical_max': {
      const samples: number[] | undefined = p.samples_sorted
      if (samples && samples.length) {
        // The samples are sorted ascending, so a binary search gives the survival share directly.
        let lo = 0
        let hi = samples.length
        while (lo < hi) {
          const mid = (lo + hi) >> 1
          if (samples[mid] > line) hi = mid
          else lo = mid + 1
        }
        return clamp01((samples.length - lo) / samples.length)
      }
      return clamp01(1 - interpCdf(p.quantiles ?? {}, line))
    }
    case 'deterministic': {
      const support: number[] = p.support ?? []
      const probs: number[] = p.probs ?? []
      let total = 0
      for (let i = 0; i < support.length; i++) if (support[i] > line) total += probs[i]
      return clamp01(total)
    }
    default:
      return NaN
  }
}

/** P(X = line). Non-zero only for an integer-valued stat at a whole number. */
export function probExact(dist: DistributionParams, line: number): number {
  if (!dist.integer_valued || !Number.isInteger(line)) return 0
  const p = dist.params
  switch (dist.family) {
    case 'negative_binomial':
      return clamp01(nbinomPmf(line, p.r, p.p))
    case 'poisson':
      return clamp01(poissonPmf(line, p.lam))
    case 'bernoulli':
      return line === 1 ? clamp01(p.p) : line === 0 ? clamp01(1 - p.p) : 0
    default:
      return 0
  }
}

/** P(X < line). Equals 1 − P(over) only when a push is impossible. */
export function probUnder(dist: DistributionParams, line: number): number {
  return clamp01(1 - probOver(dist, line) - probExact(dist, line))
}

/** Quantile function, for redrawing the interval when a user changes the confidence level. */
export function quantile(dist: DistributionParams, q: number): number {
  const p = dist.params
  switch (dist.family) {
    case 'poisson':
    case 'negative_binomial': {
      const mean = dist.mean
      const limit = Math.max(10, Math.ceil(mean * 12 + 40))
      let cumulative = 0
      for (let k = 0; k <= limit; k++) {
        cumulative +=
          dist.family === 'poisson' ? poissonPmf(k, p.lam) : nbinomPmf(k, p.r, p.p)
        if (cumulative >= q) return k
      }
      return limit
    }
    case 'bernoulli':
      return q > 1 - p.p ? 1 : 0
    case 'empirical_max': {
      const samples: number[] | undefined = p.samples_sorted
      if (samples && samples.length) {
        const idx = Math.min(samples.length - 1, Math.max(0, Math.round(q * (samples.length - 1))))
        return samples[idx]
      }
      return interpQuantile(p.quantiles ?? {}, q)
    }
    case 'deterministic': {
      const support: number[] = p.support ?? []
      const probs: number[] = p.probs ?? []
      let cumulative = 0
      for (let i = 0; i < support.length; i++) {
        cumulative += probs[i]
        if (cumulative >= q) return support[i]
      }
      return support[support.length - 1] ?? 0
    }
    default:
      return NaN
  }
}

/** Points for the distribution chart: {x, density} over a sensible range. */
export function densityCurve(dist: DistributionParams, points = 60): { x: number; y: number }[] {
  const p = dist.params
  if (dist.family === 'bernoulli') {
    return [
      { x: 0, y: 1 - p.p },
      { x: 1, y: p.p },
    ]
  }
  if (dist.family === 'deterministic') {
    const support: number[] = p.support ?? []
    const probs: number[] = p.probs ?? []
    return support.map((x, i) => ({ x, y: probs[i] }))
  }
  if (dist.family === 'empirical_max') {
    const samples: number[] = p.samples_sorted ?? []
    if (!samples.length) return []
    return histogram(samples, Math.min(points, 40))
  }

  const lo = 0
  const hi = Math.max(1, Math.ceil(quantile(dist, 0.995)))
  const step = Math.max(1, Math.round((hi - lo) / points))
  const out: { x: number; y: number }[] = []
  for (let k = lo; k <= hi; k += step) {
    let y = 0
    for (let j = k; j < k + step; j++) {
      y += dist.family === 'poisson' ? poissonPmf(j, p.lam) : nbinomPmf(j, p.r, p.p)
    }
    out.push({ x: k, y })
  }
  return out
}

/** Fair American odds for a probability, the way a book would post it before vig. */
export function toAmericanOdds(probability: number): number | null {
  if (!(probability > 0) || !(probability < 1)) return null
  return probability >= 0.5
    ? -Math.round((100 * probability) / (1 - probability))
    : Math.round((100 * (1 - probability)) / probability)
}

/** Format American odds with an explicit sign. */
export function formatOdds(probability: number): string {
  const odds = toAmericanOdds(probability)
  if (odds === null) return '—'
  return odds > 0 ? `+${odds}` : `${odds}`
}

/** A sensible default line: the median, offset to a half point so a push is impossible. */
export function defaultLine(dist: DistributionParams): number {
  const m = dist.median
  if (dist.family === 'bernoulli') return 0.5
  return Number.isInteger(m) ? m + 0.5 : Math.round(m * 2) / 2
}

// --- helpers ---------------------------------------------------------------

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return NaN
  return Math.min(1, Math.max(0, x))
}

function interpQuantile(quantiles: Record<string, number>, q: number): number {
  const xs = Object.keys(quantiles)
    .map(Number)
    .sort((a, b) => a - b)
  if (!xs.length) return NaN
  const ys = xs.map((x) => quantiles[String(x)])
  return interp(q, xs, ys)
}

function interpCdf(quantiles: Record<string, number>, value: number): number {
  const xs = Object.keys(quantiles)
    .map(Number)
    .sort((a, b) => a - b)
  if (!xs.length) return NaN
  const ys = xs.map((x) => quantiles[String(x)])
  return interp(value, ys, xs)
}

function interp(x: number, xs: number[], ys: number[]): number {
  if (x <= xs[0]) return ys[0]
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1]
  for (let i = 1; i < xs.length; i++) {
    if (x <= xs[i]) {
      const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1] || 1)
      return ys[i - 1] + t * (ys[i] - ys[i - 1])
    }
  }
  return ys[ys.length - 1]
}

function histogram(sorted: number[], bins: number): { x: number; y: number }[] {
  const lo = sorted[0]
  const hi = sorted[sorted.length - 1]
  if (hi === lo) return [{ x: lo, y: 1 }]
  const width = (hi - lo) / bins
  const counts = new Array(bins).fill(0)
  for (const v of sorted) {
    const idx = Math.min(bins - 1, Math.floor((v - lo) / width))
    counts[idx] += 1
  }
  return counts.map((c, i) => ({ x: lo + width * (i + 0.5), y: c / sorted.length }))
}
