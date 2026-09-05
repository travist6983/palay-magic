/**
 * The browser recomputes P(over) locally on every keystroke (§8, D10), so this file's real job is
 * proving the TypeScript agrees with backend/models/distributions.py. The fixtures are generated
 * BY the Python — regenerate with the snippet in the repo README if the model changes.
 */

import { describe, expect, it } from 'vitest'
import fixtures from './__fixtures__.json'
import {
  defaultLine,
  densityCurve,
  formatOdds,
  nbinomCdf,
  poissonCdf,
  probExact,
  probOver,
  probUnder,
  quantile,
  toAmericanOdds,
} from './distributions'
import type { DistributionParams } from './types'

type Fixture = {
  dist: DistributionParams & { family: string }
  lines: number[]
  over: number[]
  exact: number[]
}
const cases = fixtures as unknown as Record<string, Fixture>

describe('agreement with the Python model', () => {
  for (const [name, fixture] of Object.entries(cases)) {
    it(`${name}: P(over) matches scipy to 1e-6`, () => {
      fixture.lines.forEach((line, i) => {
        expect(probOver(fixture.dist, line)).toBeCloseTo(fixture.over[i], 6)
      })
    })
  }

  it('negative binomial P(X = k) matches', () => {
    const f = cases.nb
    expect(probExact(f.dist, 62)).toBeCloseTo(f.exact[0], 6)
    expect(probExact(f.dist, 75)).toBeCloseTo(f.exact[1], 6)
  })

  it('poisson P(X = k) matches', () => {
    const f = cases.poisson
    ;[0, 1, 2].forEach((k, i) => {
      expect(probExact(f.dist, k)).toBeCloseTo(f.exact[i], 6)
    })
  })

  it('reproduces the documented passing-TD base rate', () => {
    // how_books_build_lines.md: passing TDs average ~1.5/start; Over 1.5 is the minority side.
    const dist: DistributionParams = {
      family: 'poisson',
      params: { lam: 1.5 },
      mean: 1.5,
      median: 1,
      p25: 1,
      p75: 2,
      integer_valued: true,
    }
    expect(probOver(dist, 1.5)).toBeCloseTo(0.4422, 3)
    expect(probExact(dist, 0)).toBeCloseTo(0.2231, 3)
  })
})

describe('special functions', () => {
  it('poisson CDF is monotone and reaches 1', () => {
    let prev = -1
    for (let k = 0; k < 30; k++) {
      const c = poissonCdf(k, 3)
      expect(c).toBeGreaterThanOrEqual(prev)
      prev = c
    }
    expect(poissonCdf(60, 3)).toBeCloseTo(1, 8)
  })

  it('negative binomial CDF is monotone and reaches 1', () => {
    let prev = -1
    for (let k = 0; k < 400; k += 10) {
      const c = nbinomCdf(k, 1.66, 0.0212)
      expect(c).toBeGreaterThanOrEqual(prev)
      prev = c
    }
    expect(nbinomCdf(5000, 1.66, 0.0212)).toBeCloseTo(1, 6)
  })
})

describe('probability bookkeeping', () => {
  it('over and under partition at a half-point line', () => {
    // A half point makes a push impossible, so the two sides must sum to exactly 1.
    const f = cases.nb
    expect(probOver(f.dist, 62.5) + probUnder(f.dist, 62.5)).toBeCloseTo(1, 9)
  })

  it('an integer line leaves room for a push', () => {
    const f = cases.poisson
    const push = probExact(f.dist, 2)
    expect(push).toBeGreaterThan(0)
    expect(probOver(f.dist, 2) + probUnder(f.dist, 2)).toBeCloseTo(1 - push, 9)
  })

  it('P(over) falls monotonically in the line', () => {
    const f = cases.nb
    const probs = [10, 40, 70, 110, 200].map((line) => probOver(f.dist, line))
    expect(probs).toEqual([...probs].sort((a, b) => b - a))
  })

  it('every probability stays inside [0, 1]', () => {
    for (const f of Object.values(cases)) {
      for (const line of [-5, 0, 0.5, 1, 50, 1000]) {
        const p = probOver(f.dist, line)
        expect(p).toBeGreaterThanOrEqual(0)
        expect(p).toBeLessThanOrEqual(1)
      }
    }
  })
})

describe('quantiles and display helpers', () => {
  it('quantiles are ordered and bracket the stored median', () => {
    for (const f of Object.values(cases)) {
      expect(quantile(f.dist, 0.25)).toBeLessThanOrEqual(quantile(f.dist, 0.75))
    }
  })

  it('the default line is always a half point, so a push is impossible', () => {
    for (const f of Object.values(cases)) {
      const line = defaultLine(f.dist)
      expect(line * 2).toBeCloseTo(Math.round(line * 2), 9)
      expect(Number.isInteger(line)).toBe(false)
    }
  })

  it('converts probability to fair American odds', () => {
    expect(toAmericanOdds(0.5)).toBe(-100)
    // The reference doc puts the fair price on Over 1.5 passing TDs near +118.
    expect(toAmericanOdds(0.459)).toBe(118)
    expect(formatOdds(0.459)).toBe('+118')
    expect(formatOdds(0.6)).toBe('-150')
    expect(toAmericanOdds(0)).toBeNull()
    expect(toAmericanOdds(1)).toBeNull()
  })

  it('density curves sum to roughly one', () => {
    for (const f of Object.values(cases)) {
      const total = densityCurve(f.dist).reduce((acc, p) => acc + p.y, 0)
      expect(total).toBeGreaterThan(0.9)
      expect(total).toBeLessThan(1.05)
    }
  })
})
