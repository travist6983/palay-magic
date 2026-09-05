/**
 * The timezone cases exist because the app shipped a four-hour error in the header.
 *
 * The API serialises naive UTC with no zone marker; `new Date()` reads that as local time. In
 * Detroit a four-hour-old refresh rendered as "23m ago" beside the backend's own "age: 4.1h", and
 * east of UTC the age goes negative and pins at "just now" forever — which would leave the §8
 * freshness badges permanently unable to report staleness.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  localStamp,
  multiplierColour,
  multiplierLabel,
  pct,
  relativeTime,
  signed,
  stat,
  utcIso,
} from './format'

const NOW = new Date('2026-09-05T20:00:00Z')

describe('utcIso', () => {
  it('stamps a zone-less backend timestamp as UTC', () => {
    expect(utcIso('2026-09-05 16:30:40.543348')).toBe('2026-09-05T16:30:40.543348Z')
  })

  it('leaves a value that already carries a zone alone', () => {
    expect(utcIso('2026-09-05T16:30:40Z')).toBe('2026-09-05T16:30:40Z')
    expect(utcIso('2026-09-05T16:30:40+02:00')).toBe('2026-09-05T16:30:40+02:00')
    expect(utcIso('2026-09-05T16:30:40-0400')).toBe('2026-09-05T16:30:40-0400')
  })

  it('is idempotent, so wrapping twice is harmless', () => {
    const once = utcIso('2026-09-05 16:30:40')!
    expect(utcIso(once)).toBe(once)
  })

  it('returns null for nothing', () => {
    expect(utcIso(null)).toBeNull()
    expect(utcIso(undefined)).toBeNull()
    expect(utcIso('')).toBeNull()
  })
})

describe('relativeTime', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(NOW)
  })
  afterEach(() => vi.useRealTimers())

  it('reads a naive backend stamp as UTC, not as local time', () => {
    // The regression: this is 4h before NOW in UTC. Parsed as local it would be wrong by the
    // viewer's offset, and in Detroit it read as "23m ago".
    expect(relativeTime('2026-09-05 16:00:00.000000')).toBe('4h ago')
  })

  it('agrees with an explicit-zone stamp for the same instant', () => {
    expect(relativeTime('2026-09-05 16:00:00')).toBe(relativeTime('2026-09-05T16:00:00Z'))
  })

  it('reports staleness rather than pinning at "just now"', () => {
    // The failure mode east of UTC: a positive offset made every age negative.
    expect(relativeTime('2026-09-05 19:30:00')).toBe('30m ago')
    expect(relativeTime('2026-09-04 20:00:00')).toBe('24h ago')
    expect(relativeTime('2026-09-01 20:00:00')).toBe('4d ago')
  })

  it('names a future timestamp instead of calling it fresh', () => {
    expect(relativeTime('2026-09-05 22:00:00')).toBe('in the future')
  })

  it('handles the boundary cases', () => {
    expect(relativeTime(null)).toBe('never')
    expect(relativeTime(undefined)).toBe('never')
    expect(relativeTime('not a date')).toBe('not a date')
    expect(relativeTime('2026-09-05 20:00:00')).toBe('just now')
  })
})

describe('localStamp', () => {
  it('renders never for a missing value', () => {
    expect(localStamp(null)).toBe('never')
  })

  it('returns the input unchanged when it cannot be parsed', () => {
    expect(localStamp('garbage')).toBe('garbage')
  })

  it('renders the same instant regardless of how the zone was written', () => {
    expect(localStamp('2026-09-05 16:30:40')).toBe(localStamp('2026-09-05T16:30:40Z'))
  })
})

describe('numeric formatting', () => {
  it('never renders NaN, null or undefined as text', () => {
    for (const bad of [null, undefined, NaN, Infinity]) {
      expect(stat(bad as number)).toBe('—')
      expect(pct(bad as number)).toBe('—')
      expect(signed(bad as number)).toBe('—')
    }
  })

  it('scales precision so columns stay scannable', () => {
    expect(stat(1.234)).toBe('1.23')
    expect(stat(12.34)).toBe('12.3')
    expect(stat(123.4)).toBe('123')
  })

  it('formats percentages and signed deltas', () => {
    expect(pct(0.713)).toBe('71%')
    expect(pct(0.713, 1)).toBe('71.3%')
    expect(signed(2.5)).toBe('+2.5')
    expect(signed(-2.5)).toBe('-2.5')
  })
})

describe('matchup display', () => {
  it('renders a multiplier as a plain-English delta', () => {
    expect(multiplierLabel(1.07)).toBe('+7%')
    expect(multiplierLabel(0.93)).toBe('-7%')
    expect(multiplierLabel(1.001)).toBe('neutral')
  })

  it('colours by whether the matchup helps the player whose prop this is', () => {
    // An "allowed" metric: more allowed is a softer matchup, so green.
    expect(multiplierColour(1.1, true)).toContain('good')
    expect(multiplierColour(0.9, true)).toContain('bad')
    // A "generated" metric such as INTs forced inverts.
    expect(multiplierColour(1.1, false)).toContain('bad')
    expect(multiplierColour(0.9, false)).toContain('good')
  })
})
