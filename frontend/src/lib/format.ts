/** Display helpers shared across the UI. */

/** A projection number: one decimal below 10, whole numbers above, so tables stay scannable. */
export function stat(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  if (Math.abs(value) >= 100) return value.toFixed(0)
  if (Math.abs(value) >= 10) return value.toFixed(1)
  return value.toFixed(2)
}

export function pct(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

export function signed(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`
}

/** A defensive multiplier as a plain-English delta: 1.07 → "+7% allowed". */
export function multiplierLabel(multiplier: number): string {
  const delta = (multiplier - 1) * 100
  if (Math.abs(delta) < 0.5) return 'neutral'
  return `${delta > 0 ? '+' : ''}${delta.toFixed(0)}%`
}

/**
 * Colour for a matchup cell. Green means a friendlier matchup for the player whose prop this is,
 * which for an "allowed" metric means the defence gives up MORE than average.
 */
export function multiplierColour(multiplier: number, higherIsSofter = true): string {
  const delta = higherIsSofter ? multiplier - 1 : 1 - multiplier
  if (delta > 0.06) return 'text-good'
  if (delta > 0.02) return 'text-good/70'
  if (delta < -0.06) return 'text-bad'
  if (delta < -0.02) return 'text-bad/70'
  return 'text-chalk-dim'
}

export function freshnessColour(status: string): string {
  return status === 'green' ? 'bg-good' : status === 'yellow' ? 'bg-warn' : 'bg-bad'
}

/**
 * Normalise a backend timestamp to an unambiguous ISO instant.
 *
 * The API serialises naive UTC — `"2026-09-05 16:30:40.543348"`, straight out of the backend's
 * `utcnow()` — with no zone marker. `new Date()` reads a zone-less string as **local** time, so
 * every age silently shifted by the viewer's UTC offset: a four-hour-old refresh read as
 * "23m ago" in Detroit. East of UTC it is worse than wrong, because the offset makes the age
 * negative and it pins at "just now" no matter how stale the data gets — which would leave the
 * freshness badges of §8 structurally incapable of reporting staleness, the one job they have.
 *
 * A string that already carries a zone is returned untouched, so this stays correct if the API
 * ever starts emitting offsets.
 */
export function utcIso(stamp: string | null | undefined): string | null {
  if (!stamp) return null
  const trimmed = stamp.trim()
  if (/(?:Z|[+-]\d{2}:?\d{2})$/i.test(trimmed)) return trimmed
  return `${trimmed.replace(' ', 'T')}Z`
}

/** That instant in the viewer's own timezone — the only clock they can compare against. */
export function localStamp(stamp: string | null | undefined): string {
  const iso = utcIso(stamp)
  if (!iso) return 'never'
  const at = new Date(iso)
  if (Number.isNaN(at.getTime())) return String(stamp)
  return at.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

/** How long ago, in words. Accepts the API's naive-UTC stamps as well as full ISO instants. */
export function relativeTime(stamp: string | null | undefined): string {
  const iso = utcIso(stamp)
  if (!iso) return 'never'
  const then = new Date(iso)
  if (Number.isNaN(then.getTime())) return String(stamp)

  const mins = Math.round((Date.now() - then.getTime()) / 60000)
  // Clock skew can put a stamp slightly in the future. Say so rather than reporting it as fresh.
  if (mins < -2) return 'in the future'
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.round(mins / 60)
  if (hours < 48) return `${hours}h ago`
  return `${Math.round(hours / 24)}d ago`
}

export function weekLabel(season: number, week: number): string {
  return `${season} · Week ${week}`
}
