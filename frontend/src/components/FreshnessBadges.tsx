/**
 * One dot per ingest source, coloured by freshness. The tooltip carries the detail an analyst
 * needs before trusting a number: when it last succeeded, how stale it is, how many rows landed,
 * and whatever the loader had to say about it.
 */

import { freshnessColour, relativeTime } from '../lib/format'
import type { SourceStatus } from '../lib/types'

export interface FreshnessBadgesProps {
  sources: SourceStatus[]
  className?: string
}

function tooltip(s: SourceStatus): string {
  const lines: string[] = [
    `${s.source} — ${s.status}`,
    s.last_success_at
      ? `last success: ${s.last_success_at} (${relativeTime(s.last_success_at)})`
      : 'last success: never',
    s.age_hours === null ? 'age: unknown' : `age: ${s.age_hours.toFixed(1)}h`,
    s.rows === null ? 'rows: n/a' : `rows: ${s.rows.toLocaleString()}`,
  ]
  if (s.last_attempt_at && s.last_attempt_at !== s.last_success_at) {
    lines.push(`last attempt: ${s.last_attempt_at}`)
  }
  if (s.detail) lines.push(`detail: ${s.detail}`)
  return lines.join('\n')
}

export function FreshnessBadges({ sources, className = '' }: FreshnessBadgesProps) {
  if (sources.length === 0) return null
  const degraded = sources.filter((s) => s.status !== 'green').length
  return (
    <div
      className={`flex flex-wrap items-center gap-x-3 gap-y-1 ${className}`}
      title={degraded === 0 ? 'all sources fresh' : `${degraded} source(s) stale or degraded`}
    >
      {sources.map((s) => (
        <span
          key={s.source}
          title={tooltip(s)}
          className="inline-flex cursor-help items-center gap-1.5 text-[11px] leading-none text-chalk-dim hover:text-chalk"
        >
          <span
            aria-hidden
            className={`h-1.5 w-1.5 shrink-0 rounded-full ${freshnessColour(s.status)}`}
          />
          <span className="whitespace-nowrap">{s.source}</span>
          <span className="sr-only">{s.status}</span>
        </span>
      ))}
    </div>
  )
}

export default FreshnessBadges
