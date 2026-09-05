/**
 * The game around the player: what the market expects the two teams to do, and the weather that
 * scales it. Every projection multiplies through these numbers (§5.3), so they belong next to the
 * name, not buried.
 */

import { pct, signed, stat } from '../lib/format'
import type { Environment } from '../lib/types'

export interface EnvironmentCardProps {
  environment: Environment
  team: string | null
  opponent: string | null
  className?: string
}

interface Field {
  label: string
  value: string
  title?: string
  tone?: 'chalk' | 'dim'
}

export function EnvironmentCard({ environment, team, opponent, className }: EnvironmentCardProps) {
  const env = environment
  const site = env.is_home === null ? 'vs' : env.is_home ? 'vs' : 'at'
  // A missing source is not a live one: the ingest writes games with no line at all as null (D4),
  // and those must not get the green "live market" dot.
  const missingOdds = env.odds_source === null
  const degraded = !env.odds_source?.includes('odds-api')

  const fields: Field[] = [
    {
      label: 'Implied total',
      value: env.implied_total === null ? '—' : stat(env.implied_total),
      title: 'Points this team is expected to score, from the spread and the game total.',
    },
    {
      label: 'Spread',
      value: env.spread === null ? '—' : signed(env.spread, 1),
      title: 'Negative means this team is favoured.',
    },
    {
      label: 'Game total',
      value: env.total_line === null ? '—' : stat(env.total_line),
    },
    {
      label: 'Expected TDs',
      value: env.expected_team_tds === null ? '—' : stat(env.expected_team_tds),
      title: 'Team touchdowns, which drive every anytime-TD projection.',
    },
    {
      label: 'Plays',
      value: env.expected_plays === null ? '—' : stat(env.expected_plays),
      title: 'Expected offensive plays: pace times possessions.',
    },
    {
      label: 'Pass rate',
      value: env.expected_pass_rate === null ? '—' : pct(env.expected_pass_rate, 1),
    },
    {
      label: 'Pass att',
      value: env.expected_pass_attempts === null ? '—' : stat(env.expected_pass_attempts),
    },
    {
      label: 'Rush att',
      value: env.expected_rush_attempts === null ? '—' : stat(env.expected_rush_attempts),
    },
  ]

  return (
    <section className={`card flex flex-col ${className ?? ''}`}>
      <header className="flex items-baseline justify-between gap-2 border-b border-ink-line px-3 py-2">
        <h2 className="text-sm font-semibold text-chalk">Game environment</h2>
        <span className="num text-[11px] text-chalk-faint">
          {team ?? '—'} {site} {opponent ?? '—'}
        </span>
      </header>

      <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 px-3 py-2 sm:grid-cols-4">
        {fields.map((field) => (
          <div key={field.label} title={field.title}>
            <dt className="text-[10px] uppercase tracking-wide text-chalk-faint">{field.label}</dt>
            <dd className="num text-sm text-chalk">{field.value}</dd>
          </div>
        ))}
      </dl>

      <div className="flex flex-wrap items-center gap-1.5 border-t border-ink-line px-3 py-2 text-[11px]">
        <span className="chip bg-ink-line text-chalk-dim">{env.roof ?? 'roof unknown'}</span>
        {env.indoor ? (
          <span className="chip bg-good/15 text-good" title="No wind adjustment indoors.">
            indoor
          </span>
        ) : (
          <span className="chip bg-ink-line text-chalk-dim">
            wind {env.wind === null ? 'unknown' : `${Math.round(env.wind)} mph`}
          </span>
        )}
        {env.temp !== null ? (
          <span className="chip bg-ink-line text-chalk-dim">{Math.round(env.temp)}°F</span>
        ) : null}
        {env.wind_multiplier !== 1 ? (
          <span
            className={`chip ${env.wind_multiplier < 1 ? 'bg-bad/15 text-bad' : 'bg-good/15 text-good'}`}
            title="Applied to passing and kicking projections. Sustained wind, not cold or rain, is the variable that moves them."
          >
            wind ×{env.wind_multiplier.toFixed(2)}
          </span>
        ) : null}
        <span
          className={`ml-auto flex items-center gap-1 ${degraded ? 'text-warn' : 'text-chalk-faint'}`}
          title={
            missingOdds
              ? 'No line was recorded for this game, so the spread, total and everything derived from them are missing.'
              : degraded
                ? 'Live odds were unavailable, so the spread and total come from the nflverse schedule file (D4). The numbers are real, just not the current market.'
                : 'Spread and total from the live odds feed.'
          }
        >
          <span className={`inline-block h-1.5 w-1.5 rounded-full ${degraded ? 'bg-warn' : 'bg-good'}`} />
          {env.odds_source ?? 'no line'}
        </span>
      </div>
    </section>
  )
}

export default EnvironmentCard
