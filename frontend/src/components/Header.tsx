/**
 * Global header: identity, the current week, how fresh the data behind it is, the position
 * boards, and the caveat that everything on screen is built on last season's games.
 */

import { useQuery } from '@tanstack/react-query'
import { Link, NavLink } from 'react-router-dom'
import { api } from '../lib/api'
import { relativeTime, weekLabel } from '../lib/format'
import type { Meta } from '../lib/types'
import { FreshnessBadges } from './FreshnessBadges'
import { PlayerSearch } from './PlayerSearch'

/** Fallback order if /api/meta is down; the API returns the same six. */
const POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K', 'LB']

const CONTAINER = 'mx-auto w-full max-w-[1680px] px-4'

export interface HeaderProps {
  className?: string
}

function tabClass({ isActive }: { isActive: boolean }): string {
  return [
    'border-b-2 px-3 py-2 text-xs font-semibold uppercase tracking-wider transition-colors',
    isActive
      ? 'border-accent text-chalk'
      : 'border-transparent text-chalk-dim hover:border-ink-line hover:text-chalk',
  ].join(' ')
}

function kickoffLabel(iso: string | null): string | null {
  if (!iso) return null
  const d = new Date(`${iso}T00:00:00`)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export interface PriorSeasonBannerProps {
  meta: Meta
}

/** The single most important caveat right now: zero games played, so every number is last season. */
export function PriorSeasonBanner({ meta }: PriorSeasonBannerProps) {
  const kickoff = kickoffLabel(meta.season_start_date)
  return (
    <div className="border-y border-warn/30 bg-warn/[0.07]">
      <div className={`${CONTAINER} flex flex-wrap items-baseline gap-x-3 gap-y-1 py-2`}>
        <span className="chip border border-warn/40 bg-warn/10 text-warn">Prior-season basis</span>
        <p className="text-xs leading-5 text-chalk-dim">
          <span className="text-chalk">
            No {meta.season} games have been played yet
            {kickoff ? `; Week 1 kicks off ${kickoff}` : ''}.
          </span>{' '}
          Every projection, ranking and probability here is built from last season's game logs.
          Team, coach and pass-rate change flags carry what Week&nbsp;1 signal there is — read them
          before you trust a rank.
        </p>
      </div>
    </div>
  )
}

export function Header({ className = '' }: HeaderProps) {
  // networkMode 'always': the API is on 127.0.0.1, so the browser's public-internet online
  // heuristic must not pause the fetch (it would leave the whole shell without a week or status).
  const { data: meta, isError, fetchStatus } = useQuery({
    queryKey: ['meta'],
    queryFn: api.meta,
    networkMode: 'always',
  })

  const positions = meta && meta.positions.length > 0 ? meta.positions : POSITIONS

  return (
    <header className={`sticky top-0 z-30 border-b border-ink-line bg-ink/95 backdrop-blur ${className}`}>
      <div className={`${CONTAINER} flex flex-wrap items-center gap-x-4 gap-y-2 py-2`}>
        <Link to="/" className="shrink-0 text-sm font-semibold tracking-tight text-chalk">
          Prop<span className="text-accent">Lab</span>
        </Link>

        <div className="flex min-w-0 shrink-0 items-baseline gap-3 text-xs">
          {!meta &&
            (isError ? (
              <span className="text-bad" title="/api/meta did not respond">
                status unavailable
              </span>
            ) : fetchStatus === 'paused' ? (
              <span
                className="text-warn"
                title="React Query paused the request (browser offline or tab unfocused); it retries on its own."
              >
                status paused — will retry
              </span>
            ) : (
              <span className="text-chalk-faint">loading status…</span>
            ))}
          {meta && (
            <>
              <span className="num text-chalk-dim">{weekLabel(meta.season, meta.week)}</span>
              <span
                className="text-chalk-faint"
                title={[
                  `last refresh: ${meta.last_refresh_at ?? 'never'}`,
                  meta.last_refresh_seconds === null
                    ? null
                    : `took ${meta.last_refresh_seconds.toFixed(1)}s`,
                  `week source: ${meta.state_source}`,
                  `games played: ${meta.games_played_this_season}`,
                ]
                  .filter(Boolean)
                  .join('\n')}
              >
                refreshed {relativeTime(meta.last_refresh_at)}
              </span>
              {meta.odds_budget_remaining !== null && (
                <span className="num text-chalk-faint" title="The Odds API requests left this month">
                  odds {meta.odds_budget_remaining}
                </span>
              )}
            </>
          )}
        </div>

        <div className="ml-auto flex flex-1 items-center justify-end gap-4">
          {meta && <FreshnessBadges sources={meta.sources} className="justify-end" />}
          <PlayerSearch className="w-56 shrink-0" />
        </div>
      </div>

      <nav className={`${CONTAINER} flex items-center gap-1 overflow-x-auto`} aria-label="Positions">
        {positions.map((p) => (
          <NavLink key={p} to={`/board/${p}`} className={tabClass}>
            {p}
          </NavLink>
        ))}
        <span className="mx-2 h-4 w-px bg-ink-line" aria-hidden />
        <NavLink to="/model" className={tabClass}>
          Model
        </NavLink>
      </nav>

      {meta?.all_history_prior_season && <PriorSeasonBanner meta={meta} />}

      {meta && meta.failed_stages.length > 0 && (
        <div className="border-b border-bad/30 bg-bad/[0.07]">
          <div className={`${CONTAINER} flex flex-wrap items-baseline gap-x-2 gap-y-1 py-1.5`}>
            <span className="chip border border-bad/40 bg-bad/10 text-bad">Refresh failed</span>
            <span className="text-xs text-chalk-dim">
              {meta.failed_stages.length} stage{meta.failed_stages.length === 1 ? '' : 's'} did not
              complete on the last refresh — data below may be stale:
            </span>
            {meta.failed_stages.map((stage) => (
              <span key={stage} className="chip bg-ink-line text-chalk-dim">
                {stage}
              </span>
            ))}
          </div>
        </div>
      )}
    </header>
  )
}

export default Header
