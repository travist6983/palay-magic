/**
 * Position board page. Reads :position from the route, fetches the precomputed board, and hands it
 * to BoardTable. Never blanks the screen: a skeleton while it loads, a short message on failure,
 * and the last good data plus a warning strip if a refetch fails (section 10).
 */

import { useQuery } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { BoardTable } from '../components/BoardTable'
import { ApiError, api } from '../lib/api'
import { weekLabel } from '../lib/format'
import type { Board } from '../lib/types'

export interface BoardPageProps {
  /** Overrides the route param, for embedding a board outside its own route. */
  position?: string
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.status === 404
      ? 'no board for this position'
      : `board request failed (${error.status})${error.message ? `: ${error.message.slice(0, 160)}` : ''}`
  }
  return error instanceof Error && error.message
    ? error.message.slice(0, 160)
    : 'board request failed'
}

export function BoardSkeleton({ rows = 10 }: { rows?: number }) {
  return (
    <div className="card animate-pulse overflow-hidden">
      <div className="h-8 border-b border-ink-line bg-ink-line/30" />
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="flex items-center gap-3 border-b border-ink-line/60 px-3 py-2.5 last:border-0">
          <div className="h-3 w-4 rounded bg-ink-line" />
          <div className="h-7 w-7 shrink-0 rounded bg-ink-line" />
          <div className="h-3 w-40 rounded bg-ink-line" />
          <div className="h-3 w-10 rounded bg-ink-line" />
          <div className="h-3 w-14 rounded bg-ink-line" />
          <div className="ml-auto h-3 w-16 rounded bg-ink-line" />
          <div className="h-3 w-16 rounded bg-ink-line" />
          <div className="h-3 w-16 rounded bg-ink-line" />
        </div>
      ))}
    </div>
  )
}

export function BoardPage({ position: positionProp }: BoardPageProps) {
  const params = useParams<{ position?: string }>()
  const position = (positionProp ?? params.position ?? '').toUpperCase()

  const query = useQuery<Board>({
    queryKey: ['board', position],
    queryFn: () => api.board(position),
    enabled: position.length > 0,
  })

  const board = query.data

  const header = (
    <div className="flex flex-wrap items-baseline justify-between gap-2">
      <h1 className="text-sm font-semibold uppercase tracking-wider text-chalk">
        {position || 'Board'}
        {board && board.rows.length ? (
          <span className="ml-2 font-normal normal-case tracking-normal text-chalk-faint">
            top {board.rows.length} by projected value
          </span>
        ) : null}
      </h1>
      {board ? (
        <span className="text-xs text-chalk-dim">{weekLabel(board.season, board.week)}</span>
      ) : null}
    </div>
  )

  if (!position) {
    return (
      <section className="space-y-3">
        {header}
        <div className="card px-3 py-4 text-sm text-chalk-dim">
          no position in the URL — pick one from the nav.
        </div>
      </section>
    )
  }

  if (query.isPending) {
    return (
      <section className="space-y-3">
        {header}
        <BoardSkeleton />
      </section>
    )
  }

  if (query.isError && !board) {
    return (
      <section className="space-y-3">
        {header}
        <div className="card space-y-2 px-3 py-4">
          <div className="text-sm text-bad">{errorMessage(query.error)}</div>
          <div className="text-xs text-chalk-dim">
            The API serves precomputed rankings; if it is empty or down, run{' '}
            <code className="rounded bg-ink px-1 text-chalk">make refresh</code> in the repo root.
          </div>
          <button
            type="button"
            onClick={() => void query.refetch()}
            className="chip border border-ink-line bg-ink text-chalk-dim hover:text-chalk"
          >
            retry
          </button>
        </div>
      </section>
    )
  }

  if (!board || !board.rows.length) {
    return (
      <section className="space-y-3">
        {header}
        <div className="card px-3 py-4 text-sm text-chalk-dim">
          no board yet — run <code className="rounded bg-ink px-1 text-chalk">make refresh</code>
        </div>
      </section>
    )
  }

  return (
    <section className="space-y-3">
      {header}
      {query.isError ? (
        <div className="card border-warn/40 px-3 py-2 text-xs text-warn">
          showing the last board that loaded — the latest request failed ({errorMessage(query.error)})
        </div>
      ) : null}
      <BoardTable rows={board.rows} />
      <p className="text-[11px] leading-relaxed text-chalk-faint">
        Rank is projected value for the week. Every 2026 projection is built on prior-season game
        logs, so the chips on a player — new team, new coach, a moved pass rate, thin history — mark
        exactly where that baseline no longer describes the player. Matchup is the defensive
        adjustment already applied to the first projection in the row: 1.00 is a league-average
        defence, and rank 1 of 32 is the softest. Intervals are p25–p75.
      </p>
    </section>
  )
}

export default BoardPage
