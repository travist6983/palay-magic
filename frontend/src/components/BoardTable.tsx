/**
 * The position board (section 8): ten ranked players, their matchup, their availability, and the
 * two or three headline props for the position.
 *
 * Every number a bettor compares down a column is tabular (.num). Probabilities on a Bernoulli
 * stat carry fair American odds beside them, because that is the number that gets compared to a
 * book, not a percentage.
 */

import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { formatOdds } from '../lib/distributions'
import { pct, signed, stat as fmtStat } from '../lib/format'
import type { BoardRow, Projection } from '../lib/types'
import { InjuryChip } from './InjuryChip'
import { MatchupCell } from './MatchupCell'

export interface BoardTableProps {
  rows: BoardRow[]
  className?: string
}

/** A pass-rate move of more than three points is a different offence, not noise (D3). */
const PASS_RATE_FLAG_THRESHOLD = 0.03

const RIGHT_ALIGNED = new Set(['rank'])

/** A board is a ranked list, so it always opens in rank order. Module-level so the reset is a no-op
 *  when the sorting state already holds it. */
const DEFAULT_SORTING: SortingState = [{ id: 'rank', desc: false }]

function alignClass(columnId: string): string {
  return RIGHT_ALIGNED.has(columnId) || columnId.startsWith('h:') ? 'text-right' : 'text-left'
}

function initials(name: string): string {
  const parts = name.split(/\s+/).filter(Boolean)
  if (!parts.length) return '??'
  return ((parts[0][0] ?? '') + (parts.length > 1 ? (parts[parts.length - 1][0] ?? '') : '')).toUpperCase()
}

// --- flags -----------------------------------------------------------------

interface Flag {
  key: string
  label: string
  tone: string
  title: string
}

/**
 * Week 1 2026 projections are built entirely on 2025 logs (D3), so anything that changed since
 * those logs were recorded is the most load-bearing thing on the row.
 */
export function rowFlags(row: BoardRow): Flag[] {
  const flags: Flag[] = []
  if (row.changed_team) {
    flags.push({
      key: 'team',
      label: 'new team',
      tone: 'bg-accent/15 text-accent',
      title:
        'Changed team since the game logs this projection is built on. Volume, scheme and target ' +
        'competition are all new; treat the usage baseline as the weakest part of the number.',
    })
  }
  if (row.changed_coach) {
    flags.push({
      key: 'coach',
      label: 'new coach',
      tone: 'bg-warn/15 text-warn',
      title:
        'New head coach or coordinator since the game logs this projection is built on. Pace, ' +
        'pass rate and red-zone usage are the parts most likely to move.',
    })
  }
  if (Math.abs(row.pass_rate_shift) > PASS_RATE_FLAG_THRESHOLD) {
    flags.push({
      key: 'pass',
      label: `pass ${signed(row.pass_rate_shift * 100, 0)}%`,
      tone: row.pass_rate_shift > 0 ? 'bg-accent/15 text-accent' : 'bg-warn/15 text-warn',
      title:
        `Team pass rate has moved ${signed(row.pass_rate_shift * 100, 0)} points against the season ` +
        'the projection is built on. Volume props for pass catchers and runners move in opposite ' +
        'directions on this.',
    })
  }
  if (row.insufficient_history) {
    flags.push({
      key: 'history',
      label: 'thin history',
      tone: 'bg-bad/15 text-bad',
      title:
        'Too few qualifying games to fit this player directly; the projection leans hard on the ' +
        'positional prior. The interval is wide for a reason.',
    })
  }
  return flags
}

// --- headline projection cell ----------------------------------------------

function projectionFor(row: BoardRow, statKey: string): Projection | undefined {
  return row.headline.find((p) => p.stat === statKey)
}

/** Sort value: the win probability for a Bernoulli market, the median for everything else. */
function projectionSortValue(row: BoardRow, statKey: string): number {
  const projection = projectionFor(row, statKey)
  if (!projection) return Number.NEGATIVE_INFINITY
  const dist = projection.distribution
  return dist.family === 'bernoulli' ? dist.mean : dist.median
}

export interface ProjectionCellProps {
  projection: Projection | undefined
}

export function ProjectionCell({ projection }: ProjectionCellProps) {
  if (!projection) return <span className="text-chalk-faint">—</span>
  const dist = projection.distribution

  if (dist.family === 'bernoulli') {
    const p = dist.mean
    const title =
      `${projection.label}: ${pct(p, 1)} to hit, fair price ${formatOdds(p)} before vig. ` +
      (projection.settlement_note ? `${projection.settlement_note} ` : '') +
      'Compare the fair price to the book: anything shorter is the hold.'
    return (
      <div className="leading-tight" title={title}>
        <div className="num text-sm text-chalk">{pct(p, 0)}</div>
        <div className="num text-[11px] text-chalk-faint">{formatOdds(p)} fair</div>
      </div>
    )
  }

  const degenerate = dist.p25 === dist.p75
  const spread = degenerate
    ? `mean ${fmtStat(dist.mean)}`
    : `${fmtStat(dist.p25)}–${fmtStat(dist.p75)}`
  const title =
    `${projection.label}: median ${fmtStat(dist.median)}, p25–p75 ${fmtStat(dist.p25)}–${fmtStat(dist.p75)}, ` +
    `mean ${fmtStat(dist.mean)} (${dist.family.replace(/_/g, ' ')}).` +
    (projection.high_variance
      ? ' High variance: the median is a poor guide here, price off the tail.'
      : '') +
    (projection.settlement_note ? ` ${projection.settlement_note}` : '')

  return (
    <div className="leading-tight" title={title}>
      <div className="num text-sm text-chalk">
        {fmtStat(dist.median)}
        {projection.high_variance ? (
          <span className="ml-1 align-top text-[9px] uppercase tracking-wide text-warn/80">hv</span>
        ) : null}
      </div>
      <div className="num text-[11px] text-chalk-faint">{spread}</div>
    </div>
  )
}

// --- table -----------------------------------------------------------------

export function BoardTable({ rows, className }: BoardTableProps) {
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING)

  /** Headline stats vary by position and arrive in the payload, so the columns are derived. */
  const headlineStats = useMemo(() => {
    const seen = new Map<string, string>()
    for (const row of rows) {
      for (const projection of row.headline) {
        if (!seen.has(projection.stat)) seen.set(projection.stat, projection.label)
      }
    }
    return [...seen.entries()].map(([key, label]) => ({ key, label }))
  }, [rows])

  const primaryLabel = headlineStats[0]?.label ?? null

  /**
   * The projection columns are named after the payload's stats, so a sort on one of them does not
   * survive a move to another position. TanStack drops a sort on a column that no longer exists
   * without telling anyone, which leaves the table with NO sorted column -- every header reading
   * aria-sort="none" and no marker -- even though the rows are still in rank order. Put the board
   * back on its default whenever the column set changes.
   */
  const statSignature = headlineStats.map((s) => s.key).join(',')
  useEffect(() => {
    setSorting(DEFAULT_SORTING)
  }, [statSignature])

  const columns = useMemo<ColumnDef<BoardRow>[]>(() => {
    const base: ColumnDef<BoardRow>[] = [
      {
        id: 'rank',
        header: '#',
        accessorFn: (row) => row.rank,
        sortingFn: 'basic',
        cell: ({ row }) => <span className="num text-chalk-dim">{row.original.rank}</span>,
      },
      {
        id: 'player',
        header: 'Player',
        accessorFn: (row) => row.display_name,
        sortingFn: 'text',
        cell: ({ row }) => {
          const player = row.original
          const flags = rowFlags(player)
          return (
            <div className="flex items-start gap-2">
              <span className="relative mt-0.5 inline-flex h-7 w-7 shrink-0 items-center justify-center overflow-hidden rounded bg-ink-line text-[10px] font-medium text-chalk-faint">
                <span>{initials(player.display_name)}</span>
                {player.headshot_url ? (
                  <img
                    src={player.headshot_url}
                    alt=""
                    loading="lazy"
                    className="absolute inset-0 h-full w-full object-cover"
                    onError={(event) => {
                      event.currentTarget.style.visibility = 'hidden'
                    }}
                  />
                ) : null}
              </span>
              <span className="min-w-0">
                <Link
                  to={`/player/${player.gsis_id}`}
                  className="block truncate text-sm text-chalk hover:text-accent hover:underline"
                >
                  {player.display_name}
                </Link>
                {flags.length ? (
                  <span className="mt-0.5 flex flex-wrap gap-1">
                    {flags.map((flag) => (
                      <span key={flag.key} className={`chip ${flag.tone}`} title={flag.title}>
                        {flag.label}
                      </span>
                    ))}
                  </span>
                ) : null}
              </span>
            </div>
          )
        },
      },
      {
        id: 'team',
        header: 'Tm',
        accessorFn: (row) => row.team ?? '',
        sortingFn: 'text',
        cell: ({ row }) => <span className="text-chalk-dim">{row.original.team ?? '—'}</span>,
      },
      {
        id: 'opponent',
        header: 'Opp',
        accessorFn: (row) => row.opponent ?? '',
        sortingFn: 'text',
        cell: ({ row }) => <span className="text-chalk-dim">{row.original.opponent ?? '—'}</span>,
      },
      {
        id: 'matchup',
        header: 'Matchup',
        accessorFn: (row) => row.opponent_multiplier,
        sortingFn: 'basic',
        sortDescFirst: true,
        cell: ({ row }) => (
          <MatchupCell
            multiplier={row.original.opponent_multiplier}
            rank={row.original.opponent_rank}
            opponent={row.original.opponent}
            statLabel={row.original.opponent_metric_label ?? primaryLabel}
          />
        ),
      },
      {
        id: 'injury',
        header: 'Inj',
        accessorFn: (row) => row.play_probability,
        sortingFn: 'basic',
        cell: ({ row }) => (
          <InjuryChip
            status={row.original.injury_status}
            playProbability={row.original.play_probability}
          />
        ),
      },
    ]

    const projections: ColumnDef<BoardRow>[] = headlineStats.map(({ key, label }) => ({
      id: `h:${key}`,
      header: label,
      accessorFn: (row) => projectionSortValue(row, key),
      sortingFn: 'basic',
      sortDescFirst: true,
      cell: ({ row }) => <ProjectionCell projection={projectionFor(row.original, key)} />,
    }))

    return [...base, ...projections]
  }, [headlineStats, primaryLabel])

  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableSortingRemoval: false,
  })

  if (!rows.length) {
    return (
      <div className={`card px-3 py-4 text-sm text-chalk-dim ${className ?? ''}`}>
        no rows on this board
      </div>
    )
  }

  return (
    <div className={`card overflow-x-auto ${className ?? ''}`}>
      <table className="w-full min-w-[720px] border-collapse">
        <thead>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id} className="border-b border-ink-line">
              {headerGroup.headers.map((header) => {
                const sorted = header.column.getIsSorted()
                return (
                  <th
                    key={header.id}
                    scope="col"
                    className={`th ${alignClass(header.column.id)}`}
                    aria-sort={sorted === 'asc' ? 'ascending' : sorted === 'desc' ? 'descending' : 'none'}
                  >
                    <button
                      type="button"
                      onClick={header.column.getToggleSortingHandler()}
                      className={`inline-flex items-center gap-1 uppercase tracking-wider hover:text-chalk ${
                        sorted ? 'text-chalk' : ''
                      }`}
                      title="Sort by this column"
                    >
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      <span className="text-[8px] text-chalk-faint">
                        {sorted === 'asc' ? '▲' : sorted === 'desc' ? '▼' : ''}
                      </span>
                    </button>
                  </th>
                )
              })}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className="border-b border-ink-line/60 last:border-0 hover:bg-ink-line/25">
              {row.getVisibleCells().map((cell) => (
                <td
                  key={cell.id}
                  className={`td align-top ${alignClass(cell.column.id)} ${
                    cell.column.id.startsWith('h:') ? 'tabular-nums' : ''
                  }`}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default BoardTable
