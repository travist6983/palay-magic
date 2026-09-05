/**
 * Every defence graded on one opponent-adjustment metric, softest first.
 *
 * The metric list per position mirrors `backend/models/stats.py` (each stat names the defensive
 * metric that adjusts it) and `backend/models/adjust.py` (labels and the higher_is_softer flag).
 * The API takes `?metric=`, but does not enumerate them, so the map lives here.
 */

import { keepPreviousData, useQuery } from '@tanstack/react-query'
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from '@tanstack/react-table'
import { useMemo, useState } from 'react'
import { ApiError, api } from '../lib/api'
import { multiplierLabel, pct, signed } from '../lib/format'
import type { DefenseRow } from '../lib/types'

export interface DefenseTableProps {
  /** Position selected on first render. Defaults to WR. */
  initialPosition?: string
  className?: string
}

interface MetricOption {
  key: string
  /** Full label, matching the backend Metric.label. */
  label: string
  /** Short label for the selector chips. */
  short: string
  /** True when a bigger number is a friendlier matchup for the player holding the prop. */
  higherIsSofter: boolean
  /** Which projected stats this metric adjusts, for the "why do I care" line. */
  drives: string
}

export const POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K', 'LB'] as const

const METRICS_BY_POSITION: Record<string, MetricOption[]> = {
  QB: [
    { key: 'pass_volume_allowed', label: 'Pass attempts allowed / game', short: 'Pass att', higherIsSofter: true, drives: 'Pass attempts' },
    { key: 'completion_rate_allowed', label: 'Completion % allowed', short: 'Comp %', higherIsSofter: true, drives: 'Completions' },
    { key: 'pass_yards_allowed', label: 'Passing yards allowed / game', short: 'Pass yds', higherIsSofter: true, drives: 'Passing yards' },
    { key: 'pass_td_rate_allowed', label: 'Passing TDs allowed / game', short: 'Pass TD', higherIsSofter: true, drives: 'Passing TDs' },
    { key: 'int_rate_generated', label: 'INTs forced / attempt', short: 'INTs forced', higherIsSofter: false, drives: 'Interceptions' },
    { key: 'rush_volume_allowed', label: 'Rush attempts allowed / game', short: 'Rush att', higherIsSofter: true, drives: 'QB rush attempts' },
    { key: 'rush_yards_allowed', label: 'Rush yards allowed / game', short: 'Rush yds', higherIsSofter: true, drives: 'QB rushing yards' },
    { key: 'explosive_pass_allowed', label: '20+ yard completions / completion', short: '20+ pass', higherIsSofter: true, drives: 'Longest completion' },
    { key: 'rush_td_rate_allowed', label: 'Rush TDs allowed / game', short: 'Rush TD', higherIsSofter: true, drives: 'Anytime rushing TD' },
  ],
  RB: [
    { key: 'rush_volume_allowed', label: 'Rush attempts allowed / game', short: 'Rush att', higherIsSofter: true, drives: 'Rush attempts' },
    { key: 'rush_yards_allowed_rb', label: 'Yards per carry allowed to RBs', short: 'Yds / carry', higherIsSofter: true, drives: 'Rushing yards, rush + rec yards' },
    { key: 'rec_volume_allowed_rb', label: 'Receptions allowed to RBs / game', short: 'RB rec', higherIsSofter: true, drives: 'Receptions' },
    { key: 'rec_yards_allowed_rb', label: 'Yards per target allowed to RBs', short: 'Yds / tgt', higherIsSofter: true, drives: 'Receiving yards' },
    { key: 'explosive_rush_allowed', label: '10+ yard runs / carry', short: '10+ rush', higherIsSofter: true, drives: 'Longest rush' },
    { key: 'rz_td_rate_allowed', label: 'Red-zone TD rate allowed', short: 'RZ TD', higherIsSofter: true, drives: 'Anytime TD' },
  ],
  WR: [
    { key: 'target_volume_allowed_wr', label: 'Targets to WRs / game', short: 'Targets', higherIsSofter: true, drives: 'Targets' },
    { key: 'rec_volume_allowed_wr', label: 'Receptions allowed to WRs / game', short: 'Receptions', higherIsSofter: true, drives: 'Receptions' },
    { key: 'rec_yards_allowed_wr', label: 'Yards per target allowed to WRs', short: 'Yds / tgt', higherIsSofter: true, drives: 'Receiving yards' },
    { key: 'explosive_pass_allowed', label: '20+ yard completions / completion', short: '20+ pass', higherIsSofter: true, drives: 'Longest reception' },
    { key: 'rz_td_rate_allowed', label: 'Red-zone TD rate allowed', short: 'RZ TD', higherIsSofter: true, drives: 'Anytime TD' },
  ],
  TE: [
    { key: 'target_volume_allowed_te', label: 'Targets to TEs / game', short: 'Targets', higherIsSofter: true, drives: 'Targets' },
    { key: 'rec_volume_allowed_te', label: 'Receptions allowed to TEs / game', short: 'Receptions', higherIsSofter: true, drives: 'Receptions' },
    { key: 'rec_yards_allowed_te', label: 'Yards per target allowed to TEs', short: 'Yds / tgt', higherIsSofter: true, drives: 'Receiving yards' },
    { key: 'rz_td_rate_allowed', label: 'Red-zone TD rate allowed', short: 'RZ TD', higherIsSofter: true, drives: 'Anytime TD' },
  ],
  K: [
    { key: 'fg_attempts_allowed', label: 'FG attempts allowed / game', short: 'FG att', higherIsSofter: true, drives: 'FG attempts, FG made, kicking points, longest FG' },
    { key: 'rz_td_rate_allowed', label: 'Red-zone TD rate allowed', short: 'RZ TD', higherIsSofter: true, drives: 'XP made' },
  ],
  LB: [
    { key: 'opp_plays_allowed', label: 'Offensive plays run / game', short: 'Opp plays', higherIsSofter: true, drives: 'Tackles + assists, solo tackles' },
    { key: 'pressure_allowed', label: 'Sacks allowed / dropback', short: 'Sacks allowed', higherIsSofter: true, drives: 'Sacks' },
    { key: 'pass_volume_allowed', label: 'Pass attempts allowed / game', short: 'Pass att', higherIsSofter: true, drives: 'Passes defended' },
  ],
}

/** Colour scale for the multiplier column: green is a friendlier matchup for the player. */
export function multiplierScale(multiplier: number, higherIsSofter: boolean): string {
  const delta = higherIsSofter ? multiplier - 1 : 1 - multiplier
  if (delta >= 0.08) return 'bg-good/20 text-good'
  if (delta >= 0.04) return 'bg-good/10 text-good'
  if (delta >= 0.015) return 'bg-good/[0.06] text-good/80'
  if (delta <= -0.08) return 'bg-bad/20 text-bad'
  if (delta <= -0.04) return 'bg-bad/10 text-bad'
  if (delta <= -0.015) return 'bg-bad/[0.06] text-bad/80'
  return 'text-chalk-dim'
}

/** A fixed-precision number that degrades to an em dash instead of printing "NaN". */
function fixed(value: number | null | undefined, digits: number): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return value.toFixed(digits)
}

/** Rates run down to 0.02, so the shared stat() formatter is too coarse below 1. */
function rawValue(value: number): string {
  if (!Number.isFinite(value)) return '—'
  const abs = Math.abs(value)
  if (abs < 1) return value.toFixed(3)
  if (abs < 10) return value.toFixed(2)
  if (abs < 100) return value.toFixed(1)
  return value.toFixed(0)
}

/** How much out-of-sample error the adjustment removes. Near zero means the matchup says nothing. */
function reductionColour(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return 'text-chalk-faint'
  if (value >= 5) return 'text-good'
  if (value >= 1.5) return 'text-chalk'
  if (value > 0) return 'text-warn'
  return 'text-bad'
}

const RIGHT_ALIGNED = new Set(['rank', 'multiplier', 'raw_value', 'delta', 'league_avg', 'percentile', 'n_games'])

const column = createColumnHelper<DefenseRow>()

function columns(metric: MetricOption) {
  return [
    column.accessor('rank', {
      header: 'Rk',
      cell: (c) => <span className="num text-chalk-faint">{c.getValue()}</span>,
    }),
    column.accessor('team', {
      header: 'Team',
      cell: (c) => <span className="font-medium text-chalk">{c.getValue()}</span>,
    }),
    column.accessor('multiplier', {
      header: 'Multiplier',
      cell: (c) => (
        <span
          className={`num inline-block rounded px-1.5 py-0.5 ${multiplierScale(c.getValue(), metric.higherIsSofter)}`}
        >
          <span className="inline-block w-11 text-right">{fixed(c.getValue(), 3)}</span>
          <span className="ml-2 inline-block w-12 text-right text-[11px] opacity-80">
            {Number.isFinite(c.getValue()) ? multiplierLabel(c.getValue()) : '—'}
          </span>
        </span>
      ),
    }),
    column.accessor('raw_value', {
      header: 'Raw',
      cell: (c) => <span className="num">{rawValue(c.getValue())}</span>,
    }),
    column.accessor((row) => row.raw_value - row.league_avg, {
      id: 'delta',
      header: 'vs avg',
      cell: (c) => {
        const v = c.getValue()
        const digits = Math.abs(c.row.original.league_avg) < 1 ? 3 : 1
        return <span className="num text-chalk-dim">{signed(v, digits)}</span>
      },
    }),
    column.accessor('league_avg', {
      header: 'Lg avg',
      cell: (c) => <span className="num text-chalk-faint">{rawValue(c.getValue())}</span>,
    }),
    column.accessor('percentile', {
      header: 'Pct-ile',
      cell: (c) => <span className="num text-chalk-dim">{pct(c.getValue(), 0)}</span>,
    }),
    column.accessor('n_games', {
      header: 'G',
      cell: (c) => <span className="num text-chalk-faint">{c.getValue()}</span>,
    }),
  ]
}

function SkeletonRows({ rows }: { rows: number }) {
  return (
    <div className="space-y-1 px-4 py-4" aria-hidden>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="h-6 rounded bg-ink-line/60" />
      ))}
    </div>
  )
}

/** Position + metric selector over the full 32-team opponent-adjustment table. */
export function DefenseTable({ initialPosition = 'WR', className = '' }: DefenseTableProps) {
  const [position, setPosition] = useState(
    METRICS_BY_POSITION[initialPosition] ? initialPosition : 'WR',
  )
  const [metricKey, setMetricKey] = useState<string | null>(null)

  const metrics = METRICS_BY_POSITION[position]
  const metric = metrics.find((m) => m.key === metricKey) ?? metrics[0]

  // Softest first: the highest multiplier, unless a bigger number is the tougher matchup.
  const [sorting, setSorting] = useState<SortingState>([
    { id: 'multiplier', desc: metric.higherIsSofter },
  ])

  // networkMode 'always': the API is on 127.0.0.1, so the browser's public-internet online
  // heuristic must not pause the fetch (it would leave the section on a skeleton forever).
  const { data, error, isPending, isPaused, isFetching, refetch } = useQuery({
    queryKey: ['defense', position, metric.key],
    queryFn: () => api.defense(position, metric.key),
    placeholderData: keepPreviousData,
    networkMode: 'always',
  })

  // keepPreviousData hands back the previously selected metric's table while the new one is in
  // flight. Those rows are on a different scale and a possibly opposite direction, so nothing that
  // names or grades the current metric may read from them, and they are never left on screen once
  // the replacement has stopped loading.
  const matches = data?.metric === metric.key
  const rows = useMemo(() => data?.rows ?? [], [data])
  const cols = useMemo(() => columns(metric), [metric])
  const table = useReactTable({
    data: rows,
    columns: cols,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  function choosePosition(next: string) {
    const first = METRICS_BY_POSITION[next][0]
    setPosition(next)
    setMetricKey(null)
    setSorting([{ id: 'multiplier', desc: first.higherIsSofter }])
  }

  function chooseMetric(next: MetricOption) {
    setMetricKey(next.key)
    setSorting([{ id: 'multiplier', desc: next.higherIsSofter }])
  }

  const reduction = matches ? (data?.mse_reduction_pct ?? null) : null

  return (
    <section className={`card ${className}`}>
      <header className="border-b border-ink-line px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-chalk">
            Opponent adjustment — all 32 defences
          </h2>
          <div className="num text-xs text-chalk-faint">
            {matches && data ? `${data.season} · week ${data.week}` : error ? '—' : 'loading…'}
            {isPaused ? (
              <span className="ml-2 text-warn">paused</span>
            ) : isFetching ? (
              <span className="ml-2">refreshing…</span>
            ) : null}
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-1">
          {POSITIONS.map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => choosePosition(p)}
              aria-pressed={p === position}
              className={`rounded border px-2 py-1 text-xs font-medium ${
                p === position
                  ? 'border-accent bg-accent/10 text-accent'
                  : 'border-ink-line text-chalk-dim hover:border-chalk-faint hover:text-chalk'
              }`}
            >
              {p}
            </button>
          ))}
        </div>

        {metrics.length > 1 ? (
          <div className="mt-2 flex flex-wrap items-center gap-1">
            <span className="mr-1 text-[11px] uppercase tracking-wider text-chalk-faint">
              Metric
            </span>
            {metrics.map((m) => (
              <button
                key={m.key}
                type="button"
                onClick={() => chooseMetric(m)}
                aria-pressed={m.key === metric.key}
                title={m.label}
                className={`rounded border px-2 py-0.5 text-xs ${
                  m.key === metric.key
                    ? 'border-accent bg-accent/10 text-accent'
                    : 'border-ink-line text-chalk-dim hover:border-chalk-faint hover:text-chalk'
                }`}
              >
                {m.short}
              </button>
            ))}
          </div>
        ) : null}

        <div className="mt-3 text-xs leading-relaxed text-chalk-dim">
          <span className="text-chalk">{metric.label}</span> — adjusts{' '}
          {metric.drives.toLowerCase()}.{' '}
          <span className={`num font-medium ${reductionColour(reduction)}`}>
            {fixed(reduction, 1)}{reduction === null ? '' : '%'} MSE reduction
          </span>
          : how much of the out-of-sample error in this metric the adjustment removes versus
          ignoring the matchup entirely, measured walk-forward. Under about 1% the matchup carries
          almost no signal here and the multiplier should not move your line much; a negative number
          means the adjustment hurt.
        </div>
      </header>

      {error && !matches ? (
        <div className="px-4 py-6">
          <div className="text-sm font-medium text-bad">Could not load the defence table</div>
          <p className="mt-2 text-xs text-chalk-dim">
            {error instanceof ApiError
              ? `${error.status} — ${error.message || 'request failed'}`
              : String(error)}
          </p>
          <button
            type="button"
            onClick={() => void refetch()}
            className="mt-3 rounded border border-ink-line px-2 py-1 text-xs text-chalk hover:border-accent hover:text-accent"
          >
            Retry
          </button>
        </div>
      ) : isPaused && !matches ? (
        <div className="px-4 py-6">
          <div className="text-sm font-medium text-chalk">Paused — the table has not loaded</div>
          <p className="mt-2 text-xs text-chalk-dim">
            The request is paused rather than retried, which happens while the browser reports no
            connection or the tab sits in the background. It resumes on its own when the tab is
            focused again.
          </p>
          <button
            type="button"
            onClick={() => void refetch()}
            className="mt-3 rounded border border-ink-line px-2 py-1 text-xs text-chalk hover:border-accent hover:text-accent"
          >
            Try now
          </button>
        </div>
      ) : isPending ? (
        <SkeletonRows rows={10} />
      ) : rows.length === 0 ? (
        <p className="px-4 py-6 text-sm text-chalk-dim">
          No defensive multipliers stored for {position} on this metric yet. They are built during a
          refresh.
        </p>
      ) : (
        <>
          {error || isPaused ? (
            <p className="border-b border-ink-line bg-warn/10 px-4 py-2 text-xs text-warn">
              Showing the last loaded table — the refresh{' '}
              {isPaused
                ? 'is paused until the tab is focused or the connection returns'
                : `failed${error instanceof ApiError ? ` (${error.status})` : ''}`}
              .
            </p>
          ) : null}
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px]">
              <thead>
                {table.getHeaderGroups().map((group) => (
                  <tr key={group.id} className="border-b border-ink-line">
                    {group.headers.map((header) => {
                      const sorted = header.column.getIsSorted()
                      return (
                        <th
                          key={header.id}
                          onClick={header.column.getToggleSortingHandler()}
                          className={`th cursor-pointer select-none hover:text-chalk ${
                            RIGHT_ALIGNED.has(header.column.id) ? 'text-right' : ''
                          }`}
                        >
                          {flexRender(header.column.columnDef.header, header.getContext())}
                          <span className="ml-1 text-chalk-faint">
                            {sorted === 'asc' ? '↑' : sorted === 'desc' ? '↓' : ''}
                          </span>
                        </th>
                      )
                    })}
                  </tr>
                ))}
              </thead>
              <tbody>
                {table.getRowModel().rows.map((row) => (
                  <tr key={row.id} className="border-b border-ink-line/60 hover:bg-ink-line/30">
                    {row.getVisibleCells().map((cell) => (
                      <td
                        key={cell.id}
                        className={`td ${RIGHT_ALIGNED.has(cell.column.id) ? 'text-right' : ''}`}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="border-t border-ink-line px-4 py-3 text-xs leading-relaxed text-chalk-dim">
            Sorted softest first — the defence at the top is the friendliest matchup for a{' '}
            {position} on this metric. The multiplier is what a projection is scaled by, shrunk
            toward 1.0 on small samples, so 1.08 means &ldquo;about 8% more than league average
            happens here&rdquo;. <span className="text-chalk">Rk</span> is the backend&rsquo;s rank
            by multiplier, where 1 allows the most
            {metric.higherIsSofter ? '' : ' — on this metric more is tougher, so rank 32 is the softest draw'}
            . <span className="text-chalk">vs avg</span> is the same gap in the metric&rsquo;s own
            units. Click any header to re-sort.
          </p>
        </>
      )}
    </section>
  )
}

export default DefenseTable
