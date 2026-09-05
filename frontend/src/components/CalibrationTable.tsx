/**
 * The §6 backtest calibration table: per (position, stat) accuracy of the last walk-forward run.
 *
 * The one thing a reader has to understand before trusting a row is the degenerate flag. For a
 * low-count stat (LB sacks, anytime TD) the projected p25 and p75 collapse onto the same integer,
 * so "actual landed inside the interval" catches every zero game and coverage reads 85%+ however
 * good the model is. Those rows are marked, their coverage is greyed out and excluded from the
 * headline number, and they are judged on PIT central mass and Brier instead (migration 011).
 */

import { useQuery } from '@tanstack/react-query'
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from '@tanstack/react-table'
import { useMemo, useState, type ReactNode } from 'react'
import { ApiError, api } from '../lib/api'
import { pct, stat as fmtStat, signed } from '../lib/format'
import type { CalibrationRow } from '../lib/types'

export interface CalibrationTableProps {
  className?: string
}

/** Display names for the D9 stat keys. Keys do not collide across positions, so one map is enough. */
const STAT_LABELS: Record<string, string> = {
  pass_attempts: 'Pass attempts',
  completions: 'Completions',
  passing_yards: 'Passing yards',
  passing_tds: 'Passing TDs',
  interceptions: 'Interceptions',
  rush_attempts: 'Rush attempts',
  rushing_yards: 'Rushing yards',
  longest_completion: 'Longest completion',
  anytime_rush_td: 'Anytime rushing TD',
  receptions: 'Receptions',
  receiving_yards: 'Receiving yards',
  rush_rec_yards: 'Rush + rec yards',
  longest_rush: 'Longest rush',
  anytime_td: 'Anytime TD',
  targets: 'Targets',
  longest_reception: 'Longest reception',
  fg_attempts: 'FG attempts',
  fg_made: 'FG made',
  xp_made: 'XP made',
  kicking_points: 'Kicking points',
  longest_fg: 'Longest FG',
  tackles_assists: 'Tackles + assists',
  solo_tackles: 'Solo tackles',
  sacks: 'Sacks',
  passes_defended: 'Passes defended',
}

/** Both interval metrics target 50%. Green inside 40–60, yellow inside 30–70, red beyond. */
export function bandColour(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'text-chalk-faint'
  if (value >= 0.4 && value <= 0.6) return 'text-good'
  if (value >= 0.3 && value <= 0.7) return 'text-warn'
  return 'text-bad'
}

/** A fixed-precision number that degrades to an em dash instead of printing "NaN". */
function fixed(value: number | null | undefined, digits: number): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return value.toFixed(digits)
}

/** A calibrated P(over median) scores 0.25. Much worse means the median is not the median. */
function brierColour(value: number): string {
  if (!Number.isFinite(value)) return 'text-chalk-faint'
  if (value > 0.28) return 'text-bad'
  if (value > 0.26) return 'text-warn'
  return 'text-chalk'
}

/** Bias only matters relative to the error it sits inside: a tilt worth a quarter of MAE is real. */
function biasColour(bias: number, mae: number): string {
  if (!Number.isFinite(bias) || !Number.isFinite(mae) || mae <= 0) return 'text-chalk-dim'
  const share = Math.abs(bias) / mae
  if (share > 0.5) return 'text-bad'
  if (share > 0.25) return 'text-warn'
  return 'text-chalk-dim'
}

const RIGHT_ALIGNED = new Set(['n', 'mae', 'bias', 'coverage', 'pit_central', 'brier'])

const column = createColumnHelper<CalibrationRow>()

const COLUMNS = [
  column.accessor('position', {
    header: 'Pos',
    cell: (c) => <span className="font-medium text-chalk">{c.getValue()}</span>,
  }),
  column.accessor('stat', {
    header: 'Stat',
    cell: (c) => (
      <span className="flex items-center gap-2">
        <span>{STAT_LABELS[c.getValue()] ?? c.getValue()}</span>
        {c.row.original.degenerate ? (
          <span className="chip bg-warn/15 text-warn" title="p25 = p75 — coverage is not usable">
            low-count
          </span>
        ) : null}
      </span>
    ),
  }),
  column.accessor('n', {
    header: 'n',
    cell: (c) => <span className="num text-chalk-dim">{c.getValue()}</span>,
  }),
  column.accessor('mae', {
    header: 'MAE',
    cell: (c) => <span className="num">{fmtStat(c.getValue())}</span>,
  }),
  column.accessor('bias', {
    header: 'Bias',
    cell: (c) => (
      <span className={`num ${biasColour(c.getValue(), c.row.original.mae)}`}>
        {signed(c.getValue(), 2)}
      </span>
    ),
  }),
  column.accessor('coverage', {
    header: 'p25–p75 cov',
    cell: (c) =>
      c.row.original.degenerate ? (
        <span
          className="num text-chalk-faint"
          title="Degenerate interval: p25 = p75, so coverage is meaningless here. Read PIT and Brier."
        >
          {pct(c.getValue(), 1)}
          <sup className="ml-0.5 text-warn">†</sup>
        </span>
      ) : (
        <span className={`num ${bandColour(c.getValue())}`}>{pct(c.getValue(), 1)}</span>
      ),
  }),
  column.accessor('pit_central', {
    header: 'PIT central',
    cell: (c) => <span className={`num ${bandColour(c.getValue())}`}>{pct(c.getValue(), 1)}</span>,
  }),
  column.accessor('brier', {
    header: 'Brier',
    cell: (c) => <span className={`num ${brierColour(c.getValue())}`}>{fixed(c.getValue(), 3)}</span>,
  }),
]

function HeadlineTile({
  label,
  value,
  note,
}: {
  label: string
  value: number | null
  note: string
}) {
  const delta = value === null || !Number.isFinite(value) ? null : (value - 0.5) * 100
  return (
    <div className="rounded border border-ink-line bg-ink px-3 py-2">
      <div className="text-[11px] uppercase tracking-wider text-chalk-faint">{label}</div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className={`num text-2xl font-semibold ${bandColour(value)}`}>{pct(value, 1)}</span>
        <span className="num text-xs text-chalk-faint">
          target 50%{delta === null ? '' : ` · ${signed(delta, 1)} pts`}
        </span>
      </div>
      <div className="mt-1 text-xs leading-snug text-chalk-dim">{note}</div>
    </div>
  )
}

function Notice({
  title,
  children,
  tone = 'chalk',
}: {
  title: string
  children?: ReactNode
  tone?: 'chalk' | 'bad'
}) {
  return (
    <div className="px-4 py-6">
      <div className={`text-sm font-medium ${tone === 'bad' ? 'text-bad' : 'text-chalk'}`}>
        {title}
      </div>
      {children ? <div className="mt-2 text-xs leading-relaxed text-chalk-dim">{children}</div> : null}
    </div>
  )
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

/** The calibration section of the model-health page: headline numbers, caveats, and the table. */
export function CalibrationTable({ className = '' }: CalibrationTableProps) {
  const [sorting, setSorting] = useState<SortingState>([{ id: 'position', desc: false }])
  // networkMode 'always': the API is on 127.0.0.1, so the browser's public-internet online
  // heuristic must not pause the fetch (it would leave the section on a skeleton forever).
  const { data, error, isPending, isPaused, isFetching, refetch } = useQuery({
    queryKey: ['calibration'],
    queryFn: () => api.calibration(),
    networkMode: 'always',
  })

  const rows = useMemo(() => data?.rows ?? [], [data])
  const table = useReactTable({
    data: rows,
    columns: COLUMNS,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  const degenerateCount = rows.filter((r) => r.degenerate).length
  const graded = rows.reduce((sum, r) => sum + r.n, 0)
  const failing = rows.filter((r) => !r.degenerate && (r.coverage < 0.3 || r.coverage > 0.7)).length

  return (
    <section className={`card ${className}`}>
      <header className="border-b border-ink-line px-4 py-3">
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-chalk">
            Calibration — walk-forward backtest
          </h2>
          <div className="num text-xs text-chalk-faint">
            {data ? (
              <>
                {data.season} · weeks {data.week_start}–{data.week_end} · {graded.toLocaleString()}{' '}
                graded projections · {rows.length} cells · run {data.run_id}
              </>
            ) : (
              'no run loaded'
            )}
            {isPaused ? (
              <span className="ml-2 text-warn">paused</span>
            ) : isFetching ? (
              <span className="ml-2 text-chalk-faint">refreshing…</span>
            ) : null}
          </div>
        </div>
        {data ? (
          <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
            <HeadlineTile
              label="p25–p75 coverage"
              value={data.coverage}
              note="Share of actuals inside the projected interval, n-weighted over the continuous cells only. Below 50% means the distributions are too narrow."
            />
            <HeadlineTile
              label="PIT central mass"
              value={data.pit_central}
              note="Share of randomised PIT values in [0.25, 0.75], n-weighted over every cell. The one interval metric that survives a low-count stat."
            />
          </div>
        ) : null}
      </header>

      {error && !data ? (
        <Notice title="Could not load the calibration run" tone="bad">
          <p>
            {error instanceof ApiError
              ? `${error.status} — ${error.message || 'request failed'}`
              : String(error)}
          </p>
          <p className="mt-2">
            If no backtest has been stored yet, produce one with{' '}
            <code className="rounded bg-ink px-1 py-0.5 font-mono text-chalk">make backtest</code>.
          </p>
          <button
            type="button"
            onClick={() => void refetch()}
            className="mt-3 rounded border border-ink-line px-2 py-1 text-xs text-chalk hover:border-accent hover:text-accent"
          >
            Retry
          </button>
        </Notice>
      ) : isPaused && !data ? (
        <Notice title="Paused — the calibration run has not loaded">
          <p>
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
        </Notice>
      ) : isPending ? (
        <SkeletonRows rows={8} />
      ) : !data ? (
        <Notice title="No backtest has been run">
          <p>
            The model-health numbers come from a stored walk-forward replay of a finished season.
            Nothing is stored yet, so there is nothing here to judge the projections against.
          </p>
          <p className="mt-2">
            Run{' '}
            <code className="rounded bg-ink px-1 py-0.5 font-mono text-chalk">make backtest</code>{' '}
            and reload this page.
          </p>
        </Notice>
      ) : (
        <>
          {error || isPaused ? (
            <p className="border-b border-ink-line bg-warn/10 px-4 py-2 text-xs text-warn">
              Showing the last loaded run — the refresh{' '}
              {isPaused
                ? 'is paused until the tab is focused or the connection returns'
                : `failed${error instanceof ApiError ? ` (${error.status})` : ''}`}
              .
            </p>
          ) : null}

          {degenerateCount > 0 ? (
            <p className="border-b border-ink-line px-4 py-3 text-xs leading-relaxed text-chalk-dim">
              <span className="text-warn">†</span> {degenerateCount} of {rows.length} cells are
              marked <span className="text-warn">low-count</span>. For a stat like LB sacks or
              anytime TD the projected p25 and p75 land on the same integer — usually 0 — so
              &ldquo;the actual was inside the interval&rdquo; is true for every scoreless game and
              coverage reads 85%+ no matter how good the model is. Coverage on those rows is greyed
              out and left out of the headline number above; judge them on{' '}
              <span className="text-chalk">PIT central mass</span> and{' '}
              <span className="text-chalk">Brier</span>, which stay meaningful when the distribution
              is mostly a point mass at zero.
            </p>
          ) : null}

          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px]">
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
                  <tr
                    key={row.id}
                    className={`border-b border-ink-line/60 hover:bg-ink-line/30 ${
                      row.original.degenerate ? 'bg-warn/[0.04]' : ''
                    }`}
                  >
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

          <dl className="grid grid-cols-1 gap-x-6 gap-y-1 border-t border-ink-line px-4 py-3 text-xs leading-relaxed text-chalk-dim md:grid-cols-2">
            <div>
              <dt className="inline font-medium text-chalk">MAE </dt>
              <dd className="inline">mean absolute error of the projected median, in the stat&rsquo;s own units.</dd>
            </div>
            <div>
              <dt className="inline font-medium text-chalk">Bias </dt>
              <dd className="inline">
                mean (actual − median). Positive means the model projects low. Coloured once the tilt
                is worth a quarter of MAE.
              </dd>
            </div>
            <div>
              <dt className="inline font-medium text-chalk">p25–p75 coverage </dt>
              <dd className="inline">
                share of actuals inside the interval. 50% is calibrated; green 40–60%, yellow
                30–70%. Under-coverage means the distribution is too narrow, so every P(over) is
                too confident.
              </dd>
            </div>
            <div>
              <dt className="inline font-medium text-chalk">PIT central </dt>
              <dd className="inline">
                share of randomised probability-integral-transform values in [0.25, 0.75]. Uniform
                under a correct model for discrete and continuous families alike. Same 50% target
                and the same colour bands.
              </dd>
            </div>
            <div>
              <dt className="inline font-medium text-chalk">Brier </dt>
              <dd className="inline">
                mean squared error of P(over median) — of P(scores at least one) for anytime-TD
                markets. A calibrated model scores about 0.25; much worse means the median is not
                the median.
              </dd>
            </div>
            <div>
              <dt className="inline font-medium text-chalk">Reading it </dt>
              <dd className="inline">
                {failing === 0
                  ? 'Every continuous cell sits inside 30–70% coverage.'
                  : `${failing} continuous ${failing === 1 ? 'cell sits' : 'cells sit'} outside 30–70% coverage — treat those stats' P(over) as directional only.`}{' '}
                Click any header to sort.
              </dd>
            </div>
          </dl>
        </>
      )}
    </section>
  )
}

export default CalibrationTable
