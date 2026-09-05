/**
 * The last six games, one column per stat.
 *
 * Every cell shows the raw gamebook number and, dimmer, the opponent-adjusted one the model
 * actually learns from (§4). The bar under each pair is the defence that game was played against,
 * ranked 1 (softest) to 32 (toughest) for that stat.
 */

import { multiplierColour, multiplierLabel, stat as fmtStat } from '../lib/format'
import type { GameLogCell, GameLogRow } from '../lib/types'

export interface GameLogTableProps {
  rows: GameLogRow[]
  statOrder: string[]
  /** stat key → display label, taken from the projections so the two tables agree. */
  labels: Record<string, string>
  selectedStat?: string
  onSelectStat?: (stat: string) => void
}

const RANKS = 32

export function GameLogTable({
  rows,
  statOrder,
  labels,
  selectedStat,
  onSelectStat,
}: GameLogTableProps) {
  if (!rows.length) {
    return (
      <section className="card p-3 text-sm text-chalk-dim">
        No game log. This player has no scored games in the window.
      </section>
    )
  }

  const present = new Set<string>()
  for (const row of rows) for (const key of Object.keys(row.cells)) present.add(key)
  const columns = statOrder.filter((key) => present.has(key))
  for (const key of present) if (!columns.includes(key)) columns.push(key)

  const priorCount = rows.filter((row) => row.prior_season).length

  return (
    <section className="card overflow-hidden">
      <header className="flex flex-wrap items-baseline justify-between gap-2 border-b border-ink-line px-3 py-2">
        <div>
          <h2 className="text-sm font-semibold text-chalk">Game log</h2>
          <p className="text-[11px] text-chalk-faint">
            raw <span className="text-chalk-dim">(opponent-adjusted)</span> · bar is the defence
            faced, wider means softer
          </p>
        </div>
        {priorCount > 0 ? (
          <span
            className="chip bg-warn/15 text-warn"
            title="Prior-season games carry a 0.85 discount in the projection. With zero 2026 games played, that discount applies to every player, so it does not move anyone's ranking relative to anyone else."
          >
            {priorCount === rows.length ? 'all prior season' : `${priorCount} prior season`}
          </span>
        ) : null}
      </header>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead className="bg-ink/60">
            <tr className="border-b border-ink-line">
              <th className="th">Game</th>
              {columns.map((key) => (
                <th
                  key={key}
                  className={`th text-right ${
                    key === selectedStat ? 'text-accent' : ''
                  } ${onSelectStat ? 'cursor-pointer' : ''}`}
                  onClick={onSelectStat ? () => onSelectStat(key) : undefined}
                >
                  {labels[key] ?? key.replace(/_/g, ' ')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={`${row.season}-${row.week}`}
                className="border-b border-ink-line/60 last:border-0"
              >
                <td className="td">
                  <div className="flex items-center gap-1.5">
                    <span className="num text-chalk">
                      {row.season} W{row.week}
                    </span>
                    <span className="text-chalk-dim">{row.opponent ? `vs ${row.opponent}` : '—'}</span>
                    {row.prior_season ? (
                      <span className="chip bg-ink-line text-chalk-faint" title="Prior season">
                        prior
                      </span>
                    ) : null}
                  </div>
                </td>
                {columns.map((key) => (
                  <LogCell
                    key={key}
                    cell={row.cells[key]}
                    opponent={row.opponent}
                    label={labels[key] ?? key}
                    highlighted={key === selectedStat}
                  />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

interface LogCellProps {
  cell: GameLogCell | undefined
  opponent: string | null
  label: string
  highlighted: boolean
}

function LogCell({ cell, opponent, label, highlighted }: LogCellProps) {
  if (!cell) {
    return (
      <td className={`td text-right text-chalk-faint ${highlighted ? 'bg-accent/5' : ''}`}>—</td>
    )
  }

  const rank = cell.opponent_rank
  const softness = rank === null ? 0.5 : (RANKS - rank + 1) / RANKS
  const soft = cell.opponent_multiplier > 1.02
  const tough = cell.opponent_multiplier < 0.98
  const tint = soft ? 'bg-good/[0.07]' : tough ? 'bg-bad/[0.07]' : ''
  const barColour = soft ? 'bg-good/70' : tough ? 'bg-bad/70' : 'bg-chalk-faint/60'

  return (
    <td
      className={`td text-right ${highlighted ? 'bg-accent/10' : tint}`}
      title={`${label} vs ${opponent ?? 'unknown'} — defence ranked ${
        rank === null ? 'n/a' : `${rank}/${RANKS}`
      } (1 = softest), ${multiplierLabel(cell.opponent_multiplier)} vs league average. Raw ${fmtStat(
        cell.raw_value,
      )} adjusts to ${fmtStat(cell.adjusted_value)}.`}
    >
      <div className="num whitespace-nowrap">
        <span className="text-chalk">{fmtStat(cell.raw_value)}</span>{' '}
        <span className={multiplierColour(cell.opponent_multiplier)}>
          ({fmtStat(cell.adjusted_value)})
        </span>
      </div>
      <div className="mt-1 flex items-center justify-end gap-1">
        <span className="h-1 w-10 overflow-hidden rounded-sm bg-ink-line">
          <span
            className={`block h-full ${barColour}`}
            style={{ width: `${Math.round(softness * 100)}%` }}
          />
        </span>
        <span className="num text-[10px] text-chalk-faint">{rank === null ? '—' : rank}</span>
      </div>
    </td>
  )
}

export default GameLogTable
