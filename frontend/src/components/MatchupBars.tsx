/**
 * The defensive signals feeding this player's projection, as percentile bars.
 *
 * The honest column is the last one. `mse_reduction_pct` is how much of the week-to-week squared
 * error the signal actually removes against a league-average prediction — for most receiving
 * metrics it is one to three percent. The matchup nudges the number; it does not decide the game.
 */

import { multiplierColour, multiplierLabel, pct } from '../lib/format'
import type { Matchup } from '../lib/types'

export interface MatchupBarsProps {
  matchup: Matchup[]
  opponent: string | null
  className?: string
}

function informativeness(row: Matchup): number {
  if (row.mse_reduction_pct !== null) return row.mse_reduction_pct
  if (row.predictive_weight !== null) return row.predictive_weight
  return -1
}

export function MatchupBars({ matchup, opponent, className }: MatchupBarsProps) {
  if (!matchup.length) {
    return (
      <section className={`card p-3 text-sm text-chalk-dim ${className ?? ''}`}>
        No opponent adjustments available for this position.
      </section>
    )
  }

  const rows = [...matchup].sort((a, b) => informativeness(b) - informativeness(a))
  const best = Math.max(...rows.map((row) => row.mse_reduction_pct ?? 0), 1)
  const worthwhile = rows.filter((row) => (row.mse_reduction_pct ?? 0) >= 1)

  return (
    <section className={`card overflow-hidden ${className ?? ''}`}>
      <header className="flex flex-wrap items-baseline justify-between gap-2 border-b border-ink-line px-3 py-2">
        <div>
          <h2 className="text-sm font-semibold text-chalk">
            Matchup{opponent ? ` vs ${opponent}` : ''}
          </h2>
          <p className="text-[11px] text-chalk-faint">
            Bar is the opponent&rsquo;s percentile among the 32 defences. Ordered by how much error
            each signal removes.
          </p>
        </div>
      </header>

      <ul className="divide-y divide-ink-line/60">
        {rows.map((row) => {
          const percentile = Math.min(1, Math.max(0, row.percentile))
          return (
            <li key={row.metric} className="px-3 py-2">
              <div className="flex items-baseline justify-between gap-3">
                <span className="truncate text-sm text-chalk" title={row.metric}>
                  {row.label}
                </span>
                <span className="flex shrink-0 items-baseline gap-2">
                  <span className={`num text-sm font-semibold ${multiplierColour(row.multiplier)}`}>
                    {multiplierLabel(row.multiplier)}
                  </span>
                  <span className="num text-[11px] text-chalk-faint" title="Rank among 32 defences">
                    #{row.rank}
                  </span>
                </span>
              </div>

              <div className="mt-1.5 flex items-center gap-2">
                <span className="relative h-2 flex-1 overflow-hidden rounded-sm bg-ink-line">
                  <span
                    className={`block h-full ${
                      row.multiplier > 1.02
                        ? 'bg-good/70'
                        : row.multiplier < 0.98
                          ? 'bg-bad/70'
                          : 'bg-chalk-faint/60'
                    }`}
                    style={{ width: `${Math.round(percentile * 100)}%` }}
                  />
                  <span
                    className="absolute inset-y-0 left-1/2 w-px bg-ink"
                    aria-hidden
                    title="league median"
                  />
                </span>
                <span className="num w-10 shrink-0 text-right text-[11px] text-chalk-faint">
                  {pct(percentile, 0)}
                </span>
                <span
                  className="num w-24 shrink-0 text-right text-[11px]"
                  title={
                    row.mse_reduction_pct === null
                      ? 'Not measured for this metric.'
                      : `Knowing this opponent adjustment removes ${row.mse_reduction_pct.toFixed(
                          1,
                        )}% of the squared error versus predicting the league average. Everything else is player, script and luck.`
                  }
                >
                  {row.mse_reduction_pct === null ? (
                    <span className="text-chalk-faint">—</span>
                  ) : (
                    <span className="inline-flex items-center gap-1">
                      <span className="inline-block h-1 w-8 overflow-hidden rounded-sm bg-ink-line align-middle">
                        <span
                          className="block h-full bg-accent/70"
                          style={{
                            width: `${Math.round((row.mse_reduction_pct / best) * 100)}%`,
                          }}
                        />
                      </span>
                      <span className="text-chalk-dim">{row.mse_reduction_pct.toFixed(1)}%</span>
                    </span>
                  )}
                </span>
              </div>
            </li>
          )
        })}
      </ul>

      <p className="border-t border-ink-line px-3 py-2 text-[11px] text-chalk-faint">
        The right-hand number is error actually removed, not importance.{' '}
        {worthwhile.length === 0
          ? 'None of these signals removes even 1% of the week-to-week error here: the matchup is close to noise for this player, and volume is the whole story.'
          : worthwhile.length === rows.length
            ? `All ${rows.length} clear 1%. A soft matchup shifts the projection a few percent; it does not decide the game.`
            : `Only ${worthwhile.length} of ${rows.length} clear 1%. A soft matchup shifts the projection a few percent; it does not decide the game.`}
      </p>
    </section>
  )
}

export default MatchupBars
