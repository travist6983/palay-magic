/**
 * Opponent matchup: the defensive adjustment already applied to the row's first headline
 * projection, plus where that puts the opponent among the 32.
 *
 * Rank 1 is the SOFTEST defence for the stat (the highest multiplier), so a low rank and a green
 * chip mean the same thing. The multiplier is already the number the projection was multiplied by.
 *
 * It is NOT a rate of the projected stat that the defence gives up, and it must not be described as
 * one: backend/models/stats.py maps each stat to the defence metric that adjusts it, and for two of
 * the six boards that metric measures something else entirely -- LB is scaled by offensive plays
 * allowed and K by field-goal attempts allowed. "Allows 0.5% more tackles + assists" would be flatly
 * untrue on the LB board.
 */

import { multiplierColour, multiplierLabel } from '../lib/format'

export interface MatchupCellProps {
  /** The defensive multiplier that adjusts the position's first headline stat. 1.00 = league average. */
  multiplier: number
  /** Defensive rank, 1 = softest. Null when the opponent has no measured defence yet. */
  rank: number | null
  /** Opponent team abbreviation, used only in the tooltip. */
  opponent?: string | null
  /** Headline stat label, e.g. "Rushing yards", used only in the tooltip. */
  statLabel?: string | null
  /** Number of ranked defences. */
  teamCount?: number
  className?: string
}

export function ordinal(n: number): string {
  const mod100 = Math.abs(n) % 100
  if (mod100 >= 11 && mod100 <= 13) return `${n}th`
  switch (Math.abs(n) % 10) {
    case 1:
      return `${n}st`
    case 2:
      return `${n}nd`
    case 3:
      return `${n}rd`
    default:
      return `${n}th`
  }
}

/** "3rd softest" at the friendly end, a plain "30th of 32" elsewhere -- the colour carries the rest. */
function rankLabel(rank: number | null, teamCount: number): string {
  if (rank === null) return 'unranked'
  if (rank <= 8) return `${ordinal(rank)} softest`
  return `${ordinal(rank)} of ${teamCount}`
}

export function MatchupCell({
  multiplier,
  rank,
  opponent,
  statLabel,
  teamCount = 32,
  className,
}: MatchupCellProps) {
  const delta = (multiplier - 1) * 100
  const who = opponent ?? 'This opponent'
  // Name the METRIC, not the projected stat. They differ on two of the six boards -- a kicker's
  // headline is kicking points while the metric is FG attempts allowed, and a linebacker's is
  // tackles while the metric is offensive plays run -- so inferring the wording from the
  // projection label stated something false on those pages.
  const what = statLabel ? `on ${statLabel.toLowerCase()}` : 'on this position\'s volume metric'
  const phrase =
    Math.abs(delta) < 0.5
      ? `grades as a league-average matchup ${what}`
      : `grades as a ${Math.abs(delta).toFixed(1)}% ${delta > 0 ? 'softer' : 'tougher'} matchup ${what} than a league-average defence`
  const title =
    `${who} ${phrase} ` +
    `(multiplier ${multiplier.toFixed(3)}; the projection is already scaled by it).` +
    (rank === null
      ? ' No defensive rank yet for this opponent.'
      : ` Rank ${rank} of ${teamCount}, where 1 is the softest matchup for this stat.`)

  return (
    <div className={`leading-tight ${className ?? ''}`} title={title}>
      <span className={`chip num bg-ink-line/60 ${multiplierColour(multiplier)}`}>
        {multiplierLabel(multiplier)}
      </span>
      <div className="mt-0.5 text-[11px] text-chalk-faint">{rankLabel(rank, teamCount)}</div>
    </div>
  )
}

export default MatchupCell
