/**
 * Opponent matchup: how much more (or less) of this stat the opponent allows than a league-average
 * defence, plus where that puts them among the 32.
 *
 * Rank 1 is the SOFTEST defence for the stat (the highest multiplier), so a low rank and a green
 * chip mean the same thing. The multiplier is already the number the projection was multiplied by.
 */

import { multiplierColour, multiplierLabel } from '../lib/format'

export interface MatchupCellProps {
  /** Defensive multiplier for the position's headline volume metric. 1.00 = league average. */
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
  const what = statLabel ? statLabel.toLowerCase() : 'this stat'
  const phrase =
    Math.abs(delta) < 0.5
      ? `${what} at about the league-average rate`
      : `${Math.abs(delta).toFixed(1)}% ${delta > 0 ? 'more' : 'less'} ${what} than a league-average defence`
  const title =
    `${who} allows ${phrase} ` +
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
