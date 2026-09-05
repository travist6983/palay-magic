/**
 * Injury designation + empirical play probability.
 *
 * Renders nothing for a healthy player: a board of ten rows should draw the eye only to the
 * players whose availability is actually in question. The probability is the D11 rate — measured
 * per (designation, final practice status, role bucket), not a book's or the league's guess.
 */

import { pct } from '../lib/format'

export interface InjuryChipProps {
  /** Report designation as the league posts it: Questionable, Doubtful, Out, IR, ... */
  status: string | null
  /** P(the player suits up), already conditioned on his role (D11). */
  playProbability: number
  className?: string
}

type Severity = 'out' | 'doubtful' | 'questionable' | 'minor'

/** Sleeper/league designations, normalised to the four buckets that change how a bet is priced. */
function severityOf(status: string): Severity {
  const s = status.trim().toLowerCase()
  if (s === 'out' || s.startsWith('ir') || s.startsWith('pup') || s.startsWith('sus') || s.startsWith('nfi') || s.startsWith('did not report')) {
    return 'out'
  }
  if (s.startsWith('doubt')) return 'doubtful'
  if (s.startsWith('quest')) return 'questionable'
  return 'minor'
}

function abbreviate(status: string): string {
  const s = status.trim()
  const lower = s.toLowerCase()
  if (lower.startsWith('quest')) return 'Q'
  if (lower.startsWith('doubt')) return 'D'
  if (lower === 'out') return 'OUT'
  if (lower.startsWith('prob')) return 'P'
  if (lower.startsWith('ir')) return 'IR'
  if (lower.startsWith('pup')) return 'PUP'
  if (lower.startsWith('sus')) return 'SUS'
  if (lower.startsWith('nfi')) return 'NFI'
  return s.slice(0, 3).toUpperCase()
}

const TONE: Record<Severity, string> = {
  out: 'bg-bad/15 text-bad',
  doubtful: 'bg-bad/10 text-bad/90',
  questionable: 'bg-warn/15 text-warn',
  minor: 'bg-ink-line text-chalk-dim',
}

/**
 * Sleeper emits these where a healthy player carries no designation at all; the backend treats them
 * as "no designation" for exactly that reason (backend/models/injury.py::NON_DESIGNATIONS, "treating
 * NA as one would push 95 healthy players off the board"). They reach the board unchanged --
 * `rankings.injury_status` holds 'NA' on 19 rows, several of them rank 1 or 2 -- so reading one as a
 * designation invents an injury on a fully healthy player.
 */
const NON_DESIGNATIONS = new Set([
  '',
  '-',
  'na',
  'n/a',
  'healthy',
  'active',
  'none',
  'null',
  'no injury',
])

const PROBABILITY_NOTE =
  'Play probability is empirical, not a designation lookup: the rate at which players with this ' +
  'designation and final practice status actually suited up in 2023-2025, conditioned on the ' +
  "player's role (mean snap share over the previous four weeks). A Questionable starter is a very " +
  'different bet from a Questionable rotational player. Projections on this row are conditional on ' +
  'him playing.'

export function InjuryChip({ status, playProbability, className }: InjuryChipProps) {
  const raw = (status ?? '').trim()
  const undesignated = NON_DESIGNATIONS.has(raw.toLowerCase())
  const certain = !Number.isFinite(playProbability) || playProbability >= 0.995

  // No designation and certain to play: draw nothing at all.
  if (undesignated && certain) return null

  const severity: Severity = undesignated ? 'minor' : severityOf(raw)
  const label = undesignated ? 'RISK' : abbreviate(raw)
  const designation = undesignated ? 'No designation posted' : raw
  const title = `${designation} - plays ${pct(playProbability, 0)} of the time. ${PROBABILITY_NOTE}`

  return (
    <span
      className={`chip gap-1 ${TONE[severity]} ${className ?? ''}`}
      title={title}
    >
      <span>{label}</span>
      <span className="num font-normal normal-case tracking-normal opacity-80">
        {pct(playProbability, 0)}
      </span>
    </span>
  )
}

export default InjuryChip
