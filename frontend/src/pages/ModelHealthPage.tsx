/**
 * Model health (§6): the page you read before you trust a projection.
 *
 * Two questions, in order. Is the distribution honest — does the actual land inside the interval
 * half the time, and is P(over) worth betting? And is the opponent adjustment doing anything —
 * how much out-of-sample error does each defensive metric actually remove?
 */

import { CalibrationTable } from '../components/CalibrationTable'
import { DefenseTable } from '../components/DefenseTable'

export interface ModelHealthPageProps {
  /** Position the defence table opens on. Defaults to WR. */
  initialPosition?: string
}

export function ModelHealthPage({ initialPosition = 'WR' }: ModelHealthPageProps) {
  return (
    <div className="mx-auto max-w-[1400px] space-y-4 px-4 py-4">
      <header>
        <h1 className="text-lg font-semibold tracking-tight text-chalk">Model health</h1>
        <p className="mt-1 max-w-3xl text-xs leading-relaxed text-chalk-dim">
          Everything below is out of sample. The calibration table replays a finished season week by
          week, rebuilding multipliers, environment and projections against a frozen cutoff before
          scoring them, so it measures the model you are actually about to bet. A stat whose
          coverage or PIT sits outside the band has a distribution that is the wrong width: the
          median may still be fine, but its P(over) is over- or under-confident and should be read
          as a direction, not a price.
        </p>
      </header>

      <CalibrationTable />

      <DefenseTable initialPosition={initialPosition} />
    </div>
  )
}

export default ModelHealthPage
