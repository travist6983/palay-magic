/**
 * One row per stat: the projection interval, your line, and what the line is worth.
 *
 * P(over) is computed here, in the browser, from the distribution parameters the API already
 * returned (D10, §8). Typing in a line box never touches the network.
 */

import { useMemo } from 'react'
import {
  defaultLine,
  densityCurve,
  formatOdds,
  probExact,
  probOver,
  probUnder,
} from '../lib/distributions'
import { pct, stat as fmtStat } from '../lib/format'
import type { DistributionParams, Projection } from '../lib/types'
import { LineInput } from './LineInput'

/** "if he plays" vs "including the chance he doesn't" (§5.7). */
export type ProjectionMode = 'conditional' | 'unconditional'

export interface ProjectionTableProps {
  projections: Projection[]
  statOrder: string[]
  /** Raw line text per stat. A missing key means "still on the default". */
  lines: Record<string, string>
  onLineChange: (stat: string, value: string) => void
  onLineReset: (stat: string) => void
  selectedStat: string
  onSelectStat: (stat: string) => void
  mode: ProjectionMode
  onModeChange: (mode: ProjectionMode) => void
  /** False when no stat has a materially different unconditional projection. */
  canToggleMode: boolean
  onShowMath?: (stat: string) => void
}

/** The distribution the current mode is showing. Falls back when the API sent no unconditional. */
export function activeDistribution(projection: Projection, mode: ProjectionMode): DistributionParams {
  if (mode === 'unconditional' && projection.unconditional) return projection.unconditional
  return projection.distribution
}

/** Two projections differ enough to be worth a toggle if any quantile moves by ~1%. */
export function distributionsDiffer(a: DistributionParams, b: DistributionParams): boolean {
  const scale = Math.max(1, Math.abs(a.mean))
  return (
    Math.abs(a.mean - b.mean) / scale > 0.01 ||
    a.median !== b.median ||
    a.p25 !== b.p25 ||
    a.p75 !== b.p75
  )
}

export function hasMeaningfulUnconditional(projections: Projection[]): boolean {
  return projections.some(
    (p) => p.unconditional !== null && distributionsDiffer(p.distribution, p.unconditional),
  )
}

/** Integer-valued stats print as integers; longest_* and derived means keep a decimal. */
function quantileText(dist: DistributionParams, value: number): string {
  // Math.round would turn a missing quantile into the string "NaN"; a gap prints as a dash.
  if (!Number.isFinite(value)) return '—'
  return dist.integer_valued ? String(Math.round(value)) : fmtStat(value)
}

export function ProjectionTable({
  projections,
  statOrder,
  lines,
  onLineChange,
  onLineReset,
  selectedStat,
  onSelectStat,
  mode,
  onModeChange,
  canToggleMode,
  onShowMath,
}: ProjectionTableProps) {
  const byStat = useMemo(() => {
    const map = new Map<string, Projection>()
    for (const p of projections) map.set(p.stat, p)
    return map
  }, [projections])

  const ordered = useMemo(() => {
    const seen = new Set<string>()
    const out: Projection[] = []
    for (const key of statOrder) {
      const p = byStat.get(key)
      if (p) {
        out.push(p)
        seen.add(key)
      }
    }
    for (const p of projections) if (!seen.has(p.stat)) out.push(p)
    return out
  }, [byStat, projections, statOrder])

  if (!ordered.length) {
    return <div className="card p-3 text-sm text-chalk-dim">No projections for this player.</div>
  }

  return (
    <div className="card overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-ink-line px-3 py-2">
        <div>
          <h2 className="text-sm font-semibold text-chalk">Projections</h2>
          <p className="text-[11px] text-chalk-faint">
            P(over) is recomputed in this tab from the returned parameters — no request per
            keystroke.
          </p>
        </div>
        {canToggleMode ? (
          <div
            className="inline-flex overflow-hidden rounded border border-ink-line text-[11px]"
            role="group"
            aria-label="projection basis"
          >
            <button
              type="button"
              onClick={() => onModeChange('conditional')}
              className={`px-2 py-1 ${
                mode === 'conditional' ? 'bg-accent/20 text-accent' : 'text-chalk-dim hover:text-chalk'
              }`}
              title="Conditional on the player being active."
            >
              if he plays
            </button>
            <button
              type="button"
              onClick={() => onModeChange('unconditional')}
              className={`border-l border-ink-line px-2 py-1 ${
                mode === 'unconditional'
                  ? 'bg-accent/20 text-accent'
                  : 'text-chalk-dim hover:text-chalk'
              }`}
              title="Blended with the probability that he does not play. An inactive player voids the bet at the book; this number does not."
            >
              including the chance he doesn&rsquo;t
            </button>
          </div>
        ) : null}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead className="bg-ink/60">
            <tr className="border-b border-ink-line">
              <th className="th">Stat</th>
              <th className="th text-right">Median</th>
              <th className="th text-right">p25</th>
              <th className="th text-right">p75</th>
              <th className="th">Your line</th>
              <th className="th text-right">P(over)</th>
              <th className="th text-right">Fair odds</th>
              <th className="th">Distribution</th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((projection) => (
              <ProjectionRow
                key={projection.stat}
                projection={projection}
                mode={mode}
                lineText={lines[projection.stat]}
                onLineChange={onLineChange}
                onLineReset={onLineReset}
                selected={projection.stat === selectedStat}
                onSelectStat={onSelectStat}
                onShowMath={onShowMath}
              />
            ))}
          </tbody>
        </table>
      </div>

      <p className="border-t border-ink-line px-3 py-2 text-[11px] text-chalk-faint">
        Hover a stat name for its settlement rule. Lines default to the median offset to a half
        point, which is how a book prices out a push.
      </p>
    </div>
  )
}

interface ProjectionRowProps {
  projection: Projection
  mode: ProjectionMode
  lineText: string | undefined
  onLineChange: (stat: string, value: string) => void
  onLineReset: (stat: string) => void
  selected: boolean
  onSelectStat: (stat: string) => void
  onShowMath?: (stat: string) => void
}

function ProjectionRow({
  projection,
  mode,
  lineText,
  onLineChange,
  onLineReset,
  selected,
  onSelectStat,
  onShowMath,
}: ProjectionRowProps) {
  const dist = activeDistribution(projection, mode)
  const blended =
    projection.unconditional !== null &&
    distributionsDiffer(projection.distribution, projection.unconditional)
  const fallback = String(defaultLine(projection.distribution))
  const text = lineText ?? fallback
  const line = Number.parseFloat(text)
  const hasLine = text.trim() !== '' && Number.isFinite(line)

  const pOver = hasLine ? probOver(dist, line) : NaN
  const pUnder = hasLine ? probUnder(dist, line) : NaN
  const push = hasLine ? probExact(dist, line) : 0

  return (
    <tr
      onClick={() => onSelectStat(projection.stat)}
      className={`cursor-pointer border-b border-ink-line/60 last:border-0 ${
        selected ? 'bg-accent/10' : 'hover:bg-ink-line/40'
      }`}
    >
      <td className="td">
        <div className="flex items-center gap-1.5">
          <span
            className={`inline-block h-3 w-0.5 rounded-sm ${selected ? 'bg-accent' : 'bg-transparent'}`}
            aria-hidden
          />
          <span
            className={
              projection.settlement_note
                ? 'text-chalk underline decoration-chalk-faint decoration-dotted underline-offset-4'
                : 'text-chalk'
            }
            title={projection.settlement_note || undefined}
          >
            {projection.label}
          </span>
          {projection.high_variance ? (
            <span
              className="chip bg-warn/15 text-warn"
              title="Near-random week to week. The reference doc puts sacks, interceptions and passes defended in this bucket: treat them as small situational legs, not core bets."
            >
              high var
            </span>
          ) : null}
          {mode === 'unconditional' && projection.play_probability < 0.999 ? (
            blended ? (
              <span
                className="chip bg-accent/15 text-accent"
                title={`Blended with P(plays) = ${pct(projection.play_probability, 1)}. At a book an inactive player voids the bet instead, so this basis answers "what does he do this week", not "what does this ticket pay".`}
              >
                ×{projection.play_probability.toFixed(2)}
              </span>
            ) : (
              <span
                className="chip bg-ink-line text-chalk-faint"
                title="The API returned an identical unconditional distribution for this stat, so the toggle does not move this row."
              >
                unblended
              </span>
            )
          ) : null}
        </div>
      </td>
      <td className="td num text-right font-semibold text-chalk">
        {quantileText(dist, dist.median)}
      </td>
      <td className="td num text-right text-chalk-dim">{quantileText(dist, dist.p25)}</td>
      <td className="td num text-right text-chalk-dim">{quantileText(dist, dist.p75)}</td>
      <td className="td" onClick={(event) => event.stopPropagation()}>
        <LineInput
          value={text}
          label={`${projection.label} line`}
          isDefault={lineText === undefined || lineText === fallback}
          onReset={() => onLineReset(projection.stat)}
          onChange={(next) => {
            onSelectStat(projection.stat)
            onLineChange(projection.stat, next)
          }}
        />
      </td>
      <td className="td text-right">
        <div className="num font-semibold text-chalk">{hasLine ? pct(pOver, 1) : '—'}</div>
        <div className="mt-0.5 h-1 w-full min-w-[52px] overflow-hidden rounded-sm bg-ink-line">
          <div
            className="h-full bg-accent"
            style={{ width: hasLine ? `${Math.round(pOver * 100)}%` : '0%' }}
          />
        </div>
      </td>
      <td className="td num text-right">
        <div className="text-chalk">{hasLine ? formatOdds(pOver) : '—'}</div>
        <div className="text-[11px] text-chalk-faint" title="Fair price on the under, before vig.">
          u {hasLine ? formatOdds(pUnder) : '—'}
        </div>
        {push > 0.0005 ? (
          <div className="text-[10px] text-warn" title="A whole-number line can push.">
            push {pct(push, 0)}
          </div>
        ) : null}
      </td>
      <td className="td">
        <div className="flex items-center gap-2">
          <Sparkline dist={dist} line={hasLine ? line : null} />
          <div className="text-[10px] leading-tight text-chalk-faint">
            <div>{dist.family.replace(/_/g, ' ')}</div>
            {onShowMath ? (
              <button
                type="button"
                className="uppercase tracking-wide hover:text-accent"
                onClick={(event) => {
                  event.stopPropagation()
                  onSelectStat(projection.stat)
                  onShowMath(projection.stat)
                }}
              >
                show math
              </button>
            ) : null}
          </div>
        </div>
      </td>
    </tr>
  )
}

/** A 24-bucket density strip. Buckets past the line pick up the accent, matching the big chart. */
function Sparkline({ dist, line }: { dist: DistributionParams; line: number | null }) {
  const points = useMemo(() => densityCurve(dist, 24), [dist])
  if (!points.length) return <span className="text-chalk-faint">—</span>

  const peak = Math.max(...points.map((p) => p.y), 1e-9)
  const width = 72
  const height = 18
  const barWidth = width / points.length

  return (
    <svg width={width} height={height} aria-hidden className="shrink-0">
      {points.map((point, index) => {
        const barHeight = Math.max(1, (point.y / peak) * (height - 2))
        const beyond = line !== null && point.x > line
        return (
          <rect
            key={index}
            x={index * barWidth}
            y={height - barHeight}
            width={Math.max(1, barWidth - 0.6)}
            height={barHeight}
            className={beyond ? 'fill-accent' : 'fill-chalk-faint/60'}
          />
        )
      })}
    </svg>
  )
}

export default ProjectionTable
