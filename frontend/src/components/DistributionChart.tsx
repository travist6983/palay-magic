/**
 * The selected stat's distribution, with the user's line marked and everything beyond it shaded.
 *
 * The curve comes from densityCurve() over the parameters the API returned — the same parameters
 * that produced the median in the table and the steps in the Show-math drawer (D10). Redrawing on
 * a keystroke costs one array of PMF evaluations and no network.
 */

import { useId, useMemo } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { densityCurve, formatOdds, probExact, probOver, probUnder } from '../lib/distributions'
import { pct } from '../lib/format'
import type { DistributionParams } from '../lib/types'

export interface DistributionChartProps {
  /** Human label for the stat, e.g. "Receiving yards". */
  label: string
  dist: DistributionParams
  /** null while the line box is empty or half-typed. */
  line: number | null
  /** "if he plays" / "including the chance he doesn't" — shown so the curve is never ambiguous. */
  modeNote?: string
  /** Extra context under the chart, e.g. the settlement rule. */
  note?: string
  height?: number
}

interface Point {
  x: number
  y: number
}

/** Bars read better than an area once the support is small enough to count. */
const MAX_BAR_POINTS = 16

function formatX(x: number, integerish: boolean): string {
  return integerish ? String(Math.round(x)) : x.toFixed(1)
}

export function DistributionChart({
  label,
  dist,
  line,
  modeNote,
  note,
  height = 220,
}: DistributionChartProps) {
  const gradientId = useId().replace(/[^a-zA-Z0-9-]/g, '')
  const points = useMemo<Point[]>(() => densityCurve(dist, 60), [dist])

  const hasLine = line !== null && Number.isFinite(line)
  const pOver = hasLine ? probOver(dist, line) : NaN
  const pUnder = hasLine ? probUnder(dist, line) : NaN
  const pPush = hasLine ? probExact(dist, line) : 0

  if (!points.length) {
    return (
      <div className="card p-3 text-sm text-chalk-dim">
        No distribution shape available for {label}.
      </div>
    )
  }

  const xs = points.map((p) => p.x)
  const xMin = Math.min(...xs)
  const xMax = Math.max(...xs)
  const integerish = points.every((p) => Number.isInteger(p.x))
  const useBars = points.length <= MAX_BAR_POINTS

  // Hard stop in the gradient at exactly the line, so the shading edge is the line itself
  // rather than the nearest sample. Clamped so a line off the plotted range still reads.
  const cut = xMax > xMin && hasLine ? Math.min(1, Math.max(0, (line - xMin) / (xMax - xMin))) : 1

  const data = points.map((p) => ({
    x: p.x,
    y: p.y,
    under: hasLine && p.x > line ? 0 : p.y,
    over: hasLine && p.x > line ? p.y : 0,
  }))

  const axisTick = { fill: 'currentColor', fontSize: 10 }
  const axisLine = { stroke: 'currentColor', strokeOpacity: 0.35 }

  return (
    <div className="card flex flex-col">
      <div className="flex items-baseline justify-between gap-2 border-b border-ink-line px-3 py-2">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold text-chalk">{label}</div>
          <div className="text-[11px] text-chalk-faint">
            {dist.family.replace(/_/g, ' ')}
            {modeNote ? ` · ${modeNote}` : ''}
          </div>
        </div>
        <div className="shrink-0 text-right">
          <div className="num text-lg font-semibold leading-none text-chalk">
            {hasLine ? pct(pOver, 1) : '—'}
          </div>
          <div className="num text-[11px] text-chalk-faint">
            over {hasLine ? formatX(line, false) : '—'} · {hasLine ? formatOdds(pOver) : '—'}
          </div>
        </div>
      </div>

      <div className="px-1 pt-2 text-chalk-faint">
        <ResponsiveContainer width="100%" height={height}>
          {useBars ? (
            <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
              <CartesianGrid vertical={false} stroke="currentColor" strokeOpacity={0.25} />
              <XAxis
                dataKey="x"
                tick={axisTick}
                tickLine={false}
                axisLine={axisLine}
                tickFormatter={(v: number) => formatX(v, integerish)}
              />
              <YAxis hide domain={[0, 'dataMax']} />
              <Tooltip
                cursor={{ fill: 'currentColor', fillOpacity: 0.08 }}
                content={(props) => (
                  <DensityTooltip
                    active={props.active}
                    x={readNumber(props.label)}
                    y={readPoint(props.payload)}
                    integerish={integerish}
                  />
                )}
              />
              <Bar
                dataKey="under"
                stackId="d"
                className="text-chalk-faint"
                fill="currentColor"
                fillOpacity={0.55}
                isAnimationActive={false}
              />
              <Bar
                dataKey="over"
                stackId="d"
                className="text-accent"
                fill="currentColor"
                fillOpacity={0.9}
                isAnimationActive={false}
              />
            </BarChart>
          ) : (
            <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
              <defs>
                <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
                  <stop
                    offset={cut}
                    className="text-chalk-faint"
                    stopColor="currentColor"
                    stopOpacity={0.35}
                  />
                  <stop
                    offset={cut}
                    className="text-accent"
                    stopColor="currentColor"
                    stopOpacity={0.6}
                  />
                </linearGradient>
              </defs>
              <CartesianGrid vertical={false} stroke="currentColor" strokeOpacity={0.25} />
              <XAxis
                dataKey="x"
                type="number"
                domain={[xMin, xMax]}
                tick={axisTick}
                tickLine={false}
                axisLine={axisLine}
                tickFormatter={(v: number) => formatX(v, integerish)}
              />
              <YAxis hide domain={[0, 'dataMax']} />
              <Tooltip
                cursor={{ stroke: 'currentColor', strokeOpacity: 0.35 }}
                content={(props) => (
                  <DensityTooltip
                    active={props.active}
                    x={readNumber(props.label)}
                    y={readPoint(props.payload)}
                    integerish={integerish}
                  />
                )}
              />
              <Area
                type="monotone"
                dataKey="y"
                className="text-chalk-dim"
                stroke="currentColor"
                strokeWidth={1.25}
                fill={`url(#${gradientId})`}
                isAnimationActive={false}
              />
              {hasLine ? (
                <ReferenceLine
                  x={line}
                  className="text-warn"
                  stroke="currentColor"
                  strokeDasharray="3 3"
                  label={{
                    value: `${formatX(line, false)} → ${pct(pOver, 0)}`,
                    position: 'insideTopRight',
                    fill: 'currentColor',
                    fontSize: 10,
                  }}
                />
              ) : null}
            </AreaChart>
          )}
        </ResponsiveContainer>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-ink-line px-3 py-2 text-[11px]">
        <span className="flex items-center gap-1.5 text-chalk-dim">
          <span className="inline-block h-2 w-3 rounded-sm bg-accent/70" />
          over {hasLine ? formatX(line, false) : '—'}
          <span className="num text-chalk">{hasLine ? pct(pOver, 1) : '—'}</span>
          <span className="num text-chalk-faint">{hasLine ? formatOdds(pOver) : ''}</span>
        </span>
        <span className="flex items-center gap-1.5 text-chalk-dim">
          <span className="inline-block h-2 w-3 rounded-sm bg-chalk-faint/50" />
          under
          <span className="num text-chalk">{hasLine ? pct(pUnder, 1) : '—'}</span>
          <span className="num text-chalk-faint">{hasLine ? formatOdds(pUnder) : ''}</span>
        </span>
        {pPush > 0.0005 ? (
          <span className="text-warn" title="A whole-number line can push. Books post half points to prevent this.">
            push {pct(pPush, 1)}
          </span>
        ) : null}
      </div>

      {note ? <div className="border-t border-ink-line px-3 py-2 text-[11px] text-chalk-faint">{note}</div> : null}
    </div>
  )
}

function readNumber(value: unknown): number | null {
  const n = typeof value === 'number' ? value : Number.parseFloat(String(value))
  return Number.isFinite(n) ? n : null
}

function readPoint(payload: unknown): number | null {
  if (!Array.isArray(payload) || !payload.length) return null
  const first: unknown = payload[0]
  if (typeof first !== 'object' || first === null) return null
  const inner = (first as { payload?: unknown }).payload
  if (typeof inner !== 'object' || inner === null) return null
  return readNumber((inner as { y?: unknown }).y)
}

function DensityTooltip({
  active,
  x,
  y,
  integerish,
}: {
  active?: boolean
  x: number | null
  y: number | null
  integerish: boolean
}) {
  if (!active || x === null) return null
  return (
    <div className="card px-2 py-1 text-[11px] shadow-lg">
      <div className="num text-chalk">{formatX(x, integerish)}</div>
      {y !== null ? <div className="num text-chalk-faint">mass {pct(y, 1)}</div> : null}
    </div>
  )
}

export default DistributionChart
