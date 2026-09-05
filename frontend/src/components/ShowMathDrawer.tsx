/**
 * "Every projection is reproducible" (§1.4).
 *
 * The API ships the full trace of intermediate values that produced one stat's distribution. This
 * renders them grouped by the spec section that computed them, numbered in the order they were
 * computed, so the chain from team pace to the number in the table can be read straight down —
 * and checked with a calculator.
 */

import type { DistributionParams, MathStep } from '../lib/types'

export interface ShowMathDrawerProps {
  stat: string
  label: string
  steps: MathStep[]
  open: boolean
  onToggle: (stat: string) => void
  /** Rendered as the closing line: the distribution these steps produced. */
  result?: DistributionParams
}

/** Enough significant figures to re-derive the next line by hand, at every magnitude. */
export function formatMathValue(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return '—'
  const magnitude = Math.abs(value)
  if (magnitude === 0) return '0'
  if (magnitude >= 1000) return value.toFixed(0)
  if (magnitude >= 100) return value.toFixed(1)
  if (magnitude >= 10) return value.toFixed(2)
  if (magnitude >= 1) return value.toFixed(3)
  if (magnitude >= 0.01) return value.toFixed(4)
  return value.toPrecision(3)
}

/** Quantiles of an integer-valued stat are counts, not measurements. */
function formatQuantile(dist: DistributionParams, value: number): string {
  // Math.round(NaN) stringifies to "NaN"; a missing quantile reads as a dash like every other gap.
  if (!Number.isFinite(value)) return '—'
  return dist.integer_valued ? String(Math.round(value)) : formatMathValue(value)
}

interface Section {
  name: string
  steps: { step: MathStep; index: number }[]
}

/** Group by section, keeping first-appearance order so the chain still reads top to bottom. */
function groupSections(steps: MathStep[]): Section[] {
  const sections: Section[] = []
  const byName = new Map<string, Section>()
  steps.forEach((step, index) => {
    let section = byName.get(step.section)
    if (!section) {
      section = { name: step.section, steps: [] }
      byName.set(step.section, section)
      sections.push(section)
    }
    section.steps.push({ step, index })
  })
  return sections
}

/** "5.3 environment" → ["§5.3", "environment"]. */
function splitSectionName(name: string): [string, string] {
  const match = /^([\d.]+)\s+(.*)$/.exec(name)
  return match ? [`§${match[1]}`, match[2]] : ['', name]
}

export function ShowMathDrawer({ stat, label, steps, open, onToggle, result }: ShowMathDrawerProps) {
  const sections = groupSections(steps)

  return (
    <div className="card overflow-hidden" id={`math-${stat}`}>
      <button
        type="button"
        onClick={() => onToggle(stat)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-ink-line/40"
      >
        <span className={`text-chalk-faint transition-transform ${open ? 'rotate-90' : ''}`}>
          &rsaquo;
        </span>
        <span className="text-sm font-semibold text-chalk">{label}</span>
        <span className="num text-[11px] text-chalk-faint">
          {steps.length} step{steps.length === 1 ? '' : 's'}
        </span>
        <span className="ml-auto text-[10px] uppercase tracking-wide text-chalk-faint">
          {open ? 'hide math' : 'show math'}
        </span>
      </button>

      {open ? (
        steps.length === 0 ? (
          <p className="border-t border-ink-line px-3 py-2 text-[11px] text-chalk-faint">
            No trace was recorded for this stat.
          </p>
        ) : (
          <div className="border-t border-ink-line">
            {sections.map((section) => {
              const [number, name] = splitSectionName(section.name)
              return (
                <div key={section.name} className="border-b border-ink-line/60 last:border-0">
                  <div className="flex items-baseline gap-2 bg-ink/60 px-3 py-1">
                    <span className="num text-[11px] font-semibold text-accent">{number}</span>
                    <span className="text-[11px] uppercase tracking-wide text-chalk-faint">
                      {name}
                    </span>
                  </div>
                  <dl>
                    {section.steps.map(({ step, index }) => (
                      <div
                        key={`${index}-${step.label}`}
                        className="flex items-baseline gap-3 px-3 py-1 odd:bg-ink/30"
                      >
                        <span className="num w-6 shrink-0 text-right text-[10px] text-chalk-faint">
                          {index + 1}
                        </span>
                        <dt className="min-w-0 flex-1 truncate text-sm text-chalk-dim">
                          {step.label}
                        </dt>
                        {step.detail ? (
                          <dd className="hidden shrink-0 text-[11px] text-chalk-faint sm:block">
                            {step.detail}
                          </dd>
                        ) : null}
                        <dd className="num w-28 shrink-0 text-right text-sm text-chalk">
                          {formatMathValue(step.value)}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </div>
              )
            })}

            {result ? (
              <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-t border-ink-line bg-ink/60 px-3 py-2 text-[11px]">
                <span className="uppercase tracking-wide text-chalk-faint">produces</span>
                <span className="text-chalk">{result.family.replace(/_/g, ' ')}</span>
                <span className="num text-chalk-dim">
                  mean {formatMathValue(result.mean)}
                </span>
                <span className="num text-chalk-dim">
                  median {formatQuantile(result, result.median)}
                </span>
                <span className="num text-chalk-dim">
                  p25–p75 {formatQuantile(result, result.p25)}–{formatQuantile(result, result.p75)}
                </span>
              </div>
            ) : null}
          </div>
        )
      ) : null}
    </div>
  )
}

export default ShowMathDrawer
