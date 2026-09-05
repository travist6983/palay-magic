/**
 * The betting line for one stat.
 *
 * Deliberately a text input, not `type="number"`: the value has to survive the half-typed states
 * ("", "1", "1.") that a user passes through, because P(over) is recomputed on every keystroke
 * from the distribution parameters already in the browser (D10, §8). Nothing here fetches.
 */

import { useCallback, type KeyboardEvent } from 'react'

export interface LineInputProps {
  /** Raw text, so intermediate keystrokes render exactly as typed. */
  value: string
  onChange: (next: string) => void
  /** Screen-reader label; the visible header is the column, not the field. */
  label: string
  /** Nudge size for the steppers and the arrow keys. */
  step?: number
  /** False once the user has moved off the projected median. */
  isDefault?: boolean
  onReset?: () => void
}

/** Accepts anything on the way to a number, including the empty and trailing-dot states. */
const PARTIAL_NUMBER = /^-?\d*\.?\d*$/

export function LineInput({
  value,
  onChange,
  label,
  step = 0.5,
  isDefault = true,
  onReset,
}: LineInputProps) {
  const nudge = useCallback(
    (delta: number) => {
      const current = Number.parseFloat(value)
      const base = Number.isFinite(current) ? current : 0
      const next = Math.round((base + delta) * 100) / 100
      onChange(String(next))
    },
    [onChange, value],
  )

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'ArrowUp') {
        event.preventDefault()
        nudge(event.shiftKey ? step * 10 : step)
      } else if (event.key === 'ArrowDown') {
        event.preventDefault()
        nudge(event.shiftKey ? -step * 10 : -step)
      }
    },
    [nudge, step],
  )

  const valid = value.trim() !== '' && Number.isFinite(Number.parseFloat(value))

  return (
    <span className="inline-flex items-center gap-1">
      <span className="inline-flex items-stretch overflow-hidden rounded border border-ink-line focus-within:border-accent">
        <button
          type="button"
          tabIndex={-1}
          aria-label={`decrease ${label}`}
          className="px-1.5 text-chalk-faint hover:bg-ink-line hover:text-chalk"
          onClick={() => nudge(-step)}
        >
          −
        </button>
        <input
          value={value}
          aria-label={label}
          inputMode="decimal"
          autoComplete="off"
          spellCheck={false}
          onKeyDown={handleKeyDown}
          onChange={(event) => {
            const next = event.target.value
            if (PARTIAL_NUMBER.test(next)) onChange(next)
          }}
          onFocus={(event) => event.target.select()}
          className={`num w-14 border-x border-ink-line bg-ink px-1 py-0.5 text-right text-sm outline-none ${
            valid ? 'text-chalk' : 'text-bad'
          }`}
        />
        <button
          type="button"
          tabIndex={-1}
          aria-label={`increase ${label}`}
          className="px-1.5 text-chalk-faint hover:bg-ink-line hover:text-chalk"
          onClick={() => nudge(step)}
        >
          +
        </button>
      </span>
      {!isDefault && onReset ? (
        <button
          type="button"
          onClick={onReset}
          title="back to the projected median"
          className="text-[10px] uppercase tracking-wide text-chalk-faint hover:text-accent"
        >
          reset
        </button>
      ) : null}
    </span>
  )
}

export default LineInput
