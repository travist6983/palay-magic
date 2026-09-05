/**
 * Player lookup. Debounced ~200ms against /api/search (which requires >= 2 characters), keyboard
 * driven, and it navigates straight to the deep dive on select.
 */

import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api'

/** Derived from the api helper so the shape can never drift from the fetcher. */
export type SearchHit = Awaited<ReturnType<typeof api.search>>[number]

export interface PlayerSearchProps {
  className?: string
  placeholder?: string
  /** Max rows in the dropdown. */
  limit?: number
}

const MIN_QUERY = 2

export function PlayerSearch({
  className = '',
  placeholder = 'Search players',
  limit = 8,
}: PlayerSearchProps) {
  const navigate = useNavigate()
  const [text, setText] = useState('')
  const [debounced, setDebounced] = useState('')
  const [open, setOpen] = useState(false)
  const [cursor, setCursor] = useState(0)
  const boxRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    const id = window.setTimeout(() => setDebounced(text.trim()), 200)
    return () => window.clearTimeout(id)
  }, [text])

  const query = debounced.length >= MIN_QUERY ? debounced : ''
  const { data, isFetching, isError, fetchStatus } = useQuery({
    queryKey: ['search', query],
    queryFn: () => api.search(query),
    enabled: query.length >= MIN_QUERY,
    staleTime: 60_000,
  })

  const hits = useMemo(() => (data ?? []).slice(0, limit), [data, limit])

  useEffect(() => {
    setCursor(0)
  }, [query])

  // Close on click outside.
  useEffect(() => {
    if (!open) return
    function onDown(e: MouseEvent) {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  // "/" focuses the box from anywhere that is not already a text field.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return
      const el = e.target as HTMLElement | null
      const tag = el?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el?.isContentEditable) return
      e.preventDefault()
      inputRef.current?.focus()
      inputRef.current?.select()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [])

  function select(hit: SearchHit) {
    setOpen(false)
    setText('')
    setDebounced('')
    inputRef.current?.blur()
    navigate(`/player/${hit.gsis_id}`)
  }

  function onKeyDown(e: ReactKeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Escape') {
      setOpen(false)
      inputRef.current?.blur()
      return
    }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      if (hits.length === 0) return
      e.preventDefault()
      setOpen(true)
      setCursor((c) => {
        const next = e.key === 'ArrowDown' ? c + 1 : c - 1
        return (next + hits.length) % hits.length
      })
      return
    }
    if (e.key === 'Enter') {
      const hit = hits[cursor]
      if (hit) {
        e.preventDefault()
        select(hit)
      }
    }
  }

  const showPanel = open && query.length >= MIN_QUERY
  const tooShort = open && text.trim().length > 0 && text.trim().length < MIN_QUERY

  return (
    <div ref={boxRef} className={`relative ${className}`}>
      <input
        ref={inputRef}
        type="text"
        value={text}
        role="combobox"
        aria-expanded={showPanel}
        aria-controls="player-search-list"
        aria-autocomplete="list"
        autoComplete="off"
        spellCheck={false}
        placeholder={placeholder}
        onChange={(e) => {
          setText(e.target.value)
          setOpen(true)
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
        className="w-full rounded border border-ink-line bg-ink px-2 py-1 text-sm text-chalk placeholder:text-chalk-faint focus:border-accent focus:outline-none"
      />
      {text.length === 0 && (
        <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 rounded border border-ink-line px-1 text-[10px] leading-4 text-chalk-faint">
          /
        </span>
      )}

      {tooShort && (
        <div className="card absolute right-0 z-40 mt-1 w-full min-w-[18rem] px-3 py-2 text-xs text-chalk-faint">
          Type at least {MIN_QUERY} characters.
        </div>
      )}

      {showPanel && (
        <div
          id="player-search-list"
          role="listbox"
          className="card absolute right-0 z-40 mt-1 max-h-80 w-full min-w-[20rem] overflow-y-auto py-1"
        >
          {isError ? (
            <div className="px-3 py-2 text-xs text-bad">Search unavailable.</div>
          ) : fetchStatus === 'paused' ? (
            <div className="px-3 py-2 text-xs text-warn">Offline — search is unavailable.</div>
          ) : data === undefined ? (
            <div className="px-3 py-2 text-xs text-chalk-faint">Searching…</div>
          ) : hits.length === 0 ? (
            <div className="px-3 py-2 text-xs text-chalk-faint">
              No players match "{debounced}".
            </div>
          ) : (
            hits.map((hit, i) => (
              <button
                key={hit.gsis_id}
                type="button"
                role="option"
                aria-selected={i === cursor}
                onMouseEnter={() => setCursor(i)}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => select(hit)}
                className={`flex w-full items-baseline gap-2 px-3 py-1.5 text-left text-sm ${
                  i === cursor ? 'bg-ink-line text-chalk' : 'text-chalk-dim'
                }`}
              >
                <span className="flex-1 truncate">{hit.display_name}</span>
                <span className="w-8 shrink-0 text-right text-[11px] uppercase tracking-wide text-chalk-faint">
                  {hit.position}
                </span>
                <span className="num w-9 shrink-0 text-right text-[11px] text-chalk-faint">
                  {hit.team ?? '—'}
                </span>
              </button>
            ))
          )}
          {isFetching && hits.length > 0 && (
            <div className="px-3 pt-1 text-[10px] text-chalk-faint">updating…</div>
          )}
        </div>
      )}
    </div>
  )
}

export default PlayerSearch
