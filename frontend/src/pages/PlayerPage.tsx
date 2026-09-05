/**
 * The deep dive (§8).
 *
 * One fetch of /api/player/{gsis_id} supplies everything on this page: the projections as
 * distribution PARAMETERS, the six-game log, the opponent adjustments, and the full arithmetic
 * trace. The line boxes then recompute P(over) locally on every keystroke — no request per
 * keystroke, by design (D10).
 */

import { useQuery } from '@tanstack/react-query'
import { useCallback, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { useParams } from 'react-router-dom'
import { DistributionChart } from '../components/DistributionChart'
import { EnvironmentCard } from '../components/EnvironmentCard'
import { GameLogTable } from '../components/GameLogTable'
import { MatchupBars } from '../components/MatchupBars'
import { ShowMathDrawer } from '../components/ShowMathDrawer'
import {
  ProjectionTable,
  activeDistribution,
  hasMeaningfulUnconditional,
  type ProjectionMode,
} from '../components/ProjectionTable'
import { ApiError, api } from '../lib/api'
import { defaultLine } from '../lib/distributions'
import { pct, weekLabel } from '../lib/format'
import type { PlayerDetail, Projection } from '../lib/types'

export interface PlayerPageProps {
  /** Overrides the route param. Handy for embedding the deep dive next to a board. */
  gsisId?: string
}

export function PlayerPage({ gsisId }: PlayerPageProps = {}) {
  const params = useParams()
  const id = gsisId ?? params.gsisId ?? params.id ?? params.playerId ?? ''

  const query = useQuery({
    queryKey: ['player', id],
    queryFn: () => api.player(id),
    enabled: id !== '',
  })

  const [trackedId, setTrackedId] = useState(id)
  const [lines, setLines] = useState<Record<string, string>>({})
  const [selected, setSelected] = useState<string | null>(null)
  const [mode, setMode] = useState<ProjectionMode>('conditional')
  const [openMath, setOpenMath] = useState<Record<string, boolean>>({})

  // A different player is a different set of lines. Resetting during render (rather than in an
  // effect) means the first paint of the new player is never the old player's state.
  if (trackedId !== id) {
    setTrackedId(id)
    setLines({})
    setSelected(null)
    setMode('conditional')
    setOpenMath({})
  }

  const handleLineChange = useCallback((key: string, value: string) => {
    setLines((prev) => ({ ...prev, [key]: value }))
  }, [])

  const handleLineReset = useCallback((key: string) => {
    setLines((prev) => {
      const next = { ...prev }
      delete next[key]
      return next
    })
  }, [])

  const handleShowMath = useCallback((key: string) => {
    setOpenMath((prev) => ({ ...prev, [key]: true }))
    const node = document.getElementById(`math-${key}`)
    if (node) node.scrollIntoView({ block: 'center' })
  }, [])

  const detail = query.data

  const projections = useMemo<Projection[]>(() => detail?.projections ?? [], [detail])
  const canToggleMode = useMemo(() => hasMeaningfulUnconditional(projections), [projections])
  const labels = useMemo(() => {
    const map: Record<string, string> = {}
    for (const p of projections) map[p.stat] = p.label
    return map
  }, [projections])

  const orderedStats = useMemo(() => {
    if (!detail) return []
    const known = new Set(projections.map((p) => p.stat))
    const ordered = detail.stat_order.filter((key) => known.has(key))
    for (const p of projections) if (!ordered.includes(p.stat)) ordered.push(p.stat)
    return ordered
  }, [detail, projections])

  const selectedStat = selected && orderedStats.includes(selected) ? selected : orderedStats[0] ?? ''
  const selectedProjection = projections.find((p) => p.stat === selectedStat)

  if (id === '') {
    return <Notice tone="bad" title="No player" body="This route needs a gsis_id." />
  }

  if (!detail) {
    if (query.isLoading || query.isFetching) return <PlayerSkeleton />
    return (
      <Notice
        tone="bad"
        title="Could not load this player"
        body={describeError(query.error)}
        action={
          <button
            type="button"
            className="rounded border border-ink-line px-2 py-1 text-xs text-chalk-dim hover:text-chalk"
            onClick={() => void query.refetch()}
          >
            retry
          </button>
        }
      />
    )
  }

  const selectedDist = selectedProjection ? activeDistribution(selectedProjection, mode) : null
  const selectedLineText = selectedProjection
    ? lines[selectedProjection.stat] ?? String(defaultLine(selectedProjection.distribution))
    : ''
  const parsedLine = Number.parseFloat(selectedLineText)
  const selectedLine =
    selectedLineText.trim() !== '' && Number.isFinite(parsedLine) ? parsedLine : null

  return (
    <div className="flex flex-col gap-3">
      {query.isError ? (
        <div className="rounded border border-bad/40 bg-bad/10 px-3 py-2 text-[11px] text-bad">
          Refresh failed ({describeError(query.error)}). Showing the last data this tab loaded.
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <PlayerHeader detail={detail} className="xl:col-span-2" />
        <EnvironmentCard
          environment={detail.environment}
          team={detail.team}
          opponent={detail.opponent}
        />
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <ProjectionTable
            projections={projections}
            statOrder={detail.stat_order}
            lines={lines}
            onLineChange={handleLineChange}
            onLineReset={handleLineReset}
            selectedStat={selectedStat}
            onSelectStat={setSelected}
            mode={mode}
            onModeChange={setMode}
            canToggleMode={canToggleMode}
            onShowMath={handleShowMath}
          />
        </div>
        <div className="xl:sticky xl:top-[76px] xl:self-start">
          {selectedProjection && selectedDist ? (
            <DistributionChart
              label={selectedProjection.label}
              dist={selectedDist}
              line={selectedLine}
              modeNote={
                mode === 'unconditional'
                  ? `includes P(does not play) = ${pct(1 - selectedProjection.play_probability, 0)}`
                  : canToggleMode
                    ? 'if he plays'
                    : undefined
              }
              note={selectedProjection.settlement_note || undefined}
            />
          ) : (
            <div className="card p-3 text-sm text-chalk-dim">Select a stat to see its shape.</div>
          )}
        </div>
      </div>

      <GameLogTable
        rows={detail.game_log}
        statOrder={orderedStats}
        labels={labels}
        selectedStat={selectedStat}
        onSelectStat={setSelected}
      />

      <MatchupBars matchup={detail.matchup} opponent={detail.opponent} />

      <section className="flex flex-col gap-2">
        <div className="flex items-baseline justify-between gap-2">
          <h2 className="text-sm font-semibold text-chalk">Show the math</h2>
          <p className="text-[11px] text-chalk-faint">
            Every intermediate value that produced the numbers above, in the order it was computed.
          </p>
        </div>
        {orderedStats.map((key) => {
          const projection = projections.find((p) => p.stat === key)
          return (
            <ShowMathDrawer
              key={key}
              stat={key}
              label={labels[key] ?? key.replace(/_/g, ' ')}
              steps={detail.math[key] ?? []}
              open={openMath[key] ?? key === selectedStat}
              onToggle={(target) =>
                setOpenMath((prev) => ({
                  ...prev,
                  [target]: !(prev[target] ?? target === selectedStat),
                }))
              }
              result={projection ? activeDistribution(projection, mode) : undefined}
            />
          )
        })}
      </section>
    </div>
  )
}

// --- header ---------------------------------------------------------------

function PlayerHeader({ detail, className }: { detail: PlayerDetail; className?: string }) {
  const injury = detail.injury
  const flags = detail.flags as Record<string, unknown>
  const passRateShift = typeof flags.pass_rate_shift === 'number' ? flags.pass_rate_shift : 0

  const chips: { text: string; tone: string; title: string }[] = []
  if (flags.changed_team === true)
    chips.push({
      text: 'changed team',
      tone: 'bg-warn/15 text-warn',
      title: 'Every game in the log was played for a different team. Usage carries over far less than efficiency does.',
    })
  if (flags.changed_coach === true)
    chips.push({
      text: 'changed coach',
      tone: 'bg-warn/15 text-warn',
      title: 'New play-caller. Pace and pass rate are the first things to move.',
    })
  if (flags.insufficient_history === true)
    chips.push({
      text: 'thin history',
      tone: 'bg-bad/15 text-bad',
      title: 'Too few scored games to fit this player individually; the projection leans on position priors.',
    })
  if (flags.all_history_prior_season === true)
    chips.push({
      text: 'prior season only',
      tone: 'bg-ink-line text-chalk-dim',
      title: 'No games have been played this season, so every input is prior-season and carries the 0.85 discount.',
    })
  if (Math.abs(passRateShift) > 0.005)
    chips.push({
      text: `pass rate ${passRateShift > 0 ? '+' : ''}${(passRateShift * 100).toFixed(1)}%`,
      tone: passRateShift > 0 ? 'bg-good/15 text-good' : 'bg-bad/15 text-bad',
      title: 'Change in the team pass rate versus the games in this log.',
    })

  const injuryTone =
    injury.p_played >= 0.95
      ? 'text-chalk-dim'
      : injury.p_played >= 0.7
        ? 'text-warn'
        : 'text-bad'

  return (
    <section className={`card flex flex-col ${className ?? ''}`}>
      <div className="flex items-start gap-3 p-3">
        <Headshot url={detail.headshot_url} name={detail.display_name} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <h1 className="text-xl font-semibold leading-tight text-chalk">
              {detail.display_name}
            </h1>
            <span className="chip bg-ink-line text-chalk-dim">{detail.position}</span>
            <span className="num text-sm text-chalk-dim">
              {detail.team ?? '—'}{' '}
              <span className="text-chalk-faint">
                {detail.environment.is_home === false ? 'at' : 'vs'}
              </span>{' '}
              {detail.opponent ?? '—'}
            </span>
            <span className="num text-[11px] text-chalk-faint">
              {weekLabel(detail.season, detail.week)}
            </span>
          </div>

          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-chalk-faint">
            <span className="num">{formatHeight(detail.height)}</span>
            <span className="num">{detail.weight === null ? '—' : `${detail.weight} lb`}</span>
            <span className="num">
              {detail.years_exp === null
                ? '—'
                : detail.years_exp === 0
                  ? 'rookie'
                  : `${detail.years_exp} yr exp`}
            </span>
            <span className="num text-chalk-faint">{detail.gsis_id}</span>
          </div>

          {chips.length ? (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {chips.map((chip) => (
                <span key={chip.text} className={`chip ${chip.tone}`} title={chip.title}>
                  {chip.text}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-ink-line px-3 py-2 text-[11px]">
        <span className={injuryTone}>
          <span className="uppercase tracking-wide text-chalk-faint">status </span>
          {injury.report_status ?? 'no designation'}
          {injury.practice_status ? ` · ${injury.practice_status}` : ''}
        </span>
        <span
          className={`num ${injuryTone}`}
          title={`Play probability for a ${injury.role} with this designation and practice status (D11). Source: ${injury.source}${
            injury.n_observations ? `, n=${injury.n_observations}` : ''
          }.`}
        >
          <span className="uppercase tracking-wide text-chalk-faint">P(plays) </span>
          {pct(injury.p_played, 1)}
        </span>
        <span
          className="num text-chalk-dim"
          title="Expected share of snaps given he plays. A banged-up starter who suits up is not a full-snap player."
        >
          <span className="uppercase tracking-wide text-chalk-faint">snap share </span>
          {pct(injury.expected_snap_share, 0)}
        </span>
        <span className="text-chalk-faint">{injury.role}</span>
        {injury.usage_risk ? <span className="text-warn">{injury.usage_risk}</span> : null}
        {injury.teammates_affected.length ? (
          <span className="text-chalk-dim" title="Their absence changes this player's usage.">
            watching {injury.teammates_affected.join(', ')}
          </span>
        ) : null}
      </div>

      {injury.note ? (
        <p className="border-t border-ink-line px-3 py-2 text-[11px] text-warn">{injury.note}</p>
      ) : null}

      {detail.narrative ? (
        <p className="border-t border-ink-line px-3 py-2 text-xs leading-relaxed text-chalk-dim">
          {detail.narrative}
        </p>
      ) : null}
    </section>
  )
}

function Headshot({ url, name }: { url: string | null; name: string }) {
  const [broken, setBroken] = useState(false)
  const initials = name
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part[0] ?? '')
    .join('')
    .toUpperCase()

  if (!url || broken) {
    return (
      <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded border border-ink-line bg-ink text-sm font-semibold text-chalk-faint">
        {initials || '—'}
      </div>
    )
  }
  return (
    <img
      src={url}
      alt=""
      width={64}
      height={64}
      loading="lazy"
      onError={() => setBroken(true)}
      className="h-16 w-16 shrink-0 rounded border border-ink-line bg-ink object-cover"
    />
  )
}

function formatHeight(inches: number | null): string {
  if (inches === null || !Number.isFinite(inches)) return '—'
  const feet = Math.floor(inches / 12)
  return `${feet}'${Math.round(inches - feet * 12)}"`
}

// --- states ---------------------------------------------------------------

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.status === 404 ? 'no projectable player at this id (404)' : `${error.message} (${error.status})`
  }
  if (error instanceof Error) return error.message
  return 'the API did not respond'
}

function Notice({
  tone,
  title,
  body,
  action,
}: {
  tone: 'bad' | 'warn'
  title: string
  body: string
  action?: ReactNode
}) {
  return (
    <div>
      <div
        className={`card flex items-center gap-3 p-3 ${
          tone === 'bad' ? 'border-bad/40' : 'border-warn/40'
        }`}
      >
        <div className="flex-1">
          <div className={`text-sm font-semibold ${tone === 'bad' ? 'text-bad' : 'text-warn'}`}>
            {title}
          </div>
          <div className="text-[11px] text-chalk-dim">{body}</div>
        </div>
        {action}
      </div>
    </div>
  )
}

function PlayerSkeleton() {
  return (
    <div className="flex animate-pulse flex-col gap-3">
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <div className="card h-32 xl:col-span-2" />
        <div className="card h-32" />
      </div>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <div className="card h-80 xl:col-span-2" />
        <div className="card h-80" />
      </div>
      <div className="card h-48" />
      <span className="sr-only">Loading player</span>
    </div>
  )
}

export default PlayerPage
