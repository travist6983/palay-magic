/** Typed fetch helpers. The backend is proxied at /api by Vite, so everything is origin-relative. */

import type { Board, Calibration, DefenseTable, Meta, PlayerDetail } from './types'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
  }
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { Accept: 'application/json' } })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new ApiError(detail || res.statusText, res.status)
  }
  return (await res.json()) as T
}

export const api = {
  meta: () => get<Meta>('/api/meta'),
  boards: () => get<Board[]>('/api/boards'),
  board: (position: string) => get<Board>(`/api/board/${position}`),
  player: (gsisId: string) => get<PlayerDetail>(`/api/player/${gsisId}`),
  defense: (position: string, metric?: string) =>
    get<DefenseTable>(`/api/defense/${position}${metric ? `?metric=${metric}` : ''}`),
  search: (q: string) =>
    get<{ gsis_id: string; display_name: string; position: string; team: string | null }[]>(
      `/api/search?q=${encodeURIComponent(q)}`,
    ),
  calibration: () => get<Calibration | null>('/api/calibration'),
  games: () => get<Record<string, unknown>[]>('/api/games'),
}

/** Everything is precomputed on refresh, so cached data stays valid until the next one. */
export const queryDefaults = {
  staleTime: 5 * 60 * 1000,
  gcTime: 30 * 60 * 1000,
  retry: 1,
  refetchOnWindowFocus: false,
}
