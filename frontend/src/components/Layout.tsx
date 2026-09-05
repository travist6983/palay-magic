/** Page frame: header, a max-width content column, and the quiet legal footer. */

import type { ReactNode } from 'react'
import { Header } from './Header'

export interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="flex min-h-screen flex-col bg-ink text-chalk">
      <Header />
      <main className="mx-auto w-full max-w-[1680px] flex-1 px-4 py-4">{children}</main>
      <footer className="border-t border-ink-line">
        <div className="mx-auto flex w-full max-w-[1680px] flex-wrap items-baseline justify-between gap-x-4 gap-y-1 px-4 py-3 text-[11px] leading-4 text-chalk-faint">
          <span>
            PropLab — projections are model estimates from public data, not betting advice. Check
            settlement rules at your book before wagering.
          </span>
          <span>Bet only what you can afford to lose. 1-800-GAMBLER.</span>
        </div>
      </footer>
    </div>
  )
}

export default Layout
