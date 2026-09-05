/** Routes. Everything renders inside Layout, and a page that throws does not take the shell down. */

import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { Link, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { Layout } from './components/Layout'
import BoardPage from './pages/BoardPage'
import ModelHealthPage from './pages/ModelHealthPage'
import PlayerPage from './pages/PlayerPage'

interface ErrorBoundaryProps {
  children: ReactNode
  /** Changing this resets the boundary, so navigating away from a broken page recovers. */
  resetKey: string
}

interface ErrorBoundaryState {
  error: Error | null
}

class RouteErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error }
  }

  componentDidUpdate(prev: ErrorBoundaryProps) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null })
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('PropLab route error', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="card p-4">
          <h1 className="text-sm font-semibold text-bad">This view failed to render.</h1>
          <p className="mt-1 text-xs text-chalk-dim">
            The rest of the app still works — pick another board above.
          </p>
          <pre className="mt-3 overflow-x-auto rounded border border-ink-line bg-ink p-2 text-[11px] text-chalk-faint">
            {this.state.error.message}
          </pre>
        </div>
      )
    }
    return this.props.children
  }
}

function NotFound() {
  return (
    <div className="card p-4">
      <h1 className="text-sm font-semibold text-chalk">No such page.</h1>
      <p className="mt-1 text-xs text-chalk-dim">
        Try the{' '}
        <Link to="/board/QB" className="text-accent hover:underline">
          QB board
        </Link>
        .
      </p>
    </div>
  )
}

export function App() {
  const location = useLocation()
  return (
    <Layout>
      <RouteErrorBoundary resetKey={location.pathname}>
        <Routes>
          <Route path="/" element={<Navigate to="/board/QB" replace />} />
          <Route path="/board/:position" element={<BoardPage />} />
          <Route path="/player/:gsisId" element={<PlayerPage />} />
          <Route path="/model" element={<ModelHealthPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </RouteErrorBoundary>
    </Layout>
  )
}

export default App
