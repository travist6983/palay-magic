/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: { DEFAULT: '#0f1115', soft: '#161a21', line: '#232936' },
        chalk: { DEFAULT: '#e8eaed', dim: '#9aa3b2', faint: '#6b7280' },
        good: '#34d399',
        warn: '#fbbf24',
        bad: '#f87171',
        accent: '#60a5fa',
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
}
