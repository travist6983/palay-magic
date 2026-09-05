# PropLab — local NFL player-prop projection app.
# Everything runs locally: uv for Python, npm for the frontend, DuckDB for storage.

PY := .venv/bin/python
UV := uv

.DEFAULT_GOAL := help
.PHONY: help setup backfill refresh rank project backtest dev api web test lint fmt counts clean-db nuke

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install Python and frontend dependencies
	$(UV) sync
	@if [ -f frontend/package.json ]; then cd frontend && npm install; else echo "frontend not scaffolded yet"; fi

backfill: ## One-time historical ingest (nflverse 2023-2026 + Sleeper crosswalk)
	$(UV) run proplab backfill

refresh: ## Weekly: pull new data, recompute defenses, rankings, projections
	$(UV) run proplab refresh

rank: ## Print the top-10 board for every position
	$(UV) run proplab rank

project: ## Project one player: make project PLAYER="Player Name"
	$(UV) run proplab project --player "$(PLAYER)"

backtest: ## Score last season's projections against actuals
	$(UV) run proplab backtest --season 2025 --weeks 5-18

counts: ## SELECT count(*) from every table and view
	$(UV) run proplab counts

api: ## Run the FastAPI backend alone (http://localhost:8000)
	$(UV) run uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

web: ## Run the Vite frontend alone (http://localhost:5173)
	cd frontend && npm run dev

dev: ## Run backend + frontend together (http://localhost:5173)
	@echo "backend -> http://127.0.0.1:8000   frontend -> http://localhost:5173"
	@trap 'kill 0' EXIT INT TERM; \
	$(UV) run uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000 & \
	(cd frontend && npm run dev) & \
	wait

test: ## pytest (model math) + vitest (frontend utils)
	$(UV) run pytest -q
	@if [ -f frontend/package.json ]; then cd frontend && npm run test -- --run; fi

lint: ## ruff check
	$(UV) run ruff check backend

fmt: ## ruff format + fix
	$(UV) run ruff format backend
	$(UV) run ruff check --fix backend

clean-db: ## Delete the DuckDB file (keeps the Parquet cache, so no re-download)
	rm -f data/proplab.duckdb data/proplab.duckdb.wal

nuke: ## Delete the DuckDB file AND the raw Parquet cache (forces a full re-download)
	rm -rf data/proplab.duckdb data/proplab.duckdb.wal data/raw data/cache
