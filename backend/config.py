"""Configuration for PropLab.

Everything is read from the environment (or ``.env``) with defaults that work out of the box.
Secrets are optional by design: the app must still load and render stale data when a key or a
source is missing (see docs/DECISIONS.md D8).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings. Env vars are prefixed ``PROPLAB_`` except the two API keys."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="PROPLAB_",
        extra="ignore",
    )

    # --- paths -------------------------------------------------------------
    db_path: Path = Field(default=REPO_ROOT / "data" / "proplab.duckdb")

    serve_db_path: Path = Field(default=REPO_ROOT / "data" / "proplab-serve.duckdb")
    """Read-only snapshot the API serves from.

    DuckDB locks the database file per process, and a read-only reader still blocks a writer, so an
    API process holding the main file would make `make refresh` fail outright -- which is exactly
    what `make dev` does. The refresh publishes a copy when it finishes and the API reads that, so
    the two never contend and the UI always sees a consistent snapshot rather than a half-written
    week.
    """
    raw_dir: Path = Field(default=REPO_ROOT / "data" / "raw")
    cache_dir: Path = Field(default=REPO_ROOT / "data" / "cache")

    # --- data scope --------------------------------------------------------
    seasons: list[int] = Field(default=[2023, 2024, 2025, 2026])

    # --- secrets (no PROPLAB_ prefix; read from the bare env var) ----------
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    odds_api_key: str | None = Field(default=None, alias="ODDS_API_KEY")

    # --- budgets (docs/DECISIONS.md D8) ------------------------------------
    odds_monthly_budget: int = 500
    llm_calls_per_refresh: int = 150
    sleeper_min_cache_age_hours: float = 20.0

    # --- model knobs (§5) --------------------------------------------------
    recency_window: int = 6
    """N games in the recency-weighted USAGE baseline (§5.1). Role changes fast."""

    efficiency_window: int = 17
    """N games for per-opportunity EFFICIENCY rates (yards per carry, catch rate, ...).

    §5.1 sets one window of 6 for everything. That is right for usage, where a role change is the
    signal, and wrong for efficiency, where it is mostly noise: Jahmyr Gibbs' last six games of
    2025 ran 3.09 yards per carry against a career mark near 5.0, and projecting 3.09 forward
    treats a bad stretch as a new true rate. Efficiency gets a longer window and is shrunk toward
    the positional rate on top (see `efficiency_prior_games`).
    """

    efficiency_prior_games: float = 3.0
    """Games of positional-average pseudo-data mixed into every efficiency rate.

    Expressed in GAMES rather than opportunities on purpose. The weighted opportunity total that
    a rate is computed over is much smaller than the raw count -- recency weights decay to well
    under 1 -- so a prior stated in raw opportunities silently dominated: 60 pseudo-targets against
    a running back's ~15 weighted targets gave the league average 80% of the weight and projected
    him for 3.7 yards a catch. Scaling the prior by the player's own per-game volume keeps its
    influence at the intended few games regardless of position.
    """

    recency_decay: float = 0.8
    """w_i = decay ** i, most recent game i=0 (§5.1)."""

    prior_season_discount: float = 0.85
    """Extra multiplicative discount on games from a prior season (§4)."""

    defense_window: int = 8
    """Trailing games used for position-specific defensive strength (§5.2)."""

    defense_shrink_k: float = 6.0
    """Games of league-average pseudo-data shrinking each multiplier toward 1.0 (§5.2)."""

    overdispersion_threshold: float = 1.3
    """variance/mean above which a count stat uses negative binomial, not Poisson (§5.6)."""

    longest_sims: int = 2000
    """Monte Carlo draws for `longest_*` stats (§5.6)."""

    min_games_for_history: int = 3
    """Below this, a player is flagged 'insufficient history' and projected from priors (§4)."""

    every_down_snap_threshold: float = 0.80
    """LB eligibility gate: share of defensive snaps (§4)."""

    # --- llm ---------------------------------------------------------------
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_max_tokens: int = 1024

    # --- api / logging -----------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"

    @field_validator("seasons", mode="before")
    @classmethod
    def _parse_seasons(cls, v: object) -> object:
        """Allow PROPLAB_SEASONS=2023,2024,2025,2026."""
        if isinstance(v, str):
            return [int(x) for x in v.replace(" ", "").split(",") if x]
        return v

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_odds(self) -> bool:
        return bool(self.odds_api_key)

    def ensure_dirs(self) -> None:
        """Create the data directories. Safe to call repeatedly."""
        for d in (self.db_path.parent, self.raw_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    s = Settings()
    s.ensure_dirs()
    return s
