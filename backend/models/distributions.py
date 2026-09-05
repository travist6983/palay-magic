"""Distribution fitting and evaluation (§5.6).

Posted prop lines sit at the **median** of a skewed distribution, not the mean
(``how_books_build_lines.md`` step 3), so every function here reports quantiles rather than a
point estimate. Each fit returns a :class:`Distribution` carrying the family and its parameters;
those parameters are what get stored and shipped to the browser so ``P(over line)`` can be
recomputed locally on every keystroke (D10).

Every function is pure: plain scalars and arrays in, plain values out. No database, no globals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from backend.models.stats import Family

# Guard rails. A degenerate fit is a bug elsewhere; clamping keeps it from becoming a NaN
# that silently poisons a whole projection table.
_MIN_MEAN = 1e-6
_MIN_DISPERSION = 1e-3
_MAX_DISPERSION = 1e4


@dataclass(frozen=True)
class Distribution:
    """A fitted predictive distribution for one player-stat-week.

    Attributes:
        family: which distribution family (§5.6).
        params: family-specific parameters. This is the JSON blob stored in
            ``projections.params`` and re-evaluated in the browser (D10).
        mean: expected value.
        integer_valued: whether the support is the non-negative integers. Determines whether
            ``P(X > line)`` uses the survival function at ``floor(line)``.
    """

    family: Family
    params: dict[str, Any]
    mean: float
    integer_valued: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- quantiles ---------------------------------------------------------

    def quantile(self, q: float) -> float:
        """The value at cumulative probability ``q``."""
        return _quantile(self, q)

    @property
    def median(self) -> float:
        """The 50th percentile. This is where a book would post the line."""
        return self.quantile(0.5)

    @property
    def p25(self) -> float:
        return self.quantile(0.25)

    @property
    def p75(self) -> float:
        return self.quantile(0.75)

    def summary(self) -> dict[str, float]:
        """median / p25 / p75 / mean, the four numbers the UI shows (§8)."""
        return {
            "mean": self.mean,
            "median": self.median,
            "p25": self.p25,
            "p75": self.p75,
        }

    # -- probabilities -----------------------------------------------------

    def prob_over(self, line: float) -> float:
        """``P(X > line)``.

        Prop lines carry a half-point precisely so a push is impossible, so strict ``>`` and
        ``>=`` agree at any real line. For an integer-valued stat at a whole-number line we still
        use strict ``>``: an exact match is a push, not a win.
        """
        return _prob_over(self, line)

    def prob_under(self, line: float) -> float:
        """``P(X < line)``. Equals ``1 - prob_over`` only when a push has zero probability."""
        return 1.0 - self.prob_over(line) - self.prob_exact(line)

    def prob_exact(self, line: float) -> float:
        """``P(X == line)``. Non-zero only for an integer-valued stat at a whole number."""
        if not self.integer_valued or line != int(line):
            return 0.0
        return _pmf(self, int(line))

    def to_json(self) -> dict[str, Any]:
        """The wire format: everything the browser needs to recompute P(over) itself."""
        return {
            "family": str(self.family),
            "params": self.params,
            "mean": self.mean,
            "integer_valued": self.integer_valued,
            "median": self.median,
            "p25": self.p25,
            "p75": self.p75,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_negative_binomial(mean: float, variance: float, min_dispersion: float = 0.05) -> Distribution:
    """Fit a negative binomial by moment matching (§5.6).

    Uses the ``(r, p)`` parameterisation where ``r`` is the number of successes and ``p`` the
    success probability:

        mean = r (1 - p) / p
        var  = r (1 - p) / p^2  =  mean / p

    so ``p = mean / var`` and ``r = mean^2 / (var - mean)``. The dispersion reported in ``params``
    is ``alpha = 1 / r``, the standard over-dispersion parameter: ``var = mean + alpha * mean^2``.

    A negative binomial requires ``var > mean``. Under-dispersed input is a real signal (small
    samples of a stable stat), not an error, so we floor the variance at
    ``mean * (1 + min_dispersion)`` and record a note rather than silently returning a Poisson.

    Args:
        mean: expected value, from usage x efficiency (§5.4, §5.5).
        variance: recency-weighted variance (§5.1), or a position-level fallback.
        min_dispersion: smallest allowed ``alpha``; keeps ``r`` finite.
    """
    mean = max(float(mean), _MIN_MEAN)
    notes: list[str] = []

    floor_var = mean * (1.0 + min_dispersion)
    if not math.isfinite(variance) or variance <= floor_var:
        notes.append(f"variance {variance:.3f} floored to {floor_var:.3f} (under-dispersed input)")
        variance = floor_var

    alpha = (variance - mean) / (mean * mean)
    alpha = min(max(alpha, _MIN_DISPERSION), _MAX_DISPERSION)

    r = 1.0 / alpha
    p = r / (r + mean)

    return Distribution(
        family=Family.NEGATIVE_BINOMIAL,
        params={"r": r, "p": p, "alpha": alpha, "mean": mean, "variance": variance},
        mean=mean,
        integer_valued=True,
        notes=tuple(notes),
    )


def fit_poisson(lam: float) -> Distribution:
    """Fit a Poisson with rate ``lam`` (§5.6). Variance equals the mean by construction."""
    lam = max(float(lam), _MIN_MEAN)
    return Distribution(
        family=Family.POISSON,
        params={"lam": lam, "mean": lam, "variance": lam},
        mean=lam,
        integer_valued=True,
    )


def fit_count(
    mean: float,
    variance: float,
    overdispersion_threshold: float = 1.3,
) -> Distribution:
    """Poisson, or negative binomial when the stat is over-dispersed (§5.6).

    The test is ``variance / mean > threshold``. Below it the extra parameter buys nothing and a
    Poisson is the more stable fit on a six-game sample.
    """
    mean = max(float(mean), _MIN_MEAN)
    if not math.isfinite(variance) or variance <= 0:
        return fit_poisson(mean)

    ratio = variance / mean
    if ratio > overdispersion_threshold:
        dist = fit_negative_binomial(mean, variance)
        return Distribution(
            family=dist.family,
            params={**dist.params, "variance_mean_ratio": ratio},
            mean=dist.mean,
            integer_valued=True,
            notes=(*dist.notes, f"over-dispersed (var/mean = {ratio:.2f})"),
        )

    dist = fit_poisson(mean)
    return Distribution(
        family=dist.family,
        params={**dist.params, "variance_mean_ratio": ratio},
        mean=dist.mean,
        integer_valued=True,
        notes=(f"not over-dispersed (var/mean = {ratio:.2f})",),
    )


def fit_anytime_td(lam: float) -> Distribution:
    """Anytime-TD probability from a Poisson scoring rate (§5.6).

        P(at least one TD) = 1 - e^(-lambda)

    ``lambda`` is ``expected_team_TDs x player_TD_share``, where the share comes from red-zone
    target share and goal-line carry share -- not total volume (§5.6, and the 86.6% of rushing
    TDs that come from inside the red zone).
    """
    lam = max(float(lam), 0.0)
    p = 1.0 - math.exp(-lam)
    return Distribution(
        family=Family.BERNOULLI,
        params={"p": p, "lam": lam},
        mean=p,
        integer_valued=True,
    )


def fit_longest(
    n_plays: float,
    explosive_rate: float,
    yards_mean: float,
    yards_scale: float,
    n_sims: int = 2000,
    seed: int = 0,
) -> Distribution:
    """Distribution of the longest single play, by Monte Carlo (§5.6).

    Simulates ``n_plays`` opportunities per trial. Each play gains yards drawn from an exponential
    tail scaled to the player's per-play average, and the trial's outcome is the maximum.

    The point mass at zero is the settlement rule, not a modelling convenience: a longest-X prop
    **settles Under when the player records no such play** (D9). So a trial that draws zero plays
    contributes an outcome of 0, and ``p_zero`` is reported in the params.

    Args:
        n_plays: expected opportunities (targets, carries, or attempts).
        explosive_rate: share of plays that go 15+ yards. Widens the tail.
        yards_mean: mean yards per play for this player.
        yards_scale: scale of the exponential tail; larger means a longer tail.
        n_sims: Monte Carlo draws. 2000 is plenty for quartiles (§5.6).
        seed: fixed so a projection is reproducible ("Show math", §1 point 4).
    """
    rng = np.random.default_rng(seed)
    n_plays = max(float(n_plays), 0.0)
    yards_mean = max(float(yards_mean), 0.1)
    yards_scale = max(float(yards_scale), 0.1)

    counts = rng.poisson(n_plays, size=n_sims)
    maxima = np.zeros(n_sims, dtype=float)

    # One flat draw for every simulated play across all trials, then segment by trial.
    total = int(counts.sum())
    if total > 0:
        # Mixture: routine plays around the mean, explosive plays from a heavier tail.
        is_explosive = rng.random(total) < np.clip(explosive_rate, 0.0, 1.0)
        routine = rng.exponential(yards_mean, size=total)
        explosive = yards_mean + rng.exponential(yards_scale * 2.0, size=total)
        draws = np.where(is_explosive, explosive, routine)

        offsets = np.concatenate([[0], np.cumsum(counts)])
        for i in range(n_sims):
            lo, hi = offsets[i], offsets[i + 1]
            if hi > lo:
                maxima[i] = draws[lo:hi].max()

    p_zero = float((counts == 0).mean())
    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    quantiles = {str(q): float(np.quantile(maxima, q)) for q in qs}

    return Distribution(
        family=Family.EMPIRICAL_MAX,
        params={
            "quantiles": quantiles,
            "p_zero": p_zero,
            "n_plays": n_plays,
            "explosive_rate": float(explosive_rate),
            "yards_mean": yards_mean,
            "yards_scale": yards_scale,
            "n_sims": n_sims,
            "seed": seed,
            "samples_sorted": _thin(maxima, 200),
        },
        mean=float(maxima.mean()),
        integer_valued=False,
        notes=("settles Under if the player records no such play",),
    )


def fit_deterministic(terms: list[tuple[str, float, Distribution]]) -> Distribution:
    """A linear combination of other fitted stats, e.g. ``kicking_points = 3*FGM + 1*XPM`` (§5.6).

    The exact distribution of a weighted sum of independent counts has no closed form, so we
    convolve numerically over the component supports. That is cheap here because kicking counts
    are small, and it is exact rather than a normal approximation that would misprice the tails.

    Args:
        terms: ``(stat_key, coefficient, distribution)`` triples.
    """
    if not terms:
        return Distribution(Family.DETERMINISTIC, {"terms": [], "mean": 0.0}, 0.0)

    # Numerical convolution over a bounded support.
    support = np.array([0.0])
    probs = np.array([1.0])

    for _key, coef, dist in terms:
        max_k = max(4, int(dist.quantile(0.999)) + 2)
        ks = np.arange(0, max_k + 1)
        pmf = np.array([_pmf(dist, int(k)) for k in ks])
        total = pmf.sum()
        if total <= 0:
            continue
        pmf = pmf / total

        new_support = (support[:, None] + coef * ks[None, :]).ravel()
        new_probs = (probs[:, None] * pmf[None, :]).ravel()

        # Collapse duplicate values so the support does not blow up.
        order = np.argsort(new_support)
        new_support, new_probs = new_support[order], new_probs[order]
        uniq, idx = np.unique(np.round(new_support, 6), return_inverse=True)
        collapsed = np.zeros(len(uniq))
        np.add.at(collapsed, idx, new_probs)
        support, probs = uniq, collapsed

    mean = float((support * probs).sum())
    return Distribution(
        family=Family.DETERMINISTIC,
        params={
            "terms": [{"stat": k, "coefficient": c} for k, c, _ in terms],
            "support": support.tolist(),
            "probs": probs.tolist(),
            "mean": mean,
        },
        mean=mean,
        integer_valued=False,
    )


# ---------------------------------------------------------------------------
# Evaluation (shared by quantile / prob_over / pmf)
# ---------------------------------------------------------------------------


def _quantile(dist: Distribution, q: float) -> float:
    p = dist.params
    match dist.family:
        case Family.NEGATIVE_BINOMIAL:
            return float(stats.nbinom.ppf(q, p["r"], p["p"]))
        case Family.POISSON:
            return float(stats.poisson.ppf(q, p["lam"]))
        case Family.BERNOULLI:
            return 1.0 if q > (1.0 - p["p"]) else 0.0
        case Family.EMPIRICAL_MAX:
            return _interp_quantile(p["quantiles"], q)
        case Family.DETERMINISTIC:
            support = np.asarray(p["support"])
            cdf = np.cumsum(np.asarray(p["probs"]))
            idx = int(np.searchsorted(cdf, q, side="left"))
            return float(support[min(idx, len(support) - 1)])
    raise ValueError(f"unknown family {dist.family}")


def _prob_over(dist: Distribution, line: float) -> float:
    p = dist.params
    match dist.family:
        case Family.NEGATIVE_BINOMIAL:
            return float(stats.nbinom.sf(math.floor(line), p["r"], p["p"]))
        case Family.POISSON:
            return float(stats.poisson.sf(math.floor(line), p["lam"]))
        case Family.BERNOULLI:
            return p["p"] if line < 1 else 0.0
        case Family.EMPIRICAL_MAX:
            samples = p.get("samples_sorted")
            if samples:
                arr = np.asarray(samples)
                return float((arr > line).mean())
            return 1.0 - _interp_cdf(p["quantiles"], line)
        case Family.DETERMINISTIC:
            support = np.asarray(p["support"])
            probs = np.asarray(p["probs"])
            return float(probs[support > line].sum())
    raise ValueError(f"unknown family {dist.family}")


def _pmf(dist: Distribution, k: int) -> float:
    p = dist.params
    match dist.family:
        case Family.NEGATIVE_BINOMIAL:
            return float(stats.nbinom.pmf(k, p["r"], p["p"]))
        case Family.POISSON:
            return float(stats.poisson.pmf(k, p["lam"]))
        case Family.BERNOULLI:
            return p["p"] if k == 1 else (1.0 - p["p"] if k == 0 else 0.0)
        case Family.EMPIRICAL_MAX | Family.DETERMINISTIC:
            return 0.0
    raise ValueError(f"unknown family {dist.family}")


def _interp_quantile(quantiles: dict[str, float], q: float) -> float:
    """Linear interpolation between stored Monte Carlo quantiles."""
    xs = sorted(float(k) for k in quantiles)
    ys = [quantiles[_fmt(x)] for x in xs]
    return float(np.interp(q, xs, ys))


def _interp_cdf(quantiles: dict[str, float], value: float) -> float:
    """Invert the stored quantiles to get an approximate CDF at ``value``."""
    xs = sorted(float(k) for k in quantiles)
    ys = [quantiles[_fmt(x)] for x in xs]
    return float(np.interp(value, ys, xs))


def _fmt(x: float) -> str:
    """Round-trip a quantile key the way it was stored."""
    return str(x)


def _thin(arr: np.ndarray, n: int) -> list[float]:
    """A sorted, thinned copy of the Monte Carlo samples, small enough to ship as JSON."""
    s = np.sort(arr)
    if len(s) <= n:
        return [float(x) for x in s]
    idx = np.linspace(0, len(s) - 1, n).astype(int)
    return [float(x) for x in s[idx]]
