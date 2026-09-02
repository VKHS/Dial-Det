"""Seed-aware paired inference and multiplicity helpers."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable, Mapping, Sequence

import numpy as np


def holm_adjustment(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0,1]")
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.empty(len(p), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p) - rank) * p[index])
        running = max(running, value)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(p), dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted


@dataclass(frozen=True)
class TOSTResult:
    mean_difference: float
    standard_error: float
    lower_margin: float
    upper_margin: float
    p_lower: float
    p_upper: float
    equivalent: bool


def paired_tost(
    differences: Sequence[float] | np.ndarray,
    *,
    lower_margin: float,
    upper_margin: float,
    alpha: float = 0.05,
) -> TOSTResult:
    x = np.asarray(differences, dtype=np.float64).reshape(-1)
    if len(x) < 2 or np.any(~np.isfinite(x)):
        raise ValueError("at least two finite paired differences are required")
    if not lower_margin < upper_margin or not 0 < alpha < 1:
        raise ValueError("invalid equivalence margins or alpha")
    try:
        from scipy.stats import t
    except ImportError as exc:
        raise RuntimeError("SciPy is required for TOST") from exc
    mean = float(x.mean())
    standard_error = float(x.std(ddof=1) / sqrt(len(x)))
    if standard_error == 0:
        p_lower = 0.0 if mean > lower_margin else 1.0
        p_upper = 0.0 if mean < upper_margin else 1.0
    else:
        lower_statistic = (mean - lower_margin) / standard_error
        upper_statistic = (mean - upper_margin) / standard_error
        p_lower = float(t.sf(lower_statistic, df=len(x) - 1))
        p_upper = float(t.cdf(upper_statistic, df=len(x) - 1))
    return TOSTResult(
        mean_difference=mean,
        standard_error=standard_error,
        lower_margin=lower_margin,
        upper_margin=upper_margin,
        p_lower=p_lower,
        p_upper=p_upper,
        equivalent=p_lower < alpha and p_upper < alpha,
    )


def hierarchical_seed_image_bootstrap(
    per_seed_records: Mapping[int, object],
    estimator: Callable[[list[object]], float],
    *,
    resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Resample seeds and delegate within-seed image resampling to the estimator.

    Each record may be any estimator-specific container.  The caller's
    estimator receives the seed-resampled list; this supports pooled AP
    re-evaluation rather than incorrectly averaging image-wise AP values.
    """
    if len(per_seed_records) < 2:
        raise ValueError("at least two training seeds are required")
    if resamples < 100 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    seed_ids = np.asarray(sorted(per_seed_records), dtype=np.int64)
    rng = np.random.default_rng(seed)
    observed = float(estimator([per_seed_records[int(i)] for i in seed_ids]))
    samples = np.empty(resamples, dtype=np.float64)
    for position in range(resamples):
        selected = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        records = [per_seed_records[int(i)] for i in selected]
        samples[position] = float(estimator(records))
    alpha = 1.0 - confidence
    return {
        "estimate": observed,
        "ci_low": float(np.quantile(samples, alpha / 2)),
        "ci_high": float(np.quantile(samples, 1 - alpha / 2)),
        "resamples": float(resamples),
    }


def max_t_simultaneous_intervals(
    observed: np.ndarray,
    bootstrap_replicates: np.ndarray,
    *,
    confidence: float = 0.95,
) -> dict[str, np.ndarray | float]:
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    replicates = np.asarray(bootstrap_replicates, dtype=np.float64)
    if replicates.ndim != 2 or replicates.shape[1] != len(observed):
        raise ValueError("bootstrap_replicates must have shape [B,C]")
    if len(replicates) < 100 or not 0 < confidence < 1:
        raise ValueError("invalid max-t inputs")
    standard_error = replicates.std(axis=0, ddof=1)
    if np.any(standard_error <= 0):
        raise ValueError("each contrast must have positive bootstrap standard error")
    studentized = np.abs((replicates - observed[None, :]) / standard_error[None, :])
    critical = float(np.quantile(np.max(studentized, axis=1), confidence))
    return {
        "lower": observed - critical * standard_error,
        "upper": observed + critical * standard_error,
        "critical_value": critical,
        "standard_error": standard_error,
    }
