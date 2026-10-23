"""Reusable paired statistics; callers supply actual per-image evaluation data."""
import numpy as np


def holm(pvalues, alpha=.05):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Finite p-values in [0,1] required")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1., running)
    return {"adjusted_p": adjusted.tolist(), "reject": (adjusted <= alpha).tolist(), "alpha": alpha}


def crossed_bootstrap(n_images, n_seeds, metric_pair, repeats=2000, seed=1):
    """Resample images and seeds independently, paired across the two methods.

    metric_pair(image_indices,seed_indices) must recompute the two dataset metrics
    (including repeated images), then return (method_a,method_b). AP cannot be
    averaged from per-image APs. Use a callback that runs the relevant evaluator.
    """
    for name, value in (("n_images", n_images), ("n_seeds", n_seeds), ("repeats", repeats)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if repeats < 2:
        raise ValueError("At least two replicates are required to estimate bootstrap uncertainty")
    def evaluate(images, seeds):
        pair = np.asarray(metric_pair(images, seeds), dtype=float)
        if pair.shape != (2,) or not np.isfinite(pair).all():
            raise ValueError("The evaluator must return two finite paired dataset metrics")
        return float(pair[0]), float(pair[1])
    observed_a, observed_b = evaluate(np.arange(n_images), np.arange(n_seeds))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        images = rng.integers(0, n_images, n_images)
        seeds = rng.integers(0, n_seeds, n_seeds)
        a, b = evaluate(images, seeds)
        values.append(float(a - b))
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("The evaluator returned non-finite bootstrap metrics")
    estimate = float(observed_a - observed_b)
    se = float(values.std(ddof=1))
    return {"estimate": estimate, "bootstrap_standard_error": se,
            "method_a_estimate": observed_a, "method_b_estimate": observed_b,
            "normal_95_interval": [estimate - 1.96 * se, estimate + 1.96 * se],
            "percentile_95_interval": np.quantile(values, [.025, .975]).tolist(),
            "replicates": int(repeats), "seed": int(seed),
            "method": "paired_crossed_image_seed_bootstrap",
            "n_images": int(n_images), "n_seeds": int(n_seeds),
            "primary_interval": "normal_95_interval",
            "percentile_interval_role": "supplementary diagnostic; paper uses normal approximation"}
