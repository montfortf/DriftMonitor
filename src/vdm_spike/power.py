from __future__ import annotations

from scipy.stats import norm


def min_detectable_effect(n: int, power: float = 0.8, alpha: float = 0.05) -> float:
    """Heuristic two-sample minimum detectable standardized effect (Cohen's d).

    mde = (z_alpha/2 + z_power) * sqrt(2/n). A first-order honesty signal for the
    spike, NOT a full high-dimensional power analysis.
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    return float((z_alpha + z_power) * (2.0 / n) ** 0.5)


def is_underpowered(n: int, target_effect: float, power: float = 0.8,
                    alpha: float = 0.05) -> bool:
    """True when the achieved sample size can't reliably detect `target_effect`."""
    return min_detectable_effect(n, power, alpha) > target_effect
