from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Snapshot:
    """A set of vectors read from (or destined for) a store."""

    ids: list[str]
    vectors: np.ndarray  # shape (n, d), float32
    metadata: list[dict] | None = None

    @property
    def n(self) -> int:
        return self.vectors.shape[0]

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]


@dataclass
class Expectation:
    """Which detectors a scenario asserts should fire (True) or stay quiet (False).

    Only the listed detector names are gated; others are reported but not asserted.
    """

    fires: dict[str, bool]


@dataclass
class DetectorResult:
    name: str
    statistic: float
    p_value: float | None
    fired: bool
    detail: dict = field(default_factory=dict)
