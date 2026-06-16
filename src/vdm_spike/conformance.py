from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vdm_spike.adapters.base import CapabilityError
from vdm_spike.core import Snapshot

_NS = "_conformance"


@dataclass
class ConformanceReport:
    adapter: str
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _fixture(dim: int, n: int = 40) -> Snapshot:
    rng = np.random.default_rng(0)
    return Snapshot(ids=[f"c{i}" for i in range(n)],
                    vectors=rng.normal(size=(n, dim)).astype(np.float32))


def run_conformance(adapter, dim: int) -> ConformanceReport:
    """Verify an adapter's declared capabilities against its actual behavior on a fixture."""
    caps = adapter.capabilities()
    report = ConformanceReport(adapter=adapter.describe().name)
    fx = _fixture(dim)
    adapter.load(fx, _NS)

    if adapter.count(_NS) != fx.n:
        report.failures.append(f"count() returned {adapter.count(_NS)} != {fx.n}")

    # 1. Capability honesty + 2. sampling correctness
    if caps.returns_vectors and caps.unbiased_sample:
        try:
            s = adapter.sample(_NS, k=10)
        except CapabilityError as e:
            report.failures.append(f"declared returns_vectors but sample() raised: {e}")
        else:
            if s.n == 0 or s.n > 10:
                report.failures.append(f"sample(10) returned {s.n} records (expected 1..10)")
            elif s.vectors.shape[1] != dim:
                report.failures.append(f"sample() dim {s.vectors.shape[1]} != declared {dim}")
            if len(set(s.ids)) != len(s.ids):
                report.failures.append("sample() returned duplicate ids")
            if not set(s.ids).issubset(set(fx.ids)):
                report.failures.append("sample() returned ids not in the store")
    else:
        try:
            adapter.sample(_NS, k=10)
            report.failures.append("declared no unbiased sampling but sample() did not raise")
        except CapabilityError:
            pass

    # 3. Query contract
    if caps.live_query:
        hits = adapter.query(fx.vectors[0], _NS, k=5)
        if [h.rank for h in hits] != list(range(len(hits))):
            report.failures.append("query() ranks are not monotonic 0..k-1")
        if any(not (-1.0001 <= h.score <= 1.0001) for h in hits):
            report.failures.append("query() cosine scores out of [-1, 1]")
    else:
        try:
            adapter.query(fx.vectors[0], _NS, k=5)
            report.failures.append("declared no live_query but query() did not raise")
        except CapabilityError:
            pass

    return report
