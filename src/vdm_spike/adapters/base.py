from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from vdm_spike.core import Snapshot

RandomSample = Literal["native", "reservoir", "query-seeded", "none"]


class CapabilityError(RuntimeError):
    """Raised when a method is called that the adapter's declared capabilities forbid."""


@dataclass
class Capabilities:
    returns_vectors: bool
    unbiased_sample: bool
    live_query: bool
    id_listing: bool = False
    random_sample: RandomSample = "none"
    max_batch: int = 256


@dataclass
class StoreDescriptor:
    name: str
    dimension: int
    metric: str = "cosine"


@dataclass
class ProbeResult:
    ok: bool
    latency_ms: float


@dataclass
class QueryHit:
    id: str
    score: float
    rank: int


@runtime_checkable
class VectorStoreAdapter(Protocol):
    """Read-only contract. `load` is a harness affordance on concrete adapters, not part of this Protocol."""

    def describe(self) -> StoreDescriptor: ...
    def capabilities(self) -> Capabilities: ...
    def count(self, namespace: str) -> int: ...
    def sample(self, namespace: str, k: int) -> Snapshot: ...
    def fetch_by_ids(self, ids: list[str], namespace: str) -> Snapshot: ...
    def query(self, vector: np.ndarray, namespace: str, k: int) -> list[QueryHit]: ...
    def probe(self) -> ProbeResult: ...
