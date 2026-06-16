import numpy as np
import pytest

from vdm_spike.adapters.base import CapabilityError
from vdm_spike.adapters.fakes import FakeMinimalAdapter, FakeQueryOnlyAdapter
from vdm_spike.core import Snapshot
from vdm_spike.negotiation import DriftPlan, select_drift_plan


def _snap(seed, n):
    rng = np.random.default_rng(seed)
    return Snapshot(ids=[f"d{i}" for i in range(n)],
                    vectors=rng.normal(size=(n, 384)).astype(np.float32))


def test_query_only_selects_query_plan():
    a = FakeQueryOnlyAdapter(dim=384)
    assert select_drift_plan(a.capabilities()) is DriftPlan.QUERY


def test_query_only_sample_is_gated():
    a = FakeQueryOnlyAdapter(dim=384)
    a.load(_snap(0, 50), namespace="baseline")
    with pytest.raises(CapabilityError):
        a.sample("baseline", k=10)


def test_query_only_query_works():
    a = FakeQueryOnlyAdapter(dim=384)
    snap = _snap(0, 50)
    a.load(snap, namespace="baseline")
    hits = a.query(snap.vectors[0], namespace="baseline", k=5)
    assert len(hits) == 5
    assert hits[0].id == "d0"
    assert [h.rank for h in hits] == [0, 1, 2, 3, 4]


def test_minimal_selects_minimal_plan():
    a = FakeMinimalAdapter(dim=384)
    assert select_drift_plan(a.capabilities()) is DriftPlan.MINIMAL
