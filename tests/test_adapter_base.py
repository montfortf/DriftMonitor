from vdm_spike.adapters.base import (
    Capabilities,
    CapabilityError,
    ProbeResult,
    QueryHit,
    StoreDescriptor,
)


def test_capabilities_defaults():
    c = Capabilities(returns_vectors=True, unbiased_sample=True, live_query=True)
    assert c.id_listing is False
    assert c.random_sample == "none"
    assert c.max_batch == 256


def test_dataclasses_construct():
    assert StoreDescriptor(name="x", dimension=384).metric == "cosine"
    assert ProbeResult(ok=True, latency_ms=1.2).ok is True
    assert QueryHit(id="a", score=0.9, rank=0).rank == 0


def test_capability_error_is_runtime_error():
    assert issubclass(CapabilityError, RuntimeError)
