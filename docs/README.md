# Documentation

Design history for vector-drift-monitor. Each phase follows a spec → plan → implementation cycle.

- **[PRD-v0.2.md](PRD-v0.2.md)** — the master product/technical design spec (vector-drift-monitor, Draft v0.2). Architecture, adapter contract, capability model, drift-metric catalog, conformance system, stack options.

## Phase 0 — drift-signal validation spike
- [specs/2026-06-15-vdm-phase0-spike-design.md](specs/2026-06-15-vdm-phase0-spike-design.md) — design
- [plans/2026-06-15-vdm-phase0-spike.md](plans/2026-06-15-vdm-phase0-spike.md) — implementation plan
- Result: signal validated (see top-level [README](../README.md)).

## Phase 1.1 — capability negotiation (proven in Python)
- [specs/2026-06-15-vdm-phase1.1-capability-negotiation-design.md](specs/2026-06-15-vdm-phase1.1-capability-negotiation-design.md) — design
- [plans/2026-06-15-vdm-phase1.1-capability-negotiation.md](plans/2026-06-15-vdm-phase1.1-capability-negotiation.md) — implementation plan
- Result: negotiation validated across 4 adapters / 3 plans; Findings A/B/D (see top-level [README](../README.md)).
