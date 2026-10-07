# ADR 0001: Core Runtime Invariants

- Status: Accepted
- Date: 2026-10-07

## Context

Alienese composes multiple specialized models behind one external API. Without explicit boundaries, orchestration can easily become a hidden multi-agent loop that is expensive, difficult to debug, and impossible to reproduce.

## Decision

Alienese adopts these foundational invariants:

1. the runtime owns orchestration;
2. models do not invoke other models;
3. one external request returns at most one external action;
4. request/event history is the correctness source of truth;
5. caches are accelerators only;
6. candidate actions are grounded before controller selection whenever possible;
7. generator calls are typed, narrow synthesis jobs;
8. retries and internal repairs are bounded;
9. important decisions are traceable and replayable.

## Consequences

This architecture may require more deterministic systems code than a prompt-chain implementation, but it provides:

- clearer failure isolation;
- measurable subsystem performance;
- reproducibility;
- provider replaceability;
- safer fallback behavior;
- a clean path to later controller fine-tuning.
