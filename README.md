# Alienese

Alienese is a unified inference API for coding agents that composes specialized models instead of relying on one large language model for every step.

It exposes a single OpenAI-compatible API endpoint that can be plugged into an existing coding harness, agent runtime, IDE backend, or evaluation framework.

## Core idea

Alienese separates three responsibilities:

- **Retrieval / context reduction** — EmbeddingGemma 2 reduces large context and candidate spaces into a smaller relevant working set.
- **Decision / control** — mini-Jev, later fine-tuned as **mini-Jev-SWE**, selects the next action from a bounded set of grounded candidates.
- **Generation / reasoning** — NVIDIA Nemotron on Nebius Token Factory handles open-ended reasoning and code generation when needed.

The runtime—not the models—owns orchestration, state reconstruction, validation, fallbacks, budgets, and observability.

## Initial architecture

The first implementation uses existing model endpoints with **no new training**. The goal is to validate the system architecture before fine-tuning mini-Jev-SWE.

```text
Coding harness
      |
      | OpenAI-compatible request
      v
+-----------------------------+
|          Alienese           |
|                             |
| protocol + state            |
| grounding + candidates      |
| context selection           |
| policy + validation         |
+-----+-----------+-----------+
      |           |
      |           |
      v           v
 Embedding     mini-Jev
 Gemma 2       controller
      \           /
       \         /
        v       v
       Nemotron
   Nebius Token Factory
        |
        v
OpenAI-compatible response
```

Not every request calls every model. Deterministic logic should bypass model calls whenever possible.

## Engineering principles

1. Alienese owns the state machine. Models never orchestrate one another.
2. One external request returns at most one externally visible action.
3. Correctness must not depend on opaque server-side session state.
4. Candidate actions should be grounded before they reach the controller.
5. Nemotron is a generator/reasoner, not a generic tool-argument filler.
6. Retrieval may reduce optional context but must never remove correctness-critical state.
7. Remote calls are bounded by deadlines, retries, budgets, and circuit breakers.
8. Every important decision is traceable and replayable.
9. Provider-specific SDKs stay outside the core runtime.
10. Training begins only after the architecture produces useful traces and measurable failure modes.

See [AGENTS.md](AGENTS.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/PLAN.md](docs/PLAN.md).

## Status

Architecture and repository foundation. No production implementation yet.
