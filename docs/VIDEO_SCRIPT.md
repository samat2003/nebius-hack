# Alienese — Video Screencast & Demo Script (2:45 Target)

## Overview & Timing Budget
- Total Target Duration: **2 minutes 45 seconds** (Strictly under 3:00 Devpost limit)
- Topic: Reliability Intervention & Deterministic Grounding for Coding Agents on Nebius Token Factory
- Presenter Voice: Clear, technical, concise, authoritative engineering tone.

---

## Part 1: The Problem (0:00 - 0:40)
**[Visual: Split screen showing standard LLM coding agent terminal spinning into chaos with hallucinated file paths and unbounded bash calls.]**

> "Autonomous coding agents are revolutionizing software development. But when you connect frontier reasoning models directly to coding tools, they frequently fail in production in predictable ways:
> 1. They hallucinate non-existent files or directories when investigating ambiguous bugs.
> 2. They attempt out-of-bounds file system traversals like `/etc/passwd`.
> 3. And they lack deterministic provenance or replayability when things go wrong.
>
> We don't need larger monolithic models to fix this. We need an intelligent inference gateway that enforces strict reliability and safety invariants before requests hit the model."

---

## Part 2: Introducing Alienese (0:40 - 1:15)
**[Visual: Alienese architecture diagram showing OpenAI-compatible Gateway -> WorkingState Reconstructor -> Deterministic Grounding Engine (K<=8) -> Nebius Token Factory (Nemotron 3 Super).]**

> "Meet **Alienese**: a unified inference gateway and deterministic grounding engine built specifically for coding agents.
>
> Alienese plugs directly into any OpenAI-compatible harness or IDE. Instead of passing unconstrained prompts to the model, Alienese:
> - Reconstructs a canonical, 64-character SHA-256 state digest of the repository's history.
> - Extracts typed, verified evidence from tracebacks, compiler outputs, and pytest failures.
> - Binds tools strictly to observed evidence, bounding the action space to a finite, deterministic candidate set of at most 8 actions.
> - And connects to **Nebius Token Factory**, streaming generation requests to NVIDIA Nemotron 3 Super with zero-thinking suppression and monotonic deadline guarantees."

---

## Part 3: Live End-to-End Autonomous Repair Demo (1:15 - 2:05)
**[Visual: Terminal running `python scripts/run_coding_task.py --live` with authentic logs, diff, and passing tests.]**

> "Let's see Alienese in action live on Nebius Token Factory.
>
> Here we run an authentic token expiry bug:
> - Step 1: The initial pytest run fails with an `AssertionError: Freshly issued token must be valid`.
> - Step 2: The error trace enters Alienese. Instead of hallucinating paths, Alienese deterministically grounds the failure to `tests/test_token.py` and proposes a grounded `read_file` action.
> - Step 3: Alienese feeds the failing test, grounded path, and source code to NVIDIA Nemotron 3 Super on Nebius Token Factory. Nemotron synthesizes the corrected Python implementation of `auth/token.py`.
> - Step 4: The model-generated patch is validated and applied to our sandboxed workspace.
> - Step 5: Both the target unit test and an independent acceptance check pass cleanly!
> - Step 6: Nebius Token Factory generates a concise post-mortem explanation of the fix.
>
> The entire multi-turn autonomous repair completes end-to-end in just 1.7 seconds."

---

## Part 4: Comparative Evaluation & Impact (2:05 - 2:35)
**[Visual: Displaying the safety evaluation table from `results/benchmark_report.md`.]**

> "We evaluated unconstrained Raw Nemotron against Alienese on Nebius Token Factory across targeted safety scenarios:
> - **Out-of-Workspace Path Proposals:** Raw Nemotron proposed reading `/etc/passwd` when prompted maliciously. Alienese blocked 100% of out-of-bounds traversals.
> - **Unsupported Target Proposals:** Raw Nemotron proposed reading a directory `src/auth/` on ambiguous queries. Alienese eliminated 100% of hallucinations, safely degrading to assistant clarifications.
> - **Candidate Recall:** On our separate 35-point golden evaluation dataset, Alienese achieved **88.9% Recall@1** on held-out test splits, a **+55.6% lift** over baseline ungrounded systems.
> - **State Provenance:** Every turn decision produces a ReplayArtifact with SHA-256 provenance tracking."

---

## Part 5: Conclusion & Links (2:35 - 2:45)
**[Visual: GitHub repository https://github.com/samat2003/nebius-hack and live endpoints.]**

> "Alienese proves that separating deterministic evidence grounding from remote model generation makes coding agents dramatically more reliable, safe, and cost-effective.
>
> The code, benchmark datasets, and full replay traces are completely open source on GitHub under the MIT License. Thank you to Nebius and the hackathon team!"
