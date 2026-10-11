# Alienese — Video Screencast & Demo Script (2:50 Target)

## Overview & Timing Budget
- Total Target Duration: **2 minutes 45 seconds** (Strictly under 3:00 Devpost limit)
- Topic: Reliability Intervention & Deterministic Grounding for Coding Agents on Nebius Token Factory
- Presenter Voice: Clear, technical, concise, authoritative engineering tone.

---

## Part 1: The Problem (0:00 - 0:40)
**[Visual: Split screen showing standard LLM coding agent terminal spinning into chaos with hallucinated file paths and unbounded bash calls.]**

> "Autonomous coding agents are revolutionizing software development. But when you connect frontier reasoning models directly to coding tools, they frequently fail in production in predictable ways:
> 1. They hallucinate non-existent files when investigating bugs.
> 2. They attempt out-of-bounds file system traversals like `/etc/passwd`.
> 3. And they lack deterministic provenance or replayability when things go wrong.
>
> We don't need larger monolithic models to fix this. We need an intelligent inference gateway that enforces strict reliability and safety invariants."

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
**[Visual: Terminal running `python scripts/run_coding_task.py --live` with rich logs and passing tests.]**

> "Let's see Alienese in action live on Nebius Token Factory.
>
> Here we run an authentic token expiry bug:
> - Step 1: The initial pytest run fails with an `AssertionError: Freshly issued token must be valid`.
> - Step 2: The error trace enters Alienese. Instead of hallucinating paths, Alienese grounds the failure directly to `tests/test_token.py` and proposes a grounded `read_file` action.
> - Step 3: Alienese inspects the source, verifies the fix in `auth/token.py`, and re-runs pytest.
> - Step 4: The tests turn green in 0.00s!
> - Step 5: Nebius Token Factory generates a crisp 2-sentence post-mortem explanation: *'The issue was that TokenManager incorrectly stored the issued time with a past offset... We corrected it to record the actual issuance time.'*
>
> The entire multi-turn autonomous repair completes in just 1047 milliseconds."

---

## Part 4: Comparative Evaluation & Impact (2:05 - 2:35)
**[Visual: Displaying the head-to-head benchmark table from `results/benchmark_report.md`.]**

> "We ran a head-to-head benchmark comparing unconstrained Raw Nemotron against Alienese on Nebius Token Factory:
> - **Path Traversal & Escapes:** Raw Nemotron attempted reading `/etc/passwd` when prompted maliciously. Alienese blocked 100% of out-of-bounds traversals.
> - **Hallucinated Arguments:** Raw Nemotron invented non-existent directory paths on ambiguous queries. Alienese eliminated 100% of hallucinations, safely degrading to assistant clarifications.
> - **Candidate Recall:** On our 35-point golden evaluation dataset, Alienese achieved **88.9% Recall@1** on held-out test splits, a **+55.6% lift** over baseline ungrounded systems.
> - **100% Replayability:** Every decision produces a cryptographically verified ReplayArtifact with provenance tracking."

---

## Part 5: Conclusion & Links (2:35 - 2:45)
**[Visual: GitHub repository https://github.com/samat2003/nebius-hack and live endpoints.]**

> "Alienese proves that separating deterministic evidence grounding from remote model generation makes coding agents dramatically more reliable, safe, and cost-effective.
>
> The code, benchmark datasets, and full replay traces are completely open source on GitHub. Thank you to Nebius and the hackathon team!"
