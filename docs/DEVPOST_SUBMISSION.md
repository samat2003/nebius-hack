# Alienese — Devpost Submission Package

## Project Title
**Alienese: Reliable Inference Gateway & Deterministic Grounding for Coding Agents on Nebius Token Factory**

## Elevator Pitch / Tagline
An OpenAI-compatible inference gateway that eliminates out-of-workspace file paths, suppresses unsupported tool arguments, and grounds actions deterministically for coding agents powered by Nebius Token Factory (NVIDIA Nemotron 3 Super).

---

## Inspiration
Autonomous coding agents and SWE-bench runners promise autonomous software development, but in practice, connecting frontier models directly to developer tools leads to acute failure modes:
1. **Out-of-Workspace & Traversal Escapes:** Unconstrained models readily propose path traversals (`/etc/passwd`, Windows drive paths, UNC paths) or destructive shell commands when prompted ambiguously or adversarially.
2. **Unsupported Target Proposals:** Agents invent arbitrary file paths or run commands when error traces lack explicit locations.
3. **Flaky Orchestration & Runaway Costs:** Models spin in loops re-reading the same files without making progress.
4. **Lack of Provenance:** When an agent breaks a codebase, there is no deterministic record of what state caused which decision.

We realized that making coding agents reliable doesn't require a trillion-parameter monolithic model. It requires an **intelligent inference gateway** that enforces strict state reconstruction, typed evidence grounding, and bounded candidate actions *before* requests hit the model.

---

## What It Does
**Alienese** acts as an intelligent, OpenAI-compatible proxy (`/v1/chat/completions`) between a coding agent harness and remote inference providers:

1. **Deterministic State Reconstruction:** Ingests conversation history, normalizes tool calls into typed events (`NormalizedEvent`), and computes a 64-character canonical SHA-256 state digest (`WorkingState`).
2. **Typed Evidence Grounding Subsystem:** Automatically extracts verified file paths, pytest test node IDs, stack frames, and mutation outcomes from tool outputs and user prompts using deterministic extractors.
3. **Finite Deterministic Candidate Construction ($K \le 8$):** Instead of allowing the model to hallucinate arbitrary tool arguments, Alienese constructs a bounded set of candidate actions bound strictly to observed evidence.
4. **High-Performance Nebius Token Factory Integration:** Directly interfaces with Nebius Token Factory (`https://api.tokenfactory.us-central1.nebius.com/v1`) running `nvidia/nemotron-3-super-120b-a12b` with zero-thinking suppression, monotonic turn deadlines (`DeadlineBudget`), single-probe circuit breakers, and bounded concurrency controls.
5. **SHA-256 State Provenance:** Every turn decision produces a `ReplayArtifact` with full SHA-256 state digests for debugging, auditability, and offline evaluation.

---

## How We Built It
- **Language & Runtime:** Python 3.12, FastAPI, Uvicorn, Pydantic v2 (strict schemas and invariants), HTTPX.
- **Provider Adapters:** Custom asynchronous client for **Nebius Token Factory** with end-to-end deadline propagation, safe-retry semantics, and strict bidirectional API credential isolation.
- **Grounding Engine:** Regex-based evidence extractors for Python tracebacks, pytest outputs, unified diffs, and compiler exit codes. Safe repo-relative normalization rejecting absolute paths, UNC paths, and directory traversal.
- **Evaluation Corpus:** 35 curated coding decision points across train, dev, and held-out test splits, measuring Oracle Recall@K and Evidence Support rate.
- **Testing & Quality:** 178 automated offline unit tests, 100% typechecked with strict Mypy, formatted with Ruff, and protected with Gitleaks secret scanning.

---

## Live Verification & Real-World Results

### 1. Live Autonomous Repair Loop (Nebius Token Factory)
In our automated demo workflow (`python scripts/run_coding_task.py --live`):
- **Step 1 (Failing Test):** Pytest executes on `tests/test_token.py` and fails with `AssertionError: Freshly issued token must be valid`.
- **Step 2 (Deterministic Grounding):** Alienese intercepts the traceback, extracts `tests/test_token.py`, bounds the candidate set ($K=1$), and inspects the test file.
- **Step 3 (Genuine Nemotron Code Generation):** Alienese passes the failing test and buggy code to NVIDIA Nemotron 3 Super via Nebius Token Factory (`nvidia/nemotron-3-super-120b-a12b`). Nemotron synthesizes the corrected Python implementation of `auth/token.py` (673 tokens, 844ms latency).
- **Step 4 & 5 (Verification & Acceptance):** The model-generated patch is applied to the disposable workspace. The target test suite and an independent acceptance check (`test_acceptance.py`) both pass cleanly!
- **Step 6 (Summary Explanation):** Nebius Nemotron 3 Super generates a 2-sentence explanation of why the repair succeeded.
- **Workflow Runtime:** **1,752 ms** total wall-clock time.

### 2. One-Turn Tool-Argument Safety Evaluation (Raw Nemotron vs. Alienese)
We evaluated unconstrained Raw Nemotron directly against Alienese on Nebius Token Factory across 5 adversarial and ambiguous one-turn scenarios (`results/benchmark_report.md`):
- **Out-of-Workspace Path Proposals:** Raw Nemotron proposed reading `/etc/passwd` (1/5, 20.0%). Alienese blocked traversal, proposing 0 out-of-workspace paths (0/5, 0.0%).
- **Unsupported Target Proposals:** Raw Nemotron proposed reading a directory `src/auth/` when given an ambiguous error (1/5, 20.0%). Alienese safely abstained to an assistant clarification (0/5, 0.0%).
- *Scope note:* This experiment evaluated 5 targeted one-turn safety scenarios. In addition, the separate 35-case candidate-recall evaluation across train, dev, and held-out test splits demonstrated an Oracle Recall@1 lift from 33.3% to 88.9% on held-out test cases.

---

## Challenges We Overcame
1. **Model Verbosity & Length Truncation:** Nemotron 3 Super can generate lengthy reasoning traces if unconstrained. We configured zero-thinking template flags (`chat_template_kwargs={"enable_thinking": False}`) and concise system directives to ensure ultra-low latency completions.
2. **Path Containment Across Platforms:** Enforcing safe repository-relative path normalization while defending against Windows drive-absolute paths, UNC paths, URI schemes, and directory traversal.
3. **Deterministic State Digests:** Ensuring state hashing is completely deterministic across runs regardless of dictionary ordering or whitespace formatting.

---

## What We're Proud Of
- Delivering an authentic reliability intervention where Nemotron genuinely generates the bug fix that passes unit and acceptance checks.
- Clean test suite (178 unit tests passing) with zero secrets leaked.
- Seamless, live-verified integration with Nebius Token Factory regional endpoints.
- Completely transparent, honest evaluation with published code, logs, and diffs.

---

## What's Next for Alienese
- **mini-Jev Fine-Tuning:** Train a dedicated action-selection controller on the collected ReplayArtifact decision dataset.
- **EmbeddingGemma 2 Reranking:** Enable live vector-similarity ranking across large multi-file candidate sets.
- **Harness Packaging:** Package Alienese as a Docker sidecar for coding agent harnesses (such as OpenCode and SWE-bench runners).

---

## Links & Assets
- **GitHub Repository:** https://github.com/samat2003/nebius-hack (Public, MIT License)
- **Live Run Report:** `results/live_run_report.json`
- **Comparative Safety Benchmark:** `results/benchmark_report.md`
- **Demo Script:** `docs/VIDEO_SCRIPT.md`
