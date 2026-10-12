# Alienese — Devpost Submission Package

## Project Title
**Alienese: Reliable Inference Gateway & Deterministic Grounding for Coding Agents on Nebius Token Factory**

## Elevator Pitch / Tagline
An OpenAI-compatible inference gateway that eliminates out-of-workspace file paths, suppresses unsupported tool arguments, and grounds actions deterministically for coding agents powered by Nebius Token Factory (NVIDIA Nemotron 3 Super).

---

## Inspiration
Autonomous coding agents and SWE-bench runners promise automated software engineering, but in practice, connecting frontier models directly to developer tools leads to acute production failures:
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
- **Testing & Quality:** 185 automated offline unit and regression tests, 100% typechecked with strict Mypy, formatted with Ruff, and protected with Gitleaks secret scanning.

---

## Live Verification & Real-World Results

### 1. Primary Demonstration: Cross-Module Integration Debugging (`--task integration`)
In our automated live workflow (`python scripts/run_coding_task.py --task integration --live`):
- **Step 1 (Failing Target Suite):** Pytest executes on `tests/test_session.py` and fails with `AssertionError: validate_session() returned False after refresh` (1 failed, 2 passed).
- **Step 2 (Deterministic Grounding — Alienese Decision):** Alienese intercepts the traceback, bounds the candidate set to K=1, and executes `read_file("tests/test_session.py")` (1,131 bytes) with SHA-256 state reconstruction.
- **Step 2b (Harness Context Inspection):** Harness inspects test imports and reads cross-module dependencies `auth/token_store.py` (1,302 bytes) and `auth/session_manager.py` (2,317 bytes) from the sandboxed workspace.
- **Step 3 (Genuine Nemotron Code Generation):** Alienese passes the failing test and discovered source context to NVIDIA Nemotron 3 Super via Nebius Token Factory (`nvidia/nemotron-3-super-120b-a12b`). Nemotron synthesizes the corrected implementation of `auth/session_manager.py`, ensuring token validation precedes session state updates and token revocation (2,309 tokens, 2,042.0 ms latency, Attempt 1/3).
- **Step 4 & 5 (Verification & Acceptance):** The model patch is validated via AST compilation and applied within strict write allowlists. The target test suite (`test_session.py`, 3 tests) and independent unseen acceptance suite (`test_acceptance.py`, 3 tests) both pass cleanly (6/6 tests)!
- **Step 6 (Post-Mortem Explanation):** Nebius Nemotron 3 Super receives the verified diff and failing test, generating a concise explanation correctly identifying token validation before session state update and revocation.
- **Workflow Runtime:** **3,118 ms** total wall-clock time end-to-end.

### 2. Secondary Demonstration: Single-Module Token Expiry (`--task token`)
- Baseline single-file token expiry demo repairing TTL past-offset arithmetic in `auth/token.py`.
- **Workflow Runtime:** 1,981 ms total (744 ms Nemotron generation, 850 tokens). 2/2 tests pass.

### 3. One-Turn Tool-Argument Safety Evaluation (Raw Nemotron vs. Alienese)
We evaluated unconstrained Raw Nemotron directly against Alienese on Nebius Token Factory across five targeted one-turn safety scenarios (`results/benchmark_report.md`):
- **Out-of-Workspace Path Proposals:** Raw Nemotron proposed reading `/etc/passwd` (1/5, 20.0%). Alienese blocked traversal, proposing 0 out-of-workspace paths (0/5, 0.0%).
- **Unsupported Target Proposals:** Raw Nemotron proposed reading a directory `src/auth/` when given an ambiguous error (1/5, 20.0%). Alienese safely abstained to an assistant clarification (0/5, 0.0%).
- *Scope note:* This targeted experiment evaluated five specific one-turn safety edge cases. In addition, the separate 35-case candidate-recall evaluation across train, dev, and held-out test splits demonstrated an Oracle Recall@1 lift from 33.3% to 88.9% on held-out test cases (+55.6% lift).

---

## Nebius & NVIDIA Platform Feedback

1. **Nebius Token Factory:**
   - **Throughput & Latency:** Generating 2,309 tokens with Nemotron 3 Super in ~2.0s was remarkably fast and consistent. The regional `us-central1.nebius.com` endpoint delivered ultra-low network overhead.
   - **OpenAI Protocol Alignment:** Nebius's standard OpenAI `/v1/chat/completions` API schema allowed seamless integration without bespoke proprietary SDKs.
   - **Zero-Thinking Control:** The `chat_template_kwargs={"enable_thinking": False}` flag worked reliably to suppress unnecessary reasoning bloat on direct code generation tasks.
   - **Recommendation:** Exposing structured JSON schema constraint enforcement directly on the server side would eliminate post-generation JSON extraction edge cases.

2. **NVIDIA Nemotron 3 Super:**
   - **Code Synthesis Quality:** Nemotron accurately understood cross-module session rotation and stale token states, produced clean, syntax-valid Python code on the first attempt without regression.
   - **Instruction Following:** Excellent adherence to output format instructions when prompts provide clear boundaries.

---

## What Changed During the Hackathon Sprint
- **Phase 0/1:** Built the core runtime foundation, OpenAI protocol normalization, 4-tier trust model, and deterministic SHA-256 state reconstruction.
- **Phase 2:** Implemented the remote provider runtime with deadline budgets, single-probe circuit breakers, and Nebius Token Factory integration.
- **Phase 3:** Engineered the deterministic evidence grounding subsystem and golden evaluation corpus (35 decision points, +55.6% Recall@1 lift).
- **Judge & Integration Sprint:** Added cross-module integration challenge (`auth/session_manager.py` + `auth/token_store.py`), judge quickstart guide, multi-task CLI runner, and strict subprocess secret sanitization.
- **Integrity Polish:** Hardened acceptance verification with revoked-token session refresh rollback tests to ensure zero premature mutations, verified with fresh genuine live Nebius inference.

---

## Links & Submission Assets
- **GitHub Repository:** https://github.com/samat2003/nebius-hack (Public, MIT License)
- **Interactive Trace Viewer Demo:** https://samat2003.github.io/nebius-hack/ (Read-only replay viewer)
- **Judge Quickstart Guide:** `docs/JUDGE_QUICKSTART.md` (30-second offline evaluation guide)
- **Live Run Report:** `results/live_run_report.json` (3,118 ms end-to-end, cross-module integration)
- **Safety Benchmark Report:** `results/benchmark_report.md` (Five targeted safety cases)
- **Release Tag:** `v0.1.4-submission`
