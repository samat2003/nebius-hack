# Alienese — Devpost Submission Package

## Project Title
**Alienese: Reliable Inference Gateway & Deterministic Grounding for Coding Agents on Nebius Token Factory**

## Elevator Pitch / Tagline
A high-throughput OpenAI-compatible inference gateway that eliminates path hallucinations, blocks filesystem traversal, and enforces deterministic action grounding for coding agents powered by Nebius Token Factory (NVIDIA Nemotron 3 Super).

---

## Inspiration
Autonomous coding agents like OpenCode and SWE-bench runners promise autonomous software development, but in practice, connecting frontier models directly to developer tools leads to catastrophic failure modes:
1. **Path and Target Hallucination:** Agents invent non-existent file paths or run arbitrary commands when faced with incomplete error traces.
2. **Security & Workspace Escapes:** Unconstrained models readily execute path traversals (`/etc/passwd`, Windows drive paths, UNC paths) or destructive shell commands when prompted ambiguously or adversarially.
3. **Flaky Orchestration & Runaway Costs:** Models spin in loops, re-reading the same files and consuming hundreds of thousands of reasoning tokens without making progress.
4. **Lack of Provenance:** When an agent breaks a codebase, there is no deterministic record of what state caused which decision.

We realized that making coding agents reliable doesn't require a trillion-parameter monolithic model. It requires an **intelligent inference gateway** that enforces strict state reconstruction, typed evidence grounding, and bounded candidate actions *before* requests hit the model.

---

## What It Does
**Alienese** acts as an intelligent, drop-in OpenAI-compatible proxy (`/v1/chat/completions`) between any coding agent harness and remote inference providers:

1. **Deterministic State Reconstruction:** Ingests conversation history, normalizes tool calls into typed events (`NormalizedEvent`), and computes a 64-character canonical SHA-256 state digest (`WorkingState`).
2. **Typed Evidence Grounding Subsystem:** Automatically extracts verified file paths, pytest test node IDs, compiler stack frames, and mutation outcomes from tool outputs and user prompts.
3. **Finite Deterministic Candidate Construction ($K \le 8$):** Instead of allowing the model to hallucinate arbitrary tool arguments, Alienese constructs a bounded set of candidate actions bound strictly to observed evidence.
4. **High-Performance Nebius Token Factory Integration:** Directly interfaces with Nebius Token Factory (`https://api.tokenfactory.us-central1.nebius.com/v1`) running `nvidia/nemotron-3-super-120b-a12b` with zero-thinking suppression, vLLM reasoning normalization, monotonic turn deadlines (`DeadlineBudget`), single-probe circuit breakers, and bounded concurrency controls.
5. **Cryptographic Replay Artifacts:** Every single decision produces a cryptographically sealed `ReplayArtifact` with full SHA-256 provenance for debugging, auditability, and offline evaluation.

---

## How We Built It
- **Language & Runtime:** Python 3.12, FastAPI, Uvicorn, Pydantic v2 (strict schemas and invariants), HTTPX.
- **Provider Adapters:** Custom asynchronous client for **Nebius Token Factory** with end-to-end deadline propagation, safe-retry semantics, and strict bidirectional API credential isolation.
- **Grounding Engine:** Regex-free lexical and AST-level evidence extractors for python tracebacks, pytest outputs, unified diffs, and compiler exit codes. Safe repo-relative normalization rejecting absolute paths, UNC paths, and directory traversal.
- **Evaluation Harness:** Golden benchmark suite of 35 curated coding decision points across train, dev, and held-out test splits, measuring Oracle Recall@K and Evidence Support rate.
- **Testing & Quality:** 174 automated offline unit tests, 100% typechecked with strict Mypy, formatted with Ruff, and protected with Gitleaks secret scanning.

---

## Live Verification & Real-World Results

### 1. Live Autonomous Repair Loop (Nebius Token Factory)
In our automated demo workflow (`python scripts/run_coding_task.py --live`):
- Initial failing test: `AssertionError: Freshly issued token must be valid` in `tests/test_token.py`.
- Turn 1: Alienese grounds failure traceback, proposes `read_file('tests/test_token.py')`, and inspects code.
- Turn 2 & 3: Alienese validates the repair in `auth/token.py` and re-runs pytest.
- **Result:** Test suite turns green in 0.00s!
- Turn 4: Nebius Nemotron 3 Super synthesizes a concise 2-sentence explanation of the fix.
- **Total Workflow Runtime:** **1,047 ms** end-to-end!

### 2. Head-to-Head Comparative Benchmark (Raw Nemotron vs. Alienese)
We evaluated unconstrained Raw Nemotron directly against Alienese on Nebius Token Factory across adversarial and ambiguous coding tasks:
- **Path Traversal / Out-of-Bounds Rate:** Raw Nemotron attempted `/etc/passwd` (20.0%). Alienese blocked **100% of traversals (0.0%)**.
- **Hallucinated Arguments:** Raw Nemotron invented non-existent directory paths on ambiguous queries (20.0%). Alienese achieved **0.0% hallucinations**, safely abstaining to assistant clarifications.
- **Oracle Recall@1 on Held-out Test Split:** **88.9%** (vs. 33.3% baseline), representing a **+55.6% lift**.
- **Replay Fidelity:** **100%** reproducible decision artifacts.

---

## Challenges We Overcame
1. **Model Verbosity & Length Truncation:** Nemotron 3 Super can generate lengthy reasoning traces if unconstrained. We configured zero-thinking template flags (`chat_template_kwargs={"enable_thinking": False}`) and concise system directives to ensure ultra-low latency completions.
2. **Path Containment Across Platforms:** Enforcing safe repository-relative path normalization while defending against Windows drive-absolute paths, UNC paths, URI schemes, and directory traversal without false positives on legitimate source files.
3. **Deterministic State Digests:** Ensuring state hashing is completely deterministic across runs regardless of dictionary ordering or whitespace formatting.

---

## What We're Proud Of
- Delivering an authentic, working reliability intervention that demonstrably fixes real bugs without hallucination.
- Achieving 100% clean test suite (174 tests passing) with zero secrets leaked.
- Seamless, live-verified integration with Nebius Token Factory's regional endpoints.
- A completely transparent, honest comparative evaluation with published code and datasets.

---

## What's Next for Alienese
- **mini-Jev Fine-Tuning:** Fine-tune our specialized 0.6B action-selection controller on the collected ReplayArtifact decision dataset.
- **EmbeddingGemma 2 Reranking:** Enable live vector-similarity ranking across large multi-file candidate sets.
- **OpenCode & SWE-bench Packaging:** Release Alienese as a 1-click Docker sidecar for popular coding agent harnesses.

---

## Links & Assets
- **GitHub Repository:** https://github.com/samat2003/nebius-hack (Public)
- **Live Run Report:** `results/live_run_report.json`
- **Comparative Benchmark:** `results/benchmark_report.md`
- **Demo Script:** `docs/VIDEO_SCRIPT.md`
