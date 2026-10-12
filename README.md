# Alienese — Evidence-Grounded Reliability for AI Coding Agents

[![CI](https://github.com/samat2003/nebius-hack/actions/workflows/ci.yml/badge.svg)](https://github.com/samat2003/nebius-hack/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Nebius Token Factory](https://img.shields.io/badge/Inference-Nebius%20Token%20Factory-green.svg)](https://nebius.com/)
[![Model](https://img.shields.io/badge/Model-NVIDIA%20Nemotron%203%20Super-purple.svg)](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b)

**Alienese** is an evidence-grounded inference gateway and model-composition runtime exposed as an OpenAI-compatible API.

Coding agents often fail acutely in production: unconstrained LLMs hallucinate non-existent files, propose directory traversals outside the workspace, get trapped in loops, and lack deterministic auditability. Alienese eliminates these failure modes by enforcing **deterministic state reconstruction** and **evidence-grounded candidate bounding** ($K \le 8$) before delegating code synthesis to **NVIDIA Nemotron 3 Super (120B-A12B)** running on **Nebius Token Factory**.

- 🌐 **Interactive Replay Trace Viewer:** [samat2003.github.io/nebius-hack](https://samat2003.github.io/nebius-hack/)
- 📖 **Judge Quickstart Guide:** [docs/JUDGE_QUICKSTART.md](docs/JUDGE_QUICKSTART.md)
- 📊 **Live Run Report:** [results/live_run_report.json](results/live_run_report.json)
- 🛡️ **Comparative Safety Benchmark:** [results/benchmark_report.md](results/benchmark_report.md)
- 📦 **Latest Release:** [v0.1.3-submission](https://github.com/samat2003/nebius-hack/releases)

---

## 2. See It Work

Alienese runs out of the box in both offline simulation mode (no API credentials or internet needed) and genuine live inference on Nebius Token Factory.

```bash
# Clone and prepare environment
git clone https://github.com/samat2003/nebius-hack.git
cd nebius-hack
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### A. Offline Simulation (Reproducible, 0 Credentials, < 1.0s)
Executes the full state machine, workspace containment, patch application, and pytest verification suite using deterministic offline playback:

```bash
# Recommended: Cross-module session manager integration challenge
python scripts/run_coding_task.py --task integration --out results/judge_offline.json

# Baseline fallback: Single-module token expiry demo
python scripts/run_coding_task.py --task token --out results/token_offline.json
```

### B. Live Nebius Token Factory Inference
Runs genuine live model inference against `nvidia/nemotron-3-super-120b-a12b` via Nebius Token Factory:

```bash
# Provide your own Nebius Token Factory API key
export NEBIUS_TOKEN_FACTORY_KEY="your-token-factory-key"

# Run genuine live repair on the integration challenge
python scripts/run_coding_task.py --task integration --live --out results/live_run_report.json
```

---

## 3. Why Alienese Is Different

Standard coding agents ask a large LLM to freely generate tool names and JSON arguments. When an error occurs, the model guesses file locations and commands.

Alienese separates responsibility: code extracts deterministic facts, policy ranks bounded candidates, and the model handles narrow code synthesis within harness-constrained boundaries.

```text
[Failing Test Traceback]
          │
          ▼
┌───────────────────────────────────────────────┐
│ 1. Deterministic Grounding Subsystem          │
│    - Extracts exact file paths & test targets │
│    - Binds schema parameters to real evidence │
│    - Rejects path traversals & unsafe flags   │
│    - Generates bounded candidate set (K ≤ 8)  │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ 2. Grounded Controller Decision (Turn 1)      │
│    - Selects grounded action from candidate   │
│      set (read_file tests/test_session.py)    │
│    - Produces SHA-256 state digest & trace    │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ 3. Harness-Constrained Context Inspection     │
│    - Reads imported modules from sandbox      │
│      (auth/token_store.py & session_manager)  │
│    - Verifies path containment & allowlists   │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ 4. NVIDIA Nemotron 3 Super (Nebius Factory)   │
│    - Receives labeled test + source context   │
│    - Synthesizes targeted Python repair       │
│    - Validated via AST syntax compilation     │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ 5. Verification & Independent Acceptance Check│
│    - Strict sandbox applies diff to workspace │
│    - Re-runs target suite (RED ➔ GREEN)       │
│    - Re-runs independent unseen acceptance    │
│    - Generates diff-grounded explanation      │
└───────────────────────────────────────────────┘
```

### A Real Trace Walkthrough (`--task integration`)
1. **Failure Evidence:** Initial pytest run fails on `test_session_refresh_lifecycle` in `tests/test_session.py` (`AssertionError`).
2. **Deterministic Grounding (Alienese Turn 1):** Alienese intercepts the traceback, bounds candidates ($K=1$), and executes `read_file("tests/test_session.py")` with SHA-256 state provenance.
3. **Context Inspection (Harness-Constrained):** Harness inspects test imports and reads cross-module dependencies (`auth/token_store.py` and `auth/session_manager.py`) within sandbox allowlists.
4. **Nemotron Code Synthesis (Turn 2):** `nvidia/nemotron-3-super-120b-a12b` on Nebius Token Factory receives the failure traceback and source files, synthesizing the missing session update:
   ```diff
          old_token = self.token_store.get_token(old_token_id)
          if old_token is None:
              return None
   +      # Update active token ID in session to reflect the new token
   +      session["active_token_id"] = new_token_id
   ```
5. **Test & Acceptance Verification (Turn 3):** Sandboxed harness verifies all 3 unit tests pass and both independent unseen acceptance tests pass (5/5 tests).
6. **Diff-Grounded Post-Mortem (Turn 4):** Nebius Nemotron receives the verified diff and generates a 2-sentence explanation accurately identifying the stale active token pointer and resolution.
7. **Replay Trace:** Full execution telemetry, token usage, and SHA-256 state digests are logged transparently to `results/live_run_report.json`.

---

## 4. Actual Measured Results

### A. Live Autonomous Coding Workflows (Nebius Token Factory)

| Metric | Challenge 2: Integration Debugging (`--task integration`) [Primary] | Challenge 1: Single-Module Expiry (`--task token`) [Baseline] |
|---|---|---|
| **Task Description** | Cross-module session manager and token store refresh regression | Single-module token TTL past-offset expiration bug |
| **Modules Involved** | `auth/session_manager.py` + `auth/token_store.py` | `auth/token.py` |
| **End-to-End Latency** | **3,382 ms** | **1,981 ms** |
| **Nemotron Generation Latency** | **1,936 ms** | **744 ms** |
| **Tokens Consumed** | 2,273 total (536 completion) | 850 total (134 completion) |
| **Target Test Suite** | 3 passed (100%) | 1 passed (100%) |
| **Unseen Acceptance Suite** | 2 passed (100%) | 1 passed (100%) |
| **Overall Status** | **SUCCESS** | **SUCCESS** |

### B. One-Turn Tool-Argument Safety Evaluation (Raw Nemotron vs. Alienese)
Evaluated across five targeted one-turn safety scenarios (`results/benchmark_report.md`):

| Safety Metric | Raw Nemotron (Direct) | Alienese + Nebius |
|---|---|---|
| **Out-of-Workspace Path Proposals** | 1 / 5 (20.0%) | **0 / 5 (0.0%)** |
| **Unsupported Target Proposals** | 1 / 5 (20.0%) | **0 / 5 (0.0%)** |
| **Action Space Bounding** | Unbounded | **Bounded ($K \le 8$)** |
| **State Provenance** | None | **SHA-256 Digest** |

*Evaluation limitation note: This targeted safety benchmark measures five specific edge scenarios.*

### C. 35-Case Golden Grounding Corpus Evaluation
Evaluated against 35 curated coding decision points across train, dev, and held-out test splits (`tests/fixtures/grounding/decision_points.json`):

| Evaluation Split | Metric | Baseline (Phase 1) | Alienese (Phase 3) | Improvement |
|---|---|---|---|---|
| **Held-Out Test Split** (N=10) | **Oracle Recall@1** | 33.3% (3/9) | **88.9% (8/9)** | **+55.6% lift** |
| | Exact Arg Accuracy | 33.3% (3/9) | **88.9% (8/9)** | **+55.6% lift** |
| | Executable Validity | 100.0% (10/10) | **100.0% (10/10)** | Maintained |
| **Overall Corpus** (N=35) | **Oracle Recall@1** | 20.7% (6/29) | **79.3% (23/29)** | **+58.6% lift** |
| | Exact Arg Accuracy | 20.7% (6/29) | **82.8% (24/29)** | **+62.1% lift** |
| | Abstention Correctness | 100.0% (6/6) | **100.0% (6/6)** | Maintained |

---

## 5. Architecture

```text
Coding Harness / SWE Runner
            │
            ▼ OpenAI-compatible HTTP Request (/v1/chat/completions)
┌────────────────────────────────────────────────────────┐
│                  Alienese Core Runtime                 │
│                                                        │
│  ┌──────────────────────┐    ┌──────────────────────┐  │
│  │ State Reconstruction │    │ Grounding Subsystem  │  │
│  │ (SHA-256 Digest)     │    │ (Evidence Extractors)│  │
│  └──────────┬───────────┘    └──────────┬───────────┘  │
│             │                           │              │
│             └─────────────┬─────────────┘              │
│                           ▼                            │
│              ┌───────────────────────────┐             │
│              │ Grounded Candidate Bounding│             │
│              │ (Finite Candidates K ≤ 8) │             │
│              └────────────┬──────────────┘             │
│                           │                            │
│                           ▼                            │
│              ┌───────────────────────────┐             │
│              │ GroundedRankController    │             │
│              │ (Deterministic ranking)   │             │
│              └────────────┬──────────────┘             │
└───────────────────────────┼────────────────────────────┘
                            │
                            ▼ Asynchronous Provider Client
┌────────────────────────────────────────────────────────┐
│                  Nebius Token Factory                  │
│  Model: nvidia/nemotron-3-super-120b-a12b              │
│  - Regional endpoint: us-central1                      │
│  - Monotonic turn deadlines & single-probe breakers    │
│  - Thinking suppression & strict credential isolation  │
└────────────────────────────────────────────────────────┘
```

### Components Running Today:
- **`TurnEngine`**: Single-turn coordinator executing the reconstruct $\to$ ground $\to$ decide $\to$ generate lifecycle.
- **`Grounding Subsystem`**: Deterministic extractors (`paths`, `symbols`, `commands`, `failures`, `mutation`).
- **`GroundedRankController`**: Grounded candidate policy enforcing evidence support.
- **`NebiusTokenFactoryGenerator`**: High-performance HTTP client connecting to Nebius Token Factory.
- **`StandaloneCodingHarness`**: Security sandbox enforcing strict path containment and write allowlists.

### Future Integrations (Post-Hackathon Roadmap):
- **Self-Hosted mini-Jev Controller:** Pinned at `samatv256/mini-Jev` (`step-010626`); currently mocked in offline tests until self-hosted vLLM deployment is finalized.
- **Self-Hosted EmbeddingGemma 2 Retriever:** `google/embeddinggemma-300m`; currently evaluated via offline embeddings.

---

## 6. Reproduce and Verify

### Step 1. Run Offline Quality Gates (100% Offline, No Secrets)
```bash
# Run all 180+ offline unit, regression, and invariant tests
pytest -q

# Run static type checking
mypy src tests

# Run linter and formatting checks
ruff check .
ruff format --check .

# Run 35-case grounding candidate evaluation
python -m alienese.grounding.eval
```

### Step 2. Run the Offline Coding Tasks
```bash
# Run integration challenge in offline simulation
python scripts/run_coding_task.py --task integration --out results/judge_offline.json

# Run token challenge in offline simulation
python scripts/run_coding_task.py --task token --out results/token_offline.json
```

### Step 3. Run Live Nebius Inference (Requires Your Own Key)
```bash
export NEBIUS_TOKEN_FACTORY_KEY="your-key"
python scripts/run_coding_task.py --task integration --live --out results/live_run_report.json
```

### Troubleshooting
- **Missing API Key:** If running with `--live` without `NEBIUS_TOKEN_FACTORY_KEY`, Alienese exits with an explicit missing-credential error. Run without `--live` to test offline.
- **Custom Base URL:** If using a private endpoint, set `NEBIUS_TOKEN_FACTORY_BASE_URL="https://your-endpoint/v1"`.

---

## 7. Technical Credibility

### Provider Runtime Resilience
- **Monotonic Turn Deadlines (`DeadlineBudget`):** End-to-end deadlines that count down monotonically across retries.
- **Safe-to-Retry Classification:** Distinguishes idempotent failures from ambiguous network timeouts.
- **Single-Probe Circuit Breakers:** Protects inference budgets by failing fast during outages and admitting a single test request in `HALF_OPEN` state.
- **Pre-Allocation Byte Caps:** Caps JSON decoding depth and response payloads to prevent memory exhaustion attacks.

### Security & Containment Architecture
- **Strict Path Containment:** Validated via `Path.is_relative_to(workspace)`. Traversal attempts (e.g. `../../etc/passwd`, Windows drive letters, UNC paths) are rejected.
- **Mutable File Allowlist:** Only application source files can be written. Tests (`tests/`) and configs are strictly read-only to prevent test-tampering reward hacks.
- **Environment Scrubbing:** Test executions run in an isolated environment stripped of all secret variables (`*_KEY`, `*_SECRET`, `*_TOKEN`).

### OpenAI-Compatible API Endpoints
- `GET /health` — Runtime liveness, version, and provider status.
- `GET /v1/models` — Returns `alienese-default`.
- `POST /v1/chat/completions` — Single-turn non-streaming chat completion supporting function tool schemas and `tool_choice="required"`.

---

## 8. Hackathon Information

- **Hackathon Track:** Coding and Agentic Engineering
- **Platform Provider:** **Nebius AI Studio / Token Factory** (`https://api.tokenfactory.us-central1.nebius.com/v1`)
- **Model Attribution:** **NVIDIA Nemotron 3 Super** (`nvidia/nemotron-3-super-120b-a12b`), utilized for genuine code repair generation and causal bug explanation.
- **License:** [MIT License](LICENSE)
- **Work Completed During Hackathon:** The entire Alienese architecture—including the core turn engine, deterministic grounding subsystem, provider runtime with circuit breakers, Nebius Token Factory adapter, 35-case golden evaluation corpus, multi-challenge coding harness, and replay trace viewer—was designed, implemented, and verified during this hackathon sprint.
- *Scope Note on OpenCode:* OpenCode integration was investigated during the sprint; however, to maintain strict benchmark integrity and avoid unverified harness claims, Alienese currently operates with its own sandboxed runner and OpenAI-compatible API gateway. OpenCode client integration is scheduled for future work.
