# Alienese — Judge Quickstart Guide

> **Track:** Coding and Agentic Engineering  
> **Repository:** [github.com/samat2003/nebius-hack](https://github.com/samat2003/nebius-hack)  
> **Interactive Trace Viewer:** [samat2003.github.io/nebius-hack](https://samat2003.github.io/nebius-hack/)

Alienese is a model-composition inference runtime exposed as an OpenAI-compatible API. It solves the fragility and runaway loops of agentic coding harnesses by decomposing agent execution:

1. **Deterministic Grounding & Retrieval** — Code extracts exact file paths, symbols, and test targets from tracebacks and imports into a small, bounded candidate set ($K \le 8$).
2. **Control & Decision Policy** — Evaluates and ranks grounded candidates without hallucinating non-existent paths.
3. **Narrow Model Generation** — Delegated to **NVIDIA Nemotron 3 Super (120B-A12B)** on **Nebius Token Factory** for high-precision code synthesis and post-mortem explanations.

---

## 1. Quick Verification (30 Seconds, No API Key Required)

Alienese includes a fully reproducible, deterministic offline simulation mode that executes the exact same state machine, workspace containment, patch application, and test verification suite without requiring Nebius credentials.

```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Run the realistic cross-module integration challenge (Recommended)
python scripts/run_coding_task.py --task integration --out results/judge_offline.json
```

**Expected Result:**
- Runs in **< 1.0 second**.
- Initial pytest fails on `test_session_refresh_lifecycle` (1 failed, 2 passed).
- Alienese deterministically grounds the failure traceback and discovers `auth/token_store.py` and `auth/session_manager.py`.
- Generates unified diff updating `session["active_token_id"] = new_token_id`.
- Re-runs target suite: **3 passed**.
- Runs independent unseen acceptance suite: **2 passed**.
- Workflow status: **SUCCESS**.

---

## 2. Live NVIDIA Nemotron 3 Super Inference on Nebius Token Factory

If you have a Nebius Token Factory API key, you can run genuine live inference against `nvidia/nemotron-3-super-120b-a12b`:

```bash
# Set your Nebius Token Factory key
export NEBIUS_TOKEN_FACTORY_KEY="your-token-factory-api-key"

# Optional custom base URL (defaults to https://api.tokenfactory.us-central1.nebius.com/v1)
# export NEBIUS_TOKEN_FACTORY_BASE_URL="https://api.tokenfactory.us-central1.nebius.com/v1"

# Run genuine live repair on the integration challenge
python scripts/run_coding_task.py --task integration --live --out results/live_run_report.json
```

**Observed Live Metrics:**
- **End-to-End Workflow Latency:** ~3,500 ms.
- **Nemotron Generation Latency:** ~1,900 ms (2,276 total tokens, 522 completion tokens).
- **Patch Verification:** All 3 target tests and 2 independent acceptance tests pass.
- **Transparent Output:** Complete step-by-step trace and unified diff written to stdout and `results/live_run_report.json`.

---

## 3. Two Distinct Coding Challenges

Alienese provides two independent benchmark challenges:

| Feature | Challenge 1: Single-Module (`--task token`) | Challenge 2: Integration (`--task integration`) [Default] |
|---|---|---|
| **Description** | Single-file token expiration arithmetic bug | Cross-module session manager and token store refresh regression |
| **Modules** | `auth/token.py` | `auth/session_manager.py`, `auth/token_store.py` |
| **Fault Mechanism** | Negative timestamp offset on issue | Session refresh revoked old token in store but did not update session's active pointer |
| **Grounding Scope** | Single file traceback reference | Discovers cross-module dependencies via test import inspection |
| **Target Tests** | `tests/test_token.py` (1 test) | `tests/test_session.py` (3 tests: create, terminate, refresh) |
| **Acceptance Suite** | `tests/test_acceptance.py` (1 test) | `tests/test_acceptance.py` (2 tests: multi-refresh continuity, error rejection) |
| **Model Prompt** | Buggy module + failing test | Discovered modules + failing test + partial traceback |
| **Offline Command** | `python scripts/run_coding_task.py --task token` | `python scripts/run_coding_task.py --task integration` |
| **Live Command** | `python scripts/run_coding_task.py --task token --live` | `python scripts/run_coding_task.py --task integration --live` |

---

## 4. Useful CLI Flags

```bash
# Inspect the disposable workspace on disk after execution
python scripts/run_coding_task.py --task integration --keep-workspace

# Choose output JSON report path
python scripts/run_coding_task.py --task integration --out results/my_evaluation.json
```

---

## 5. Security & Containment Architecture

Every action executed by Alienese operates within strict security boundaries:
1. **Strict Path Containment:** Validated via `Path.is_relative_to(workspace)`. Directory traversal attempts (e.g. `../../etc/passwd`) are rejected.
2. **Writable File Allowlist:** Only designated application source files may be modified. Test files (`tests/`) and configuration files are strictly read-only to prevent reward hacking.
3. **Environment Scrubbing:** Subprocess test runners execute in an environment stripped of all secret variables (`NEBIUS_TOKEN_FACTORY_KEY`, `*_KEY`, `*_SECRET`, `*_TOKEN`).
4. **Bounded Generation & Retries:** At most 3 generation attempts with AST compilation checks; recursive repair loops are strictly prohibited by runtime invariants.

---

## 6. Running the Full Test Suite & Static Quality Gates

```bash
# Run 180+ offline unit, regression, and fault tests
pytest -q

# Run static type checking
mypy src tests

# Run code formatting and linting
ruff check .
ruff format --check .
```
