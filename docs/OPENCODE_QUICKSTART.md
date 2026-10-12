# Alienese with OpenCode — Quickstart Guide

This guide walks you through using **Alienese** with **OpenCode** (v1.19.0+) and **NVIDIA Nemotron 3 Super (120B-A12B)** hosted on **Nebius Token Factory**.

---

## 1. Architecture Overview

Alienese acts as an active enforcement, deterministic grounding, and verification policy layer on top of OpenCode:

```text
┌─────────────────────────────────────────────────────────────────┐
│                          OpenCode CLI                           │
│     (Interactive Terminal Interface & Tool Execution Harness)    │
└────────────────┬───────────────────────────────┬────────────────┘
                 │ Tool Invocation               │ Tool Output
                 ▼                               ▼
       ┌───────────────────┐           ┌───────────────────┐
       │ tool.execute.before│           │ tool.execute.after│
       └─────────┬─────────┘           └─────────┬─────────┘
                 │                               │
                 ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Alienese Plugin (plugins/opencode/alienese.js)     │
│                 Subprocess Bridge: alienese.opencode.bridge     │
├────────────────────────────────┬────────────────────────────────┤
│ PRE-TOOL ENFORCEMENT           │ POST-TOOL DERIVATION           │
│ • Workspace containment guard  │ • WorkingState reconstruction  │
│ • Protected files (.env, .git) │ • SHA-256 state digest         │
│ • Test integrity lock          │ • Evidence extraction (pytest) │
│ • Path normalization & grounding│ • Verification obligation tracking│
└────────────────────────────────┴────────────────────────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │ NVIDIA Nemotron 3 Super (120B-A12B) │
              │        Nebius Token Factory         │
              └─────────────────────────────────────┘
```

---

## 2. Prerequisites

- **OS:** Linux or WSL Ubuntu (tested on Ubuntu 24.04 LTS).
- **Node.js:** v18.0+ (tested on Node v24.18.0 / Bun).
- **Python:** 3.11 or 3.12 with Alienese dependencies (`.venv`).
- **OpenCode CLI:** v1.19.0+ (`~/.opencode/bin/opencode` or global PATH).
- **Nebius Token Factory API Key:** Set in `NEBIUS_TOKEN_FACTORY_KEY`.

---

## 3. Installation

### A. Install OpenCode CLI

```bash
# Official OpenCode installer
curl -fsSL https://opencode.ai/install.sh | bash

# Add OpenCode to PATH if not already added
export PATH="$HOME/.opencode/bin:$PATH"

# Verify installed version
opencode --version
# Expected: 1.19.0 (or newer)
```

### B. Prepare Alienese Runtime

```bash
cd /path/to/nebius-hack
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 4. Configuration

OpenCode supports repository-level configuration via `opencode.json`.

Create or update `opencode.json` in your repository root:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    "/absolute/path/to/nebius-hack/plugins/opencode/alienese.js"
  ],
  "provider": {
    "nebius": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Nebius Token Factory",
      "options": {
        "baseURL": "https://api.tokenfactory.us-central1.nebius.com/v1",
        "apiKey": "{env:NEBIUS_TOKEN_FACTORY_KEY}"
      },
      "models": {
        "nvidia/nemotron-3-super-120b-a12b": {
          "name": "NVIDIA Nemotron 3 Super",
          "tool_call": true
        }
      }
    }
  }
}
```

Set your Nebius Token Factory credentials:

```bash
export NEBIUS_TOKEN_FACTORY_KEY="your-token-factory-key"
```

---

## 5. Usage

### A. Run Headless Automated Repair

To execute an autonomous repair turn using Nemotron on Nebius Token Factory:

```bash
opencode run --auto -m nebius/nvidia/nemotron-3-super-120b-a12b "Run pytest tests/test_session.py"
```

During execution, Alienese intercepts each tool call:
- Logs grounded paths: `[Alienese Grounding] Grounded verified target: auth/session_manager.py`
- Derives state digests: `[Alienese State] Step 4 | Digest: 5d45caf6a07f0f83... | Evidence: 10 items`
- Tracks mutations & obligations: `[Alienese Verification Obligation] ACTIVE: Code mutated. Verification test required.`
- Confirms verification: `[Alienese Verification Verified] PASSED: All tests green.`

### B. Interactive Coding Session

To launch OpenCode's interactive terminal agent with Alienese active:

```bash
opencode -m nebius/nvidia/nemotron-3-super-120b-a12b
```

---

## 6. Automated Evaluation & Integrity Check

Alienese includes an automated benchmark runner that executes the cross-module session management challenge inside a disposable workspace:

```bash
python scripts/run_opencode_eval.py
```

The script autonomously:
1. Provisions an isolated disposable sandbox at `/tmp/alienese_opencode_e2e`.
2. Verifies the authentic cross-module regression fails under `pytest`.
3. Evaluates Alienese's containment guards (rejection of `/etc/passwd`, `.env`, and test alterations).
4. Runs headless OpenCode with NVIDIA Nemotron 3 Super via Nebius Token Factory.
5. Verifies the repaired code passes all target tests and hidden acceptance tests.
6. Writes the complete session trace to `results/opencode_session_trace.jsonl` and summary to `results/opencode_evaluation_report.json`.

---

## 7. Security & Guard Guarantees

Alienese enforces the following guarantees on every OpenCode tool turn:

| Guard | Trigger | Enforcement Action |
|---|---|---|
| **Workspace Containment** | Path points outside workspace (`/etc/passwd`, `../../`) | `tool.execute.before` halts with `[Alienese Policy Rejection]` |
| **Protected File Shield** | Access to `.env`, `.git/config`, SSH keys | `tool.execute.before` halts immediately |
| **Test Integrity Lock** | Modifying tests during repair (`ALIENESE_READ_ONLY_TESTS=1`) | Rejection prevents agent from weakening test suites |
| **Verification Obligation** | Code file mutated (`edit`, `write`) | Sets `verification_obligation: true` until `pytest` passes |
| **Secret Redaction** | Authorization headers, API keys | Redacted in trace logs before writing to disk |
