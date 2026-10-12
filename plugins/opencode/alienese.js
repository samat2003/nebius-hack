/**
 * Alienese OpenCode Plugin.
 *
 * Connects OpenCode's tool execution lifecycle with Alienese's deterministic
 * grounding engine, repository containment guards, state reconstruction,
 * and verification obligations.
 *
 * Supported OpenCode Version: v1.19.0+
 */

import { spawnSync } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Locate repo root and python executable
const REPO_ROOT = path.resolve(__dirname, "../..");
const VENV_PYTHON = path.join(REPO_ROOT, ".venv", "bin", "python");
const PYTHON_PATH_DIR = path.join(REPO_ROOT, "src");

function getPythonCmd(workspaceDir) {
  if (process.env.ALIENESE_PYTHON && fs.existsSync(process.env.ALIENESE_PYTHON)) {
    return process.env.ALIENESE_PYTHON;
  }
  const localVenv = path.join(workspaceDir, ".venv", "bin", "python");
  if (fs.existsSync(localVenv)) {
    return localVenv;
  }
  if (fs.existsSync(VENV_PYTHON)) {
    return VENV_PYTHON;
  }
  return "python3";
}

function invokeBridge(pythonCmd, workspaceDir, args) {
  const env = {
    ...process.env,
    PYTHONPATH: process.env.PYTHONPATH
      ? `${PYTHON_PATH_DIR}:${process.env.PYTHONPATH}`
      : PYTHON_PATH_DIR,
  };

  const res = spawnSync(pythonCmd, ["-m", "alienese.opencode.bridge", ...args], {
    cwd: workspaceDir,
    encoding: "utf8",
    env,
    maxBuffer: 10 * 1024 * 1024,
  });

  if (res.error) {
    console.error("[Alienese Plugin] Bridge execution error:", res.error.message);
    return null;
  }

  if (res.status !== 0) {
    if (res.stderr) {
      console.error("[Alienese Plugin] Bridge stderr:", res.stderr.trim());
    }
    return null;
  }

  try {
    return JSON.parse(res.stdout.trim());
  } catch (err) {
    console.error("[Alienese Plugin] Failed to parse bridge output:", res.stdout);
    return null;
  }
}

export const AlienesePlugin = async ({ project, client, directory, worktree }) => {
  const workspaceDir = directory || process.cwd();
  const pythonCmd = getPythonCmd(workspaceDir);

  console.error("==================================================================");
  console.error(" [Alienese OpenCode Plugin v0.1.4]");
  console.error(" Mode: Active Enforcement & Deterministic Grounding");
  console.error(` Workspace: ${workspaceDir}`);
  console.error(` Bridge Runtime: ${pythonCmd}`);
  console.error("==================================================================");

  return {
    tool: {
      alienese_plan: {
        description:
          "Invoke Alienese's multi-model orchestration engine (EmbeddingGemma 2 retrieval ranking + mini-Jev decision controller) to analyze repository context, rank files, and select the optimal grounded next action. Call this tool before reading or editing code.",
        args: {
          goal: {
            type: "string",
            description: "The coding task, failing test, or bug description to analyze.",
          },
        },
        execute: async (args, context) => {
          const goal = args?.goal || "";
          const sessionID = context?.sessionID || "default";
          const targetDir = context?.directory || workspaceDir;
          const bridgeArgs = [
            "plan",
            "--workspace", targetDir,
            "--session", sessionID,
            "--goal", goal,
          ];
          const result = invokeBridge(pythonCmd, targetDir, bridgeArgs);
          if (!result) {
            return "Alienese planning bridge unavailable.";
          }
          if (typeof result === "string") {
            return result;
          }
          return result.plan_text || JSON.stringify(result, null, 2);
        },
      },
    },

    "tool.execute.before": async (input, output) => {
      const toolName = input?.tool || "unknown";
      const callID = input?.callID || "";
      const sessionID = input?.sessionID || "default";
      const targetDir = input?.directory || workspaceDir;
      const args = output?.args || {};

      const bridgeArgs = [
        "pre-tool",
        "--workspace", targetDir,
        "--session", sessionID,
        "--tool", toolName,
        "--call-id", callID,
        "--args", JSON.stringify(args),
      ];

      // Enable read-only test enforcement if set
      if (process.env.ALIENESE_READ_ONLY_TESTS === "1") {
        bridgeArgs.push("--read-only-tests");
      }

      const decision = invokeBridge(pythonCmd, targetDir, bridgeArgs);
      if (!decision) {
        // Fallback safely if bridge was unreachable
        return;
      }

      if (decision.allowed === false) {
        console.error(`[Alienese Enforced Guard: ${decision.guard_name}] ${decision.reason}`);
        throw new Error(`[Alienese Policy Rejection] ${decision.reason}`);
      }

      if (decision.grounded_path) {
        console.error(`[Alienese Grounding] Grounded verified target: ${decision.grounded_path}`);
      }
    },

    "tool.execute.after": async (input, output) => {
      const toolName = input?.tool || "unknown";
      const callID = input?.callID || "";
      const sessionID = input?.sessionID || "default";
      const targetDir = input?.directory || workspaceDir;
      const args = input?.args || {};

      const rawOutput = output?.output || output?.title || "";
      const outputText = typeof rawOutput === "string" ? rawOutput : JSON.stringify(rawOutput);

      const bridgeArgs = [
        "post-tool",
        "--workspace", targetDir,
        "--session", sessionID,
        "--tool", toolName,
        "--call-id", callID,
        "--args", JSON.stringify(args),
        "--output", JSON.stringify(outputText),
      ];

      const result = invokeBridge(pythonCmd, targetDir, bridgeArgs);
      if (!result) {
        return;
      }

      const digestShort = (result.state_digest || "").slice(0, 16);
      console.error(
        `[Alienese State] Step ${result.step} | Digest: ${digestShort}... | Evidence: ${result.evidence_count} items`
      );

      if (result.three_model_guidance) {
        if (result.three_model_guidance.retrieval_summary) {
          console.error(
            `[Alienese Retrieval: EmbeddingGemma 2] Ranked: ${result.three_model_guidance.retrieval_summary}`
          );
        }
        if (result.three_model_guidance.decision_summary) {
          console.error(
            `[Alienese Decision: mini-Jev] Policy Choice: ${result.three_model_guidance.decision_summary}`
          );
        }
      }

      if (result.verification_obligation === true) {
        const mutatedList = (result.mutations || []).join(", ") || "source files";
        console.error(
          `[Alienese Verification Obligation] ACTIVE: Code mutated (${mutatedList}). Verification test required.`
        );
      } else if (result.verification_obligation === false && (result.mutations || []).length > 0) {
        console.error(
          `[Alienese Verification Verified] PASSED: All tests green. Mutation verified successfully.`
        );
      }
    },
  };
};

export default AlienesePlugin;
