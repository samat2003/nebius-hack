/**
 * Alienese OpenCode Plugin.
 *
 * Connects OpenCode's tool execution lifecycle with Alienese's deterministic
 * grounding engine, repository containment guards, state reconstruction,
 * and verification obligations.
 *
 * Supported OpenCode Version: v1.19.0+
 */

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const execFileAsync = promisify(execFile);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Locate repo root and python executable
const REPO_ROOT = path.resolve(__dirname, "../..");
const VENV_PYTHON = path.join(REPO_ROOT, ".venv", "bin", "python");
const PYTHON_PATH_DIR = path.join(REPO_ROOT, "src");

const PROTECTED_TOOLS = new Set([
  "read",
  "read_file",
  "view",
  "view_file",
  "open",
  "glob",
  "grep",
  "search",
  "file_search",
  "find",
  "find_by_name",
  "list_dir",
  "list_directory",
  "ls",
  "write",
  "write_file",
  "write_to_file",
  "edit",
  "edit_file",
  "replace_file_content",
  "patch",
  "apply_patch",
  "bash",
  "run_command",
  "command",
  "execute_command",
]);

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

async function invokeBridge(pythonCmd, workspaceDir, args, timeoutMs = 15000) {
  const env = {
    ...process.env,
    PYTHONPATH: process.env.PYTHONPATH
      ? `${PYTHON_PATH_DIR}:${process.env.PYTHONPATH}`
      : PYTHON_PATH_DIR,
  };

  try {
    const { stdout, stderr } = await execFileAsync(
      pythonCmd,
      ["-m", "alienese.opencode.bridge", ...args],
      {
        cwd: workspaceDir,
        encoding: "utf8",
        env,
        timeout: timeoutMs,
        maxBuffer: 10 * 1024 * 1024,
      }
    );

    if (!stdout || !stdout.trim()) {
      if (stderr) {
        console.error("[Alienese Plugin] Bridge stderr:", stderr.trim());
      }
      return null;
    }

    return JSON.parse(stdout.trim());
  } catch (err) {
    console.error("[Alienese Plugin] Bridge execution error:", err.message);
    return null;
  }
}

export const AlienesePlugin = async ({ project, client, directory, worktree }) => {
  const workspaceDir = directory || process.cwd();
  const pythonCmd = getPythonCmd(workspaceDir);

  console.error("==================================================================");
  console.error(" [Alienese OpenCode Plugin v0.1.5]");
  console.error(" Mode: Active Enforcement & Deterministic Grounding");
  console.error(` Workspace: ${workspaceDir}`);
  console.error(` Bridge Runtime: ${pythonCmd}`);
  console.error(" Live Companion Dashboard: http://127.0.0.1:4097/");
  console.error("==================================================================");

  // Attempt to use documented OpenCode v1.19 Zod tool helper
  let toolHelper = null;
  try {
    const pluginMod = await import("@opencode-ai/plugin");
    if (pluginMod && typeof pluginMod.tool === "function") {
      toolHelper = pluginMod.tool;
    }
  } catch {
    toolHelper = null;
  }

  const executePlan = async (args, context) => {
    const goal = args?.goal || "";
    const sessionID = context?.sessionID || "default";
    const targetDir = context?.directory || workspaceDir;

    // Strict containment verification
    if (process.env.ALIENESE_EXPECTED_WORKSPACE) {
      const expected = path.resolve(process.env.ALIENESE_EXPECTED_WORKSPACE);
      const actual = path.resolve(targetDir);
      if (actual !== expected) {
        return `[Alienese Workspace Mismatch] Expected workspace '${expected}' but received '${actual}'. Rejecting uncontained demo operation.`;
      }
    }

    const bridgeArgs = [
      "plan",
      "--workspace", targetDir,
      "--session", sessionID,
      "--goal", goal,
    ];
    const result = await invokeBridge(pythonCmd, targetDir, bridgeArgs, 20000);
    if (!result) {
      return "Alienese planning bridge unavailable or timed out.";
    }
    if (typeof result === "string") {
      return result;
    }
    return result.plan_text || JSON.stringify(result, null, 2);
  };

  const planToolDef = toolHelper
    ? toolHelper({
        description:
          "Invoke Alienese's multi-model orchestration engine (EmbeddingGemma 2 retrieval ranking + mini-Jev decision controller) to analyze repository context, rank files, and select the optimal grounded next action. Call this tool before reading or editing code.",
        args: {
          goal: toolHelper.schema.string().describe("The coding task, failing test, or bug description to analyze."),
        },
        execute: executePlan,
      })
    : {
        description:
          "Invoke Alienese's multi-model orchestration engine (EmbeddingGemma 2 retrieval ranking + mini-Jev decision controller) to analyze repository context, rank files, and select the optimal grounded next action. Call this tool before reading or editing code.",
        args: {
          goal: {
            type: "string",
            description: "The coding task, failing test, or bug description to analyze.",
          },
        },
        execute: executePlan,
      };

  return {
    tool: {
      alienese_plan: planToolDef,
    },

    "tool.execute.before": async (input, output) => {
      const toolName = input?.tool || "unknown";
      const callID = input?.callID || "";
      const sessionID = input?.sessionID || "default";
      const targetDir = input?.directory || workspaceDir;
      const args = output?.args || {};

      // 1. Strict workspace containment check for demo mode
      if (process.env.ALIENESE_EXPECTED_WORKSPACE) {
        const expected = path.resolve(process.env.ALIENESE_EXPECTED_WORKSPACE);
        const actual = path.resolve(targetDir);
        if (actual !== expected) {
          throw new Error(
            `[Alienese Workspace Mismatch] Expected workspace '${expected}' but received '${actual}'. Rejecting uncontained demo operation.`
          );
        }
      }

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

      const decision = await invokeBridge(pythonCmd, targetDir, bridgeArgs, 15000);
      if (!decision) {
        // Fail-closed for protected filesystem and execution tools
        if (PROTECTED_TOOLS.has(toolName)) {
          console.error(
            `[Alienese Fail-Closed] Bridge failed/timed out on protected tool '${toolName}'. Execution blocked.`
          );
          throw new Error(
            `[Alienese Policy Rejection] Bridge execution failed or timed out. Protected tool '${toolName}' blocked (fail-closed).`
          );
        }
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

      const result = await invokeBridge(pythonCmd, targetDir, bridgeArgs, 15000);
      if (!result) {
        return;
      }

      const digestShort = (result.state_digest || "").slice(0, 16);
      console.error(
        `[Alienese State] Step ${result.step} | Digest: ${digestShort}... | Evidence: ${result.evidence_count} items`
      );

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
