#!/usr/bin/env bash
# ==============================================================================
# Alienese — OpenCode Browser Demo Launcher
#
# Launches OpenCode Web (v1.19.0) with Alienese plugin active, bound to
# NVIDIA Nemotron 3 Super on Nebius Token Factory, inside an isolated
# disposable workspace (/tmp/alienese_opencode_demo), alongside the
# Live Visual Dashboard (http://127.0.0.1:4097).
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="/tmp/alienese_opencode_demo"
PORT="4096"
HOST="127.0.0.1"
DASHBOARD_PORT="4097"
LOG_FILE="$DEMO_DIR/opencode_web.log"
PID_FILE="$DEMO_DIR/opencode_web.pid"
DASHBOARD_PID_FILE="$DEMO_DIR/dashboard.pid"
DASHBOARD_LOG_FILE="$DEMO_DIR/dashboard.log"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

# Helper: Find process ID listening on port 4096 strictly belonging to demo workspace
get_server_pid() {
  "$VENV_PYTHON" "$REPO_ROOT/scripts/prepare_opencode_demo.py" find-pid --port "$PORT" --workspace "$DEMO_DIR" 2>/dev/null || true
}

get_dashboard_pid() {
  "$VENV_PYTHON" "$REPO_ROOT/scripts/prepare_opencode_demo.py" find-dashboard-pid --port "$DASHBOARD_PORT" --workspace "$DEMO_DIR" 2>/dev/null || true
}

stop_dashboard() {
  "$VENV_PYTHON" "$REPO_ROOT/scripts/prepare_opencode_demo.py" stop-dashboard --port "$DASHBOARD_PORT" --workspace "$DEMO_DIR" >/dev/null 2>&1 || true
}

start_dashboard() {
  "$VENV_PYTHON" "$REPO_ROOT/scripts/prepare_opencode_demo.py" start-dashboard --port "$DASHBOARD_PORT" --workspace "$DEMO_DIR" >/dev/null 2>&1 || true
}

# Helper: Stop server listening on port 4096
stop_server() {
  echo "Stopping server on port $PORT..."
  "$VENV_PYTHON" "$REPO_ROOT/scripts/prepare_opencode_demo.py" stop-server --port "$PORT" --workspace "$DEMO_DIR" >/dev/null 2>&1 || true
  rm -f "$PID_FILE"
  stop_dashboard
  echo "Server and dashboard stopped."
}

# Helper: Show demo status
show_status() {
  echo "=================================================================="
  echo " Alienese OpenCode Demo Status"
  echo "=================================================================="
  local pid
  pid=$(get_server_pid)
  if [ -n "$pid" ]; then
    local proc_cwd="unknown"
    if [ -d "/proc/$pid" ]; then
      proc_cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || echo "unknown")
    fi
    echo " OpenCode Web:     RUNNING (PID: $pid)"
    echo " Server Directory: $proc_cwd"
    echo " OpenCode URL:     http://$HOST:$PORT/"
  else
    echo " OpenCode Web:     STOPPED"
  fi

  local dpid
  dpid=$(get_dashboard_pid)
  if [ -n "$dpid" ]; then
    echo " Visual Dashboard: RUNNING (PID: $dpid)"
    echo " Dashboard URL:    http://$HOST:$DASHBOARD_PORT/"
  else
    echo " Visual Dashboard: STOPPED"
  fi

  echo "------------------------------------------------------------------"
  "$VENV_PYTHON" "$REPO_ROOT/scripts/prepare_opencode_demo.py" status --workspace "$DEMO_DIR"
  echo "=================================================================="
}

# Parse CLI options
DO_RESET=0
DO_STATUS=0
DO_STOP=0
FOREGROUND=0
THREE_MODEL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --three-model)
      THREE_MODEL=1
      shift
      ;;
    --reset)
      DO_RESET=1
      shift
      ;;
    --status)
      DO_STATUS=1
      shift
      ;;
    --stop)
      DO_STOP=1
      shift
      ;;
    --foreground|-f)
      FOREGROUND=1
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--three-model] [--reset] [--status] [--stop] [--foreground]"
      echo ""
      echo "Options:"
      echo "  --three-model Enable three-model runtime: EmbeddingGemma 2 (Retrieval) + mini-Jev (Decision) + Nemotron (Generation)"
      echo "  --reset       Reset demo workspace to pristine failing bug state and restart server"
      echo "  --status      Check if demo server is running and inspect test/trace status"
      echo "  --stop        Stop running OpenCode web server and dashboard"
      echo "  --foreground  Run server in foreground instead of detached background"
      echo "  --help, -h    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run with --help for usage."
      exit 1
      ;;
  esac
done

if [ "$DO_STATUS" -eq 1 ]; then
  show_status
  exit 0
fi

if [ "$DO_STOP" -eq 1 ]; then
  stop_server
  exit 0
fi

# Ensure Python venv exists
if [ ! -f "$VENV_PYTHON" ]; then
  echo "Error: Python virtual environment not found at $VENV_PYTHON"
  exit 1
fi

# Ensure opencode executable exists
OPENCODE_BIN=""
if [ -f "$HOME/.opencode/bin/opencode" ]; then
  OPENCODE_BIN="$HOME/.opencode/bin/opencode"
elif command -v opencode >/dev/null 2>&1; then
  OPENCODE_BIN="$(command -v opencode)"
else
  echo "Error: opencode executable not found at ~/.opencode/bin/opencode or PATH."
  exit 1
fi

# If reset requested, stop existing server first
if [ "$DO_RESET" -eq 1 ]; then
  stop_server
  echo "Resetting demo workspace to pristine failing state..."
  "$VENV_PYTHON" "$REPO_ROOT/scripts/prepare_opencode_demo.py" setup --reset --workspace "$DEMO_DIR"
fi

# Check if server is already running
EXISTING_PID=$(get_server_pid)
if [ -n "$EXISTING_PID" ]; then
  PROC_CWD="unknown"
  if [ -d "/proc/$EXISTING_PID" ]; then
    PROC_CWD=$(readlink -f "/proc/$EXISTING_PID/cwd" 2>/dev/null || echo "unknown")
  fi
  if [ "$PROC_CWD" = "$DEMO_DIR" ]; then
    echo "=================================================================="
    echo " Alienese OpenCode Demo is ALREADY RUNNING!"
    echo "=================================================================="
    echo " OpenCode PID:     $EXISTING_PID"
    echo " Workspace:        $DEMO_DIR"
    echo " OpenCode Web:     http://$HOST:$PORT/"
    echo " Visual Dashboard: http://$HOST:$DASHBOARD_PORT/"
    echo " Model:            nebius/nvidia/nemotron-3-super-120b-a12b"
    echo "------------------------------------------------------------------"
    echo " To reset for a fresh recording take: $0 --reset"
    echo " To stop the server:                  $0 --stop"
    echo " To view live logs:                   tail -f $LOG_FILE"
    echo "=================================================================="
    exit 0
  else
    echo "Warning: Port $PORT is occupied by PID $EXISTING_PID running in: $PROC_CWD"
    echo "Run '$0 --stop' to stop it first, or kill PID $EXISTING_PID."
    exit 1
  fi
fi

# Ensure workspace exists
if [ ! -d "$DEMO_DIR" ]; then
  echo "Initializing disposable demo workspace at $DEMO_DIR..."
  "$VENV_PYTHON" "$REPO_ROOT/scripts/prepare_opencode_demo.py" setup --workspace "$DEMO_DIR"
fi

# Securely load NEBIUS_TOKEN_FACTORY_KEY from .env using Python (never print/log)
EXTRACTED_KEY=$("$VENV_PYTHON" "$REPO_ROOT/scripts/prepare_opencode_demo.py" extract-key || true)
if [ -z "$EXTRACTED_KEY" ]; then
  echo "Error: Could not extract NEBIUS_TOKEN_FACTORY_KEY from $REPO_ROOT/.env"
  exit 1
fi
export NEBIUS_TOKEN_FACTORY_KEY="$EXTRACTED_KEY"
export NEBIUS_API_KEY="$EXTRACTED_KEY"
unset EXTRACTED_KEY

# Set Alienese runtime environment
export ALIENESE_PYTHON="$VENV_PYTHON"
export ALIENESE_READ_ONLY_TESTS="1"
export ALIENESE_DEMO_MODE="1"
export ALIENESE_EXPECTED_WORKSPACE="$DEMO_DIR"
export XDG_DATA_HOME="$DEMO_DIR/.xdg/data"
export XDG_CONFIG_HOME="$DEMO_DIR/.xdg/config"
export XDG_STATE_HOME="$DEMO_DIR/.xdg/state"
export PATH="$(dirname "$OPENCODE_BIN"):$REPO_ROOT/.venv/bin:$PATH"

# Check or start three-model server if requested
if [ "$THREE_MODEL" -eq 1 ]; then
  echo "Checking local model server on port 8001 (EmbeddingGemma 2 + mini-Jev)..."
  if ! curl -s "http://127.0.0.1:8001/health" 2>/dev/null | grep -q '"status":"ok"'; then
    echo "Starting local model server on port 8001..."
    "$REPO_ROOT/.venv-models/bin/python" "$REPO_ROOT/scripts/serve_local_models.py" --port 8001 > "$DEMO_DIR/local_models.log" 2>&1 &
    sleep 3
  fi
  export ALIENESE_THREE_MODEL="1"
  echo "Three-model architecture active: EmbeddingGemma 2 + mini-Jev + Nemotron 3 Super"
fi

# Start live visual dashboard on port 4097
start_dashboard

# Launch OpenCode Web
cd "$DEMO_DIR"

if [ "$FOREGROUND" -eq 1 ]; then
  echo "Starting OpenCode Web in foreground..."
  exec "$OPENCODE_BIN" web --port "$PORT" --hostname "$HOST"
fi

# Start detached background daemon
START_JSON=$("$VENV_PYTHON" "$REPO_ROOT/scripts/prepare_opencode_demo.py" start-server --workspace "$DEMO_DIR" --port "$PORT" --host "$HOST")
SERVER_PID=$(echo "$START_JSON" | grep -o '"pid": [0-9]*' | awk '{print $2}' || true)
if [ -z "$SERVER_PID" ]; then
  SERVER_PID=$(get_server_pid)
fi

# Wait up to 5 seconds for server to be responsive
READY=0
for i in {1..10}; do
  if curl -s -I "http://$HOST:$PORT/" 2>/dev/null | grep -q "200 OK"; then
    READY=1
    break
  fi
  sleep 0.5
done

if [ "$READY" -ne 1 ]; then
  echo "Error: Server failed to respond on http://$HOST:$PORT/ within 5s."
  if [ -f "$LOG_FILE" ]; then
    echo "Log output:"
    cat "$LOG_FILE"
  fi
  exit 1
fi

echo "=================================================================="
echo " Alienese — OpenCode Browser Demo is READY FOR RECORDING!"
echo "=================================================================="
echo " 1. OpenCode Browser:   http://$HOST:$PORT/"
echo " 2. Live Dashboard:     http://$HOST:$DASHBOARD_PORT/"
echo " 3. Project Workspace:  $DEMO_DIR"
echo " 4. Default Model:      nebius/nvidia/nemotron-3-super-120b-a12b"
echo " 5. Provider:           Nebius Token Factory"
echo " 6. Alienese Plugin:    plugins/opencode/alienese.js (ACTIVE)"
echo " 7. Test Policy:        ALIENESE_READ_ONLY_TESTS=1 (ENFORCED)"
if [ "$THREE_MODEL" -eq 1 ]; then
  echo " 8. Three-Model Engine: ACTIVE (EmbeddingGemma 2 + mini-Jev + Nemotron 3 Super)"
else
  echo " 8. Mode:               STANDALONE NEMOTRON (Use --three-model for full 3-model demo)"
fi
echo " 9. Server PID:         $SERVER_PID"
echo "------------------------------------------------------------------"
echo " READY-TO-PASTE PROMPT:"
echo ' "Run pytest tests/test_session.py to observe the failure, investigate auth/session_manager.py, fix the session refresh bug, and verify tests pass."'
echo "------------------------------------------------------------------"
echo " MONITORING & LOGS:"
echo " • Live Companion:      http://$HOST:$DASHBOARD_PORT/"
echo " • Live Plugin Output:  tail -f $LOG_FILE"
echo " • Live Audit Trace:    tail -f $DEMO_DIR/.alienese/opencode_trace.jsonl"
echo "------------------------------------------------------------------"
echo " CONTROLS:"
echo " • Reset for next take: ./scripts/launch_opencode_demo.sh --reset"
echo " • Stop server:         ./scripts/launch_opencode_demo.sh --stop"
echo " • Check status:        ./scripts/launch_opencode_demo.sh --status"
echo "=================================================================="
