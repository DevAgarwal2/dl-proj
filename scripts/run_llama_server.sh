#!/usr/bin/env bash
# Start / stop the local llama.cpp server hosting the LFM2.5-1.2B generator used by RAG.
#
#   bash scripts/run_llama_server.sh start
#   bash scripts/run_llama_server.sh stop
#
# Override with env vars:
#   OPENMED_GGUF   path to the .gguf model (default: <repo>/models/LFM2.5-1.2B-Instruct-Q6_K.gguf)
#   OPENMED_PORT   server port (default 8080)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${OPENMED_PORT:-8080}"
MODEL="${OPENMED_GGUF:-$ROOT/models/LFM2.5-1.2B-Instruct-Q6_K.gguf}"
LOG="${OPENMED_LLAMA_LOG:-/tmp/openmed-llama-server.log}"

case "${1:-start}" in
  start)
    if lsof -ti:"${PORT}" >/dev/null 2>&1; then
      echo "Port ${PORT} already in use (server likely running)."
      exit 0
    fi
    if [[ ! -f "${MODEL}" ]]; then
      echo "Model not found: ${MODEL}" >&2
      echo "Set OPENMED_GGUF to your .gguf path." >&2
      exit 1
    fi
    echo "Starting llama-server on port ${PORT} with ${MODEL}"
    nohup llama-server \
      -m "${MODEL}" \
      --port "${PORT}" \
      --host 127.0.0.1 \
      -ngl 99 \
      --mlock \
      --repeat-penalty 1.05 \
      > "${LOG}" 2>&1 &
    sleep 3
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
      echo "llama-server healthy at http://127.0.0.1:${PORT}/v1 (log: ${LOG})"
    else
      echo "Server started but health check failed; see ${LOG}" >&2
      exit 1
    fi
    ;;
  stop)
    if lsof -ti:"${PORT}" >/dev/null 2>&1; then
      kill "$(lsof -ti:"${PORT}")"
      echo "Stopped server on port ${PORT}"
    else
      echo "No server on port ${PORT}"
    fi
    ;;
  *)
    echo "Usage: $0 {start|stop}" >&2
    exit 2
    ;;
esac
