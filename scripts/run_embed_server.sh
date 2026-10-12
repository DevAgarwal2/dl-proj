#!/usr/bin/env bash
# Start / stop the local EmbeddingGemma-2 embedding server used by the RAG layer.
#
#   bash scripts/run_embed_server.sh start
#   bash scripts/run_embed_server.sh stop
#
# Override with env vars:
#   OPENMED_EMBED_GGUF   path to the embedding .gguf (default: <repo>/models/embeddinggemma-2-Q8_0.gguf)
#   OPENMED_EMBED_BIN    llama-server binary (default: llama-server on PATH)
#   OPENMED_EMBED_PORT   server port (default 8081)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${OPENMED_EMBED_PORT:-8081}"
MODEL="${OPENMED_EMBED_GGUF:-$ROOT/models/embeddinggemma-2-Q8_0.gguf}"
BIN="${OPENMED_EMBED_BIN:-llama-server}"
LOG="${OPENMED_EMBED_LOG:-/tmp/openmed-embed-server.log}"

case "${1:-start}" in
  start)
    if lsof -ti:"${PORT}" >/dev/null 2>&1; then
      echo "Port ${PORT} already in use (embedding server likely running)."
      exit 0
    fi
    if ! command -v "${BIN}" >/dev/null 2>&1 && [[ ! -x "${BIN}" ]]; then
      echo "llama-server not found (${BIN}). Set OPENMED_EMBED_BIN." >&2
      exit 1
    fi
    if [[ ! -f "${MODEL}" ]]; then
      echo "Model not found: ${MODEL}" >&2
      echo "Set OPENMED_EMBED_GGUF to your embedding .gguf path." >&2
      exit 1
    fi
    echo "Starting embedding server on port ${PORT} with ${MODEL}"
    nohup "${BIN}" \
      -m "${MODEL}" \
      --embedding --pooling mean \
      --port "${PORT}" \
      --host 127.0.0.1 \
      -ngl 99 -c 8192 -b 2048 -ub 1024 \
      > "${LOG}" 2>&1 &
    sleep 4
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
      echo "Embedding server healthy at http://127.0.0.1:${PORT}/v1 (log: ${LOG})"
    else
      echo "Server started but health check failed; see ${LOG}" >&2
      exit 1
    fi
    ;;
  stop)
    if lsof -ti:"${PORT}" >/dev/null 2>&1; then
      kill "$(lsof -ti:"${PORT}")"
      echo "Stopped embedding server on port ${PORT}"
    else
      echo "No server on port ${PORT}"
    fi
    ;;
  *)
    echo "Usage: $0 {start|stop}" >&2
    exit 2
    ;;
esac
