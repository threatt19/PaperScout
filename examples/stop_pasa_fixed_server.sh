#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="${PAPERSCOUT_PROJECT_DIR:-$HOME/PaperScout}"
PID_FILE="$PROJECT_DIR/recipe/paper_search/local_server/pasa_corpus_server.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "No fixed-corpus server PID file."
  exit 0
fi
pid="$(cat "$PID_FILE")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "Stopped PaperScout fixed-corpus server PID $pid"
fi
rm -f "$PID_FILE"
