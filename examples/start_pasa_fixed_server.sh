#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PAPERSCOUT_PROJECT_DIR:-$HOME/PaperScout}"
LOCAL_DIR="$PROJECT_DIR/recipe/paper_search/local_server"
PID_FILE="$LOCAL_DIR/pasa_corpus_server.pid"
LOG_FILE="$PROJECT_DIR/pasa_corpus_server.log"
PORT="${PAPERSCOUT_SERVER_PORT:-8765}"

cd "$PROJECT_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    kill "$old_pid"
    for _ in {1..20}; do
      kill -0 "$old_pid" 2>/dev/null || break
      sleep 0.25
    done
  fi
  rm -f "$PID_FILE"
fi

if command -v lsof >/dev/null 2>&1; then
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" == *"recipe/paper_search/local_server/"*"server.py"* ]]; then
      echo "Stopping old PaperScout server PID $pid"
      kill "$pid"
    else
      echo "Port $PORT is used by an unrelated process: PID $pid ($cmdline)" >&2
      exit 1
    fi
  done < <(lsof -t -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | sort -u)
fi

for _ in {1..20}; do
  if ! command -v lsof >/dev/null 2>&1 || \
     ! lsof -t -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

export PAPERSCOUT_MILVUS_URI="${PAPERSCOUT_MILVUS_URI:-http://127.0.0.1:19530}"
export PAPERSCOUT_MILVUS_COLLECTION="${PAPERSCOUT_MILVUS_COLLECTION:-paperscout_pasa_bm25}"
export PAPERSCOUT_SQLITE_PATH="${PAPERSCOUT_SQLITE_PATH:-$LOCAL_DIR/pasa_corpus.sqlite3}"
export PAPERSCOUT_ZIP_PATH="${PAPERSCOUT_ZIP_PATH:-$HOME/data/pasa_original/paper_database/cs_paper_2nd.zip}"
export PAPERSCOUT_SERVER_PORT="$PORT"

nohup "$(command -v python)" -u \
  recipe/paper_search/local_server/pasa_corpus_server.py \
  >"$LOG_FILE" 2>&1 </dev/null &
server_pid=$!
echo "$server_pid" > "$PID_FILE"

for _ in {1..60}; do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/tmp/paperscout_health.json 2>/dev/null; then
    cat /tmp/paperscout_health.json
    echo
    echo "Server PID: $server_pid"
    echo "Log: $LOG_FILE"
    exit 0
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -n 100 "$LOG_FILE" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "Server did not become healthy; last log lines:" >&2
tail -n 100 "$LOG_FILE" >&2 || true
exit 1
