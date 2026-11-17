#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PAPERSCOUT_PROJECT_DIR:-$HOME/PaperScout}"
DATABASE_DIR="${PASA_DATABASE_DIR:-$HOME/data/pasa_original/paper_database}"
PASA_DATA_DIR="${PASA_DATA_DIR:-$HOME/data/pasa}"
GOLD_PATH="${PASA_GOLD_PATH:-$HOME/data/pasa_raw/RealScholarQuery_test.jsonl}"
LOCAL_DIR="$PROJECT_DIR/recipe/paper_search/local_server"
MILVUS_DIR="$LOCAL_DIR/milvus"
SQLITE_PATH="$LOCAL_DIR/pasa_corpus.sqlite3"
STATE_PATH="$LOCAL_DIR/pasa_corpus_build_state.json"
COLLECTION="${PAPERSCOUT_MILVUS_COLLECTION:-paperscout_pasa_bm25}"
PIP_MIRROR="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

fail_report() {
  code=$?
  echo >&2
  echo "FAILED at line $1 (exit $code). Fix that error and run this same command again; completed import batches are preserved." >&2
  exit "$code"
}
trap 'fail_report $LINENO' ERR

echo "[1/7] Activate paperscout environment and verify files"
if [[ "${CONDA_DEFAULT_ENV:-}" != "paperscout" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate paperscout
  else
    echo "Conda is unavailable and paperscout is not active." >&2
    exit 1
  fi
fi
cd "$PROJECT_DIR"
for path in \
  "$DATABASE_DIR/id2paper.json" \
  "$DATABASE_DIR/cs_paper_2nd.zip" \
  "$PASA_DATA_DIR/train.parquet" \
  "$PASA_DATA_DIR/test.parquet" \
  "$GOLD_PATH"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

echo "[2/7] Install the pinned Milvus Python client (using mirror: $PIP_MIRROR)"
python -m pip install -q -i "$PIP_MIRROR" "pymilvus>=2.5,<2.6" fastapi uvicorn

echo "[3/7] Create or verify 200-train / 50-validation data"
python recipe/paper_search/local_server/prepare_experiment_split.py \
  --data-dir "$PASA_DATA_DIR"

echo "[4/7] Start Milvus Standalone"
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed; installing Ubuntu's Docker and Compose packages."
  sudo apt-get update
  sudo apt-get install -y docker.io docker-compose-v2
  sudo systemctl enable --now docker
fi
if ! docker compose version >/dev/null 2>&1 && ! sudo docker compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is missing; installing it."
  sudo apt-get update
  sudo apt-get install -y docker-compose-v2
fi
if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  echo "Docker needs sudo; you may be asked for the server password once."
  sudo -v
  DOCKER=(sudo docker)
fi
"${DOCKER[@]}" compose -f "$MILVUS_DIR/docker-compose.yml" up -d

echo "Waiting for Milvus on 127.0.0.1:19530..."
python - <<'PY'
import socket, time
for _ in range(240):
    try:
        with socket.create_connection(("127.0.0.1", 19530), timeout=1):
            print("Milvus port is ready.")
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit("Milvus did not start within 4 minutes")
PY

echo "[5/7] Build/resume the 569k-paper metadata store and Milvus BM25 index"
python -u recipe/paper_search/local_server/prepare_pasa_corpus.py \
  --database-dir "$DATABASE_DIR" \
  --sqlite-path "$SQLITE_PATH" \
  --state-path "$STATE_PATH" \
  --collection "$COLLECTION"

echo "[6/7] Measure RealScholarQuery corpus coverage"
mkdir -p "$PROJECT_DIR/eval_outputs"
python recipe/paper_search/local_server/report_fixed_corpus_coverage.py \
  --sqlite-path "$SQLITE_PATH" \
  --gold-path "$GOLD_PATH" \
  --output "$PROJECT_DIR/eval_outputs/fixed_corpus_coverage.json"

echo "[7/7] Switch port 8765 to the fixed-corpus server and test it"
bash examples/start_pasa_fixed_server.sh
curl -fsS --get --data-urlencode 'q=sparse coding' --data 'top_k=3' \
  http://127.0.0.1:8765/search \
  >/tmp/paperscout_search_smoke.json
python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path("/tmp/paperscout_search_smoke.json").read_text(encoding="utf-8"))
if not rows:
    raise SystemExit("Search smoke test returned no papers")
print("Search smoke test:")
for row in rows:
    print(f"  {row['arxiv_id']}  score={row['score']:.4f}  {row['title'][:100]}")
PY

echo
echo "ALL DONE"
echo "The server now searches one fixed corpus: 569k metadata records / about 568k locally readable records."
echo "Training questions: 200; validation questions: 50; final RealScholarQuery test questions: 50."
echo "Coverage report: $PROJECT_DIR/eval_outputs/fixed_corpus_coverage.json"
echo "Server log: $PROJECT_DIR/pasa_corpus_server.log"
