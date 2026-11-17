#!/usr/bin/env python3
"""PaperScout-compatible API backed by Milvus BM25 and the fixed PaSa corpus."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import zipfile
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pymilvus import MilvusClient

from pasa_zip import read_zip_entry


MILVUS_URI = os.environ.get("PAPERSCOUT_MILVUS_URI", "http://127.0.0.1:19530")
COLLECTION = os.environ.get("PAPERSCOUT_MILVUS_COLLECTION", "paperscout_pasa_bm25")
SQLITE_PATH = Path(
    os.environ.get(
        "PAPERSCOUT_SQLITE_PATH",
        Path(__file__).parent / "pasa_corpus.sqlite3",
    )
).expanduser()
ZIP_PATH = Path(
    os.environ.get(
        "PAPERSCOUT_ZIP_PATH",
        Path.home() / "data/pasa_original/paper_database/cs_paper_2nd.zip",
    )
).expanduser()
PORT = int(os.environ.get("PAPERSCOUT_SERVER_PORT", "8765"))

if not SQLITE_PATH.is_file():
    raise RuntimeError(f"Missing metadata database: {SQLITE_PATH}")
if not ZIP_PATH.is_file():
    raise RuntimeError(f"Missing paper ZIP: {ZIP_PATH}")

app = FastAPI(title="PaperScout Fixed PaSa Corpus Server")
milvus = MilvusClient(uri=MILVUS_URI)
db = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row
db_lock = threading.RLock()
archive = zipfile.ZipFile(ZIP_PATH, "r")
archive_lock = threading.RLock()


def keep_letters(value: object) -> str:
    return "".join(char for char in str(value or "") if char.isalpha()).lower()


def normalize_arxiv_id(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^arxiv:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"v\d+$", "", text)


def public_paper(record: dict | sqlite3.Row, score: float = 0.0) -> dict:
    return {
        "arxiv_id": str(record.get("arxiv_id", "") if isinstance(record, dict) else record["arxiv_id"]),
        "title": str(record.get("title", "") if isinstance(record, dict) else record["title"]),
        "abstract": str(record.get("abstract", "") if isinstance(record, dict) else record["abstract"]),
        "categories": "",
        "authors": "",
        "update_date": "",
        "score": float(score),
    }


def sqlite_one(arxiv_id: str):
    with db_lock:
        return db.execute(
            "SELECT arxiv_id,title,abstract,zip_name FROM papers WHERE arxiv_id=?",
            (normalize_arxiv_id(arxiv_id),),
        ).fetchone()


@app.get("/health")
async def health():
    with db_lock:
        total = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        local = db.execute("SELECT COUNT(*) FROM papers WHERE zip_name <> ''").fetchone()[0]
    return {
        "status": "ok",
        "backend": "milvus_bm25_fixed_pasa_corpus",
        "collection": COLLECTION,
        "num_papers": total,
        "num_local_fulltexts": local,
    }


@app.get("/search")
async def search(q: str, top_k: int = 10, filter: Optional[str] = None):
    if not q.strip():
        return []
    results = milvus.search(
        collection_name=COLLECTION,
        data=[q],
        anns_field="sparse",
        limit=max(1, min(int(top_k), 100)),
        output_fields=["arxiv_id", "title", "abstract"],
        search_params={"metric_type": "BM25"},
    )
    papers = []
    for hit in results[0]:
        entity = dict(hit.get("entity") or {})
        entity.setdefault("arxiv_id", str(hit.get("id", "")))
        papers.append(public_paper(entity, score=float(hit.get("distance", 0.0))))
    return papers


@app.get("/query")
async def query(filter: str, top_k: int = 10):
    direct_id = normalize_arxiv_id(filter)
    row = sqlite_one(direct_id)
    if row is not None:
        return [public_paper(row)]
    return await search(filter, top_k=top_k)


@app.get("/lookup")
async def lookup(title: str):
    key = keep_letters(title)
    with db_lock:
        rows = db.execute(
            "SELECT arxiv_id,title,abstract,zip_name FROM papers WHERE title_norm=? LIMIT 10",
            (key,),
        ).fetchall()
    if rows:
        return [public_paper(row) for row in rows]
    # Expansion must follow a real citation-title match. Returning lexical-search
    # neighbors here would turn one citation into ten unrelated approximate papers.
    return []


@app.get("/paper/{arxiv_id}")
async def get_paper(arxiv_id: str):
    row = sqlite_one(arxiv_id)
    return public_paper(row) if row is not None else None


@app.get("/paper/{arxiv_id}/fulltext")
async def get_fulltext(arxiv_id: str, download: bool = False):
    row = sqlite_one(arxiv_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    if not row["zip_name"]:
        raise HTTPException(status_code=404, detail="Paper has metadata but no local PaSa document")
    try:
        with archive_lock:
            document = json.loads(read_zip_entry(archive, row["zip_name"]).decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot read local document: {exc}") from exc

    sections_raw = document.get("sections") or {}
    sections = []
    reference_titles = []
    for section_title, citations in sections_raw.items():
        sections.append({"title": str(section_title), "text": ""})
        if isinstance(citations, list):
            reference_titles.extend(str(item) for item in citations if str(item).strip())

    seen = set()
    unique_titles = []
    for title in reference_titles:
        key = keep_letters(title)
        if key and key not in seen:
            seen.add(key)
            unique_titles.append(title)
    references = {
        str(index): {"title": title, "meta_string": title}
        for index, title in enumerate(unique_titles)
    }
    return {
        "arxiv_id": normalize_arxiv_id(arxiv_id),
        "title": row["title"],
        "sections": sections,
        "references": references,
    }


@app.get("/queue")
async def queue_stats():
    return {
        "queue_size": 0,
        "max_queue_size": 0,
        "batch_size": 0,
        "batch_timeout": 0.0,
        "running": True,
        "task_done": True,
        "last_exception": None,
        "submitted_total": 0,
        "dequeued_total": 0,
        "processed_total": 0,
        "batches_total": 0,
        "last_batch_size": 0,
        "last_process_age_sec": None,
    }


if __name__ == "__main__":
    info = db.execute(
        "SELECT COUNT(*), SUM(CASE WHEN zip_name <> '' THEN 1 ELSE 0 END) FROM papers"
    ).fetchone()
    print(
        f"Loaded fixed PaSa corpus: metadata={info[0]}, local_fulltext={info[1]}, "
        f"Milvus={MILVUS_URI}/{COLLECTION}",
        flush=True,
    )
    uvicorn.run(app, host="127.0.0.1", port=PORT)
