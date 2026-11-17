#!/usr/bin/env python3
"""Build a resumable PaSa metadata store and Milvus BM25 collection."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from pasa_zip import read_zip_entry


TITLE_MAX_BYTES = 8_000
ABSTRACT_MAX_BYTES = 48_000
SEARCH_TEXT_MAX_BYTES = 60_000


def keep_letters(value: object) -> str:
    return "".join(char for char in str(value or "") if char.isalpha()).lower()


def normalize_arxiv_id(value: object) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("arxiv:"):
        text = text.split(":", 1)[1].strip()
    if "v" in text and text.rsplit("v", 1)[-1].isdigit():
        text = text.rsplit("v", 1)[0]
    return text


def truncate_utf8(value: object, max_bytes: int) -> str:
    data = str(value or "").encode("utf-8")
    if len(data) <= max_bytes:
        return data.decode("utf-8")
    return data[:max_bytes].decode("utf-8", errors="ignore")


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def create_sqlite(path: Path, rebuild: bool) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild and path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            title_norm TEXT NOT NULL,
            abstract TEXT NOT NULL,
            zip_name TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_title_norm ON papers(title_norm)")
    conn.commit()
    return conn


def create_milvus(uri: str, collection: str, rebuild: bool):
    try:
        from pymilvus import DataType, Function, FunctionType, MilvusClient
    except ImportError as exc:
        raise SystemExit("pymilvus is missing; run setup_fixed_corpus.sh") from exc

    client = MilvusClient(uri=uri)
    exists = client.has_collection(collection_name=collection)
    if rebuild and exists:
        print(f"Dropping old Milvus collection: {collection}", flush=True)
        client.drop_collection(collection_name=collection)
        exists = False

    if not exists:
        print(f"Creating Milvus BM25 collection: {collection}", flush=True)
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="arxiv_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=TITLE_MAX_BYTES)
        schema.add_field(field_name="abstract", datatype=DataType.VARCHAR, max_length=ABSTRACT_MAX_BYTES)
        schema.add_field(
            field_name="search_text",
            datatype=DataType.VARCHAR,
            max_length=SEARCH_TEXT_MAX_BYTES,
            enable_analyzer=True,
            analyzer_params={"type": "standard"},
        )
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="search_text_bm25",
                input_field_names=["search_text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
        )
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        client.create_collection(
            collection_name=collection,
            schema=schema,
            index_params=index_params,
            consistency_level="Bounded",
        )
    return client


def make_milvus_row(arxiv_id: str, title: str, abstract: str) -> dict:
    title = truncate_utf8(title, TITLE_MAX_BYTES)
    abstract = truncate_utf8(abstract, ABSTRACT_MAX_BYTES)
    search_text = truncate_utf8(f"{title}\n{abstract}", SEARCH_TEXT_MAX_BYTES)
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "search_text": search_text,
    }


def chunks(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--sqlite-path", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--collection", default="paperscout_pasa_bm25")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    database_dir = args.database_dir.expanduser().resolve()
    index_path = database_dir / "id2paper.json"
    zip_path = database_dir / "cs_paper_2nd.zip"
    for required in (index_path, zip_path):
        if not required.is_file():
            raise SystemExit(f"Missing required file: {required}")

    args.sqlite_path = args.sqlite_path.expanduser().resolve()
    args.state_path = args.state_path.expanduser().resolve()
    args.state_path.parent.mkdir(parents=True, exist_ok=True)
    if args.rebuild and args.state_path.exists():
        args.state_path.unlink()

    print(f"Loading ID/title map: {index_path}", flush=True)
    id2title_raw = json.loads(index_path.read_text(encoding="utf-8"))
    id2title = {
        normalize_arxiv_id(arxiv_id): str(title or "")
        for arxiv_id, title in id2title_raw.items()
        if normalize_arxiv_id(arxiv_id)
    }
    del id2title_raw
    title_to_ids: dict[str, list[str]] = defaultdict(list)
    for arxiv_id, title in id2title.items():
        title_to_ids[keep_letters(title)].append(arxiv_id)

    conn = create_sqlite(args.sqlite_path, args.rebuild)
    client = create_milvus(args.milvus_uri, args.collection, args.rebuild)

    state = {"next_zip_index": 0, "zip_complete": False, "missing_complete": False}
    if args.state_path.exists():
        state.update(json.loads(args.state_path.read_text(encoding="utf-8")))
        print(f"Resuming from ZIP entry {state['next_zip_index']:,}", flush=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        next_index = int(state["next_zip_index"])
        sql_rows: list[tuple[str, str, str, str, str]] = []
        milvus_rows: list[dict] = []
        batch_last_index = next_index
        started = time.time()

        def flush_batch(last_index: int) -> None:
            nonlocal sql_rows, milvus_rows
            if not milvus_rows:
                atomic_json(
                    args.state_path,
                    {**state, "next_zip_index": last_index, "zip_complete": False},
                )
                return
            conn.executemany(
                "INSERT OR REPLACE INTO papers(arxiv_id,title,title_norm,abstract,zip_name) "
                "VALUES(?,?,?,?,?)",
                sql_rows,
            )
            conn.commit()
            client.upsert(collection_name=args.collection, data=milvus_rows)
            atomic_json(
                args.state_path,
                {**state, "next_zip_index": last_index, "zip_complete": False},
            )
            sql_rows = []
            milvus_rows = []

        if not state.get("zip_complete"):
            for index in range(next_index, len(names)):
                zip_name = names[index]
                matched_ids = title_to_ids.get(zip_name, [])
                if matched_ids:
                    try:
                        document = json.loads(read_zip_entry(archive, zip_name).decode("utf-8"))
                    except Exception as exc:
                        print(f"WARNING: cannot read {zip_name!r}: {exc}", file=sys.stderr, flush=True)
                        document = {}
                    abstract = str(document.get("abstract") or "")
                    for arxiv_id in matched_ids:
                        title = id2title[arxiv_id]
                        sql_rows.append((arxiv_id, title, keep_letters(title), abstract, zip_name))
                        milvus_rows.append(make_milvus_row(arxiv_id, title, abstract))
                batch_last_index = index + 1
                if len(milvus_rows) >= args.batch_size or (index + 1) % 5000 == 0:
                    flush_batch(batch_last_index)
                    elapsed = max(time.time() - started, 0.001)
                    print(
                        f"ZIP {index + 1:,}/{len(names):,}; "
                        f"SQLite rows {conn.execute('SELECT COUNT(*) FROM papers').fetchone()[0]:,}; "
                        f"{(index + 1 - next_index) / elapsed:.1f} entries/s",
                        flush=True,
                    )
            flush_batch(len(names))
            state.update({"next_zip_index": len(names), "zip_complete": True})
            atomic_json(args.state_path, state)

    if not state.get("missing_complete"):
        print("Adding title-only records without a local ZIP document...", flush=True)
        existing = {row[0] for row in conn.execute("SELECT arxiv_id FROM papers")}
        missing_ids = [arxiv_id for arxiv_id in id2title if arxiv_id not in existing]
        for part in chunks(missing_ids, args.batch_size):
            sql_rows = []
            milvus_rows = []
            for arxiv_id in part:
                title = id2title[arxiv_id]
                sql_rows.append((arxiv_id, title, keep_letters(title), "", ""))
                milvus_rows.append(make_milvus_row(arxiv_id, title, ""))
            conn.executemany(
                "INSERT OR REPLACE INTO papers(arxiv_id,title,title_norm,abstract,zip_name) VALUES(?,?,?,?,?)",
                sql_rows,
            )
            conn.commit()
            client.upsert(collection_name=args.collection, data=milvus_rows)
        state["missing_complete"] = True
        atomic_json(args.state_path, state)

    print("Flushing and loading Milvus collection...", flush=True)
    client.flush(collection_name=args.collection)
    client.load_collection(collection_name=args.collection)
    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    local = conn.execute("SELECT COUNT(*) FROM papers WHERE zip_name <> ''").fetchone()[0]
    stats = client.get_collection_stats(collection_name=args.collection)
    print(f"DONE: metadata={total:,}, local_fulltext={local:,}, Milvus={stats}", flush=True)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
