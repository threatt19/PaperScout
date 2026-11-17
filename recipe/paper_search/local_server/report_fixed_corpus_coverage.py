#!/usr/bin/env python3
"""Report the maximum RealScholarQuery recall allowed by the fixed corpus."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path


def normalize(value: object) -> str:
    text = re.sub(r"^arxiv:\s*", "", str(value or "").strip(), flags=re.IGNORECASE)
    return re.sub(r"v\d+$", "", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite-path", type=Path, required=True)
    parser.add_argument("--gold-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.sqlite_path.expanduser())
    all_ids = {row[0] for row in conn.execute("SELECT arxiv_id FROM papers")}
    local_ids = {row[0] for row in conn.execute("SELECT arxiv_id FROM papers WHERE zip_name <> ''")}
    conn.close()

    rows = [
        json.loads(line)
        for line in args.gold_path.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mentions = 0
    metadata_hits = 0
    local_hits = 0
    zero_metadata = []
    zero_local = []
    per_query = []
    for row in rows:
        ids = [normalize(item) for item in row.get("answer_arxiv_id", []) if normalize(item)]
        metadata = sum(item in all_ids for item in ids)
        local = sum(item in local_ids for item in ids)
        mentions += len(ids)
        metadata_hits += metadata
        local_hits += local
        qid = row.get("qid", "")
        if ids and metadata == 0:
            zero_metadata.append(qid)
        if ids and local == 0:
            zero_local.append(qid)
        per_query.append(
            {
                "qid": qid,
                "gold": len(ids),
                "metadata_covered": metadata,
                "local_fulltext_covered": local,
            }
        )

    report = {
        "queries": len(rows),
        "corpus_metadata_papers": len(all_ids),
        "corpus_local_fulltexts": len(local_ids),
        "gold_paper_mentions": mentions,
        "metadata_covered_mentions": metadata_hits,
        "metadata_oracle_recall": metadata_hits / mentions if mentions else 0.0,
        "local_fulltext_covered_mentions": local_hits,
        "local_fulltext_oracle_recall": local_hits / mentions if mentions else 0.0,
        "queries_with_zero_metadata_coverage": len(zero_metadata),
        "queries_with_zero_local_fulltext_coverage": len(zero_local),
        "zero_metadata_qids": zero_metadata,
        "zero_local_fulltext_qids": zero_local,
        "per_query": per_query,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"queries: {report['queries']}")
    print(f"corpus metadata papers: {report['corpus_metadata_papers']}")
    print(f"corpus local fulltexts: {report['corpus_local_fulltexts']}")
    print(f"gold paper mentions: {mentions}")
    print(f"metadata covered: {metadata_hits} ({report['metadata_oracle_recall']:.4f})")
    print(f"local fulltext covered: {local_hits} ({report['local_fulltext_oracle_recall']:.4f})")
    print(f"zero-metadata queries: {len(zero_metadata)}")
    print(f"zero-fulltext queries: {len(zero_local)}")
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
