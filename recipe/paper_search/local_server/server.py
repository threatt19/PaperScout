import json
import re
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from rank_bm25 import BM25Okapi


DATA_PATH = Path(__file__).parent / "papers.jsonl"

app = FastAPI(title="PaperScout Local Paper Server")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", (text or "").lower())


def normalize_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


# -------------------------
# Load local paper corpus
# -------------------------

papers = []

with DATA_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        paper = json.loads(line)

        paper.setdefault("arxiv_id", "")
        paper.setdefault("title", "")
        paper.setdefault("abstract", "")
        paper.setdefault("categories", "")
        paper.setdefault("authors", "")
        paper.setdefault("update_date", "")
        paper.setdefault("score", 0.0)
        paper.setdefault("references", [])

        paper["arxiv_id"] = normalize_arxiv_id(paper["arxiv_id"])

        papers.append(paper)


id2paper = {
    paper["arxiv_id"]: paper
    for paper in papers
}

title2paper = {
    paper["title"].strip().lower(): paper
    for paper in papers
}


# BM25 retrieval corpus
corpus = [
    tokenize(f"{paper['title']} {paper['abstract']}")
    for paper in papers
]

bm25 = BM25Okapi(corpus)


def public_paper(paper: dict, score: float = 0.0) -> dict:
    """Return fields expected by PaperScout's Paper model."""
    return {
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "categories": paper["categories"],
        "authors": paper["authors"],
        "update_date": paper["update_date"],
        "score": float(score),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "num_papers": len(papers),
    }


@app.get("/search")
async def search(
    q: str,
    top_k: int = 10,
    filter: Optional[str] = None,
):
    if not papers:
        return []

    query_tokens = tokenize(q)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        range(len(papers)),
        key=lambda i: float(scores[i]),
        reverse=True,
    )

    results = []

    for i in ranked[:top_k]:
        results.append(
            public_paper(
                papers[i],
                score=float(scores[i]),
            )
        )

    return results


@app.get("/query")
async def query(
    filter: str,
    top_k: int = 10,
):
    """
    Minimal compatibility endpoint.

    For the smoke test we perform simple lexical matching against
    arXiv ID/title/abstract/categories.
    """
    tokens = tokenize(filter)

    if not tokens:
        return []

    matched = []

    for paper in papers:
        text = tokenize(
            f"{paper['arxiv_id']} "
            f"{paper['title']} "
            f"{paper['abstract']} "
            f"{paper['categories']}"
        )

        text_set = set(text)

        if all(token in text_set for token in tokens):
            matched.append(public_paper(paper))

        if len(matched) >= top_k:
            break

    return matched


@app.get("/lookup")
async def lookup(title: str):
    key = title.strip().lower()

    # Exact match first
    if key in title2paper:
        return [public_paper(title2paper[key])]

    # Fallback: substring title matching
    results = []

    for paper in papers:
        if key in paper["title"].lower():
            results.append(public_paper(paper))

    return results[:10]


@app.get("/paper/{arxiv_id}")
async def get_paper(arxiv_id: str):
    arxiv_id = normalize_arxiv_id(arxiv_id)

    paper = id2paper.get(arxiv_id)

    if paper is None:
        return None

    return public_paper(paper)


@app.get("/paper/{arxiv_id}/fulltext")
async def get_fulltext(
    arxiv_id: str,
    download: bool = False,
):
    """
    Small-pool reproduction does not currently store full text.

    Return abstract as a minimal section structure so callers do not
    fail merely because the original private full-text database is absent.
    """
    arxiv_id = normalize_arxiv_id(arxiv_id)

    paper = id2paper.get(arxiv_id)

    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    references = {}

    for i, ref in enumerate(paper.get("references", [])):
        if isinstance(ref, str):
            title = ref
        else:
            title = ref.get("title", "")

        if title:
            references[str(i)] = {
                "title": title,
                "meta_string": title,
            }

    return {
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "sections": [
            {
                "title": "Abstract",
                "text": paper["abstract"],
            }
        ],
        "references": references,
    }


@app.get("/queue")
async def queue_stats():
    # Compatibility with ArXivClient.queue_stats()
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
    print(f"Loaded {len(papers)} papers from {DATA_PATH}")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8765,
    )
