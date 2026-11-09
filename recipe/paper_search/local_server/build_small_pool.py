import argparse
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests


API_BASE = "https://api.semanticscholar.org/graph/v1"

PAPER_FIELDS = ",".join([
    "paperId",
    "title",
    "abstract",
    "authors",
    "year",
    "externalIds",
])

REFERENCE_FIELDS = ",".join([
    "title",
    "abstract",
    "authors",
    "year",
    "externalIds",
])


class SemanticScholarClient:
    def __init__(self, api_key=None, sleep=1.1):
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({
                "x-api-key": api_key,
            })

        self.sleep = sleep

    def get(self, url, params=None, retries=8):
        for attempt in range(retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=30,
                )

                if response.status_code == 200:
                    time.sleep(self.sleep)
                    return response.json()

                if response.status_code == 429:
                    wait = min(60, 2 ** attempt)
                    print(f"  rate limited, sleep {wait}s")
                    time.sleep(wait)
                    continue

                print(
                    f"  HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )

            except requests.RequestException as e:
                print(f"  request error: {e}")

            wait = min(30, 2 ** attempt)
            time.sleep(wait)

        return None

    def search(self, query, limit=10):
        data = self.get(
            f"{API_BASE}/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": PAPER_FIELDS,
            },
        )

        if not data:
            return []

        return data.get("data", [])

    def references(self, paper_id, limit=5):
        data = self.get(
            f"{API_BASE}/paper/{paper_id}/references",
            params={
                "limit": limit,
                "fields": REFERENCE_FIELDS,
            },
        )

        if not data:
            return []

        refs = []

        for item in data.get("data", []):
            cited = item.get("citedPaper")
            if cited:
                refs.append(cited)

        return refs


def normalize_arxiv_id(arxiv_id):
    if not arxiv_id:
        return None

    arxiv_id = str(arxiv_id).strip()

    # Remove "arXiv:" if it happens to occur.
    if arxiv_id.lower().startswith("arxiv:"):
        arxiv_id = arxiv_id.split(":", 1)[1]

    # Remove version suffix, e.g. 1706.03762v5.
    if "v" in arxiv_id:
        base, suffix = arxiv_id.rsplit("v", 1)
        if suffix.isdigit():
            arxiv_id = base

    return arxiv_id


def get_arxiv_id(paper):
    external_ids = paper.get("externalIds") or {}
    return normalize_arxiv_id(external_ids.get("ArXiv"))


def valid_paper(paper):
    return bool(
        paper
        and get_arxiv_id(paper)
        and paper.get("title")
        and paper.get("abstract")
    )


def convert_paper(paper):
    arxiv_id = get_arxiv_id(paper)

    authors = paper.get("authors") or []
    author_names = [
        author.get("name", "")
        for author in authors
        if author.get("name")
    ]

    year = paper.get("year")

    return {
        "arxiv_id": arxiv_id,
        "title": paper.get("title") or "",
        "abstract": paper.get("abstract") or "",
        "categories": "",
        "authors": ", ".join(author_names),
        "update_date": str(year) if year else "",
        "references": [],
    }


def extract_query(prompt):
    """
    PaSa prompt looks like:
    array([{'content': '...', 'role': 'user'}], dtype=object)
    """
    if prompt is None:
        return None

    try:
        messages = list(prompt)
    except TypeError:
        return None

    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if content:
                return str(content).strip()

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=os.path.expanduser("~/data/pasa/train.parquet"),
    )

    parser.add_argument(
        "--output",
        default="recipe/paper_search/local_server/papers.jsonl",
    )

    parser.add_argument(
        "--query-output",
        default="recipe/paper_search/local_server/queries.json",
    )

    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--papers-per-query",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--refs-per-paper",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--api-key",
        default=os.environ.get("S2_API_KEY"),
    )

    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading PaSa queries from {args.input}")

    df = pd.read_parquet(args.input)

    queries = []

    for prompt in df["prompt"]:
        query = extract_query(prompt)

        if query:
            queries.append(query)

    # Stable deduplication.
    queries = list(dict.fromkeys(queries))

    if len(queries) < args.num_queries:
        raise RuntimeError(
            f"Only found {len(queries)} valid queries, "
            f"but requested {args.num_queries}"
        )

    selected_queries = random.sample(
        queries,
        args.num_queries,
    )

    print()
    print("Selected queries:")

    for i, query in enumerate(selected_queries):
        print(f"[{i:02d}] {query}")

    Path(args.query_output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(args.query_output, "w", encoding="utf-8") as f:
        json.dump(
            selected_queries,
            f,
            ensure_ascii=False,
            indent=2,
        )

    client = SemanticScholarClient(
        api_key=args.api_key,
    )

    #
    # Stage 1: retrieve seed papers.
    #

    seeds = {}

    print()
    print("=== Stage 1: search seed papers ===")

    for i, query in enumerate(selected_queries):
        print()
        print(
            f"[query {i + 1}/{len(selected_queries)}] "
            f"{query}"
        )

        results = client.search(
            query,
            limit=args.papers_per_query,
        )

        kept = 0

        for paper in results:
            if not valid_paper(paper):
                continue

            arxiv_id = get_arxiv_id(paper)

            if arxiv_id not in seeds:
                seeds[arxiv_id] = paper

            kept += 1

        print(
            f"  returned={len(results)}, "
            f"valid_arxiv={kept}, "
            f"unique_seeds={len(seeds)}"
        )

    if not seeds:
        raise RuntimeError(
            "No usable arXiv seed papers were found."
        )

    #
    # Stage 2: expand one citation hop.
    #

    pool = {
        arxiv_id: convert_paper(paper)
        for arxiv_id, paper in seeds.items()
    }

    print()
    print("=== Stage 2: fetch references ===")

    seed_items = list(seeds.items())

    for i, (seed_arxiv_id, seed_paper) in enumerate(seed_items):
        paper_id = seed_paper.get("paperId")

        if not paper_id:
            continue

        print(
            f"[{i + 1}/{len(seed_items)}] "
            f"{seed_arxiv_id} "
            f"{seed_paper.get('title', '')[:70]}"
        )

        refs = client.references(
            paper_id,
            limit=args.refs_per_paper,
        )

        ref_titles = []

        for ref in refs:
            if not valid_paper(ref):
                continue

            ref_arxiv_id = get_arxiv_id(ref)
            ref_title = ref.get("title")

            if not ref_arxiv_id or not ref_title:
                continue

            # Add reference paper itself to local corpus.
            if ref_arxiv_id not in pool:
                pool[ref_arxiv_id] = convert_paper(ref)

            # The PaperScout expand() path looks up references by title.
            ref_titles.append({
                "title": ref_title,
            })

        pool[seed_arxiv_id]["references"] = ref_titles

        print(
            f"  refs_returned={len(refs)}, "
            f"refs_kept={len(ref_titles)}, "
            f"pool_size={len(pool)}"
        )

    #
    # Stage 3: remove references whose target somehow did not survive.
    #

    available_titles = {
        paper["title"].strip().lower()
        for paper in pool.values()
        if paper["title"]
    }

    for paper in pool.values():
        paper["references"] = [
            ref
            for ref in paper["references"]
            if ref["title"].strip().lower() in available_titles
        ]

    #
    # Stage 4: write JSONL.
    #

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sorted_papers = sorted(
        pool.values(),
        key=lambda p: p["arxiv_id"],
    )

    with output_path.open("w", encoding="utf-8") as f:
        for paper in sorted_papers:
            f.write(
                json.dumps(
                    paper,
                    ensure_ascii=False,
                )
                + "\n"
            )

    num_edges = sum(
        len(paper["references"])
        for paper in sorted_papers
    )

    num_expandable = sum(
        bool(paper["references"])
        for paper in sorted_papers
    )

    print()
    print("=== Done ===")
    print(f"queries:          {len(selected_queries)}")
    print(f"seed papers:      {len(seeds)}")
    print(f"total papers:     {len(sorted_papers)}")
    print(f"citation edges:   {num_edges}")
    print(f"expandable papers:{num_expandable}")
    print(f"paper pool:       {output_path}")
    print(f"queries:          {args.query_output}")


if __name__ == "__main__":
    main()