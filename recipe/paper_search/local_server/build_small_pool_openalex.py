import argparse
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests


API_BASE = "https://api.openalex.org"


class OpenAlexClient:
    def __init__(self, sleep=0.15):
        self.session = requests.Session()
        self.sleep = sleep

    def get(self, endpoint, params=None, retries=8):
        url = f"{API_BASE}{endpoint}"

        for attempt in range(retries):
            try:
                r = self.session.get(
                    url,
                    params=params,
                    timeout=30,
                )

                if r.status_code == 200:
                    time.sleep(self.sleep)
                    return r.json()

                if r.status_code == 429:
                    wait = min(60, 2 ** attempt)
                    print(f"  rate limited, sleep {wait}s")
                    time.sleep(wait)
                    continue

                print(
                    f"  HTTP {r.status_code}: "
                    f"{r.text[:300]}"
                )

            except requests.RequestException as e:
                print(f"  request error: {e}")

            wait = min(30, 2 ** attempt)
            time.sleep(wait)

        return None

    def search(self, query, per_page=10):
        data = self.get(
            "/works",
            params={
                "search": query,
                "filter": "has_abstract:true",
                "per_page": per_page,
            },
        )

        if not data:
            return []

        return data.get("results", [])

    def get_works(self, openalex_ids):
        """
        Batch-fetch works by OpenAlex IDs.
        OpenAlex filter values can be OR'ed with |.
        """
        if not openalex_ids:
            return []

        short_ids = [
            openalex_id.rsplit("/", 1)[-1]
            for openalex_id in openalex_ids
        ]

        results = []

        # Keep batches modest.
        for start in range(0, len(short_ids), 50):
            batch = short_ids[start:start + 50]

            data = self.get(
                "/works",
                params={
                    "filter": "openalex:" + "|".join(batch),
                    "per_page": len(batch),
                },
            )

            if data:
                results.extend(data.get("results", []))

        return results


def reconstruct_abstract(work):
    """
    OpenAlex stores abstracts as an inverted index:
        word -> [positions]
    Reconstruct ordinary text.
    """
    inverted = work.get("abstract_inverted_index")

    if not inverted:
        return None

    positioned_words = []

    for word, positions in inverted.items():
        for position in positions:
            positioned_words.append((position, word))

    positioned_words.sort(key=lambda x: x[0])

    return " ".join(word for _, word in positioned_words)


def normalize_arxiv_id(value):
    if not value:
        return None

    value = str(value).strip()

    # https://arxiv.org/abs/1706.03762
    # https://arxiv.org/pdf/1706.03762
    match = re.search(
        r"arxiv\.org/(?:abs|pdf)/([^/?#]+)",
        value,
        flags=re.IGNORECASE,
    )
    if match:
        value = match.group(1)

    # https://doi.org/10.48550/arXiv.1009.2139
    match = re.search(
        r"10\.48550/arxiv\.([^/?#]+)",
        value,
        flags=re.IGNORECASE,
    )
    if match:
        value = match.group(1)

    if value.lower().startswith("arxiv:"):
        value = value.split(":", 1)[1]

    if value.endswith(".pdf"):
        value = value[:-4]

    value = re.sub(r"v\d+$", "", value)

    # Final sanity check:
    # modern IDs: 1706.03762
    # old IDs: cs/0601001, hep-th/9901001
    if not re.fullmatch(
        r"(?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})",
        value,
    ):
        return None

    return value


def get_arxiv_id(work):
    """
    Try all available locations and IDs rather than assuming that
    the primary location is arXiv.
    """
    candidates = []

    ids = work.get("ids") or {}

    for value in ids.values():
        if value:
            candidates.append(str(value))

    primary = work.get("primary_location") or {}

    if primary.get("landing_page_url"):
        candidates.append(primary["landing_page_url"])

    if primary.get("pdf_url"):
        candidates.append(primary["pdf_url"])

    for location in work.get("locations") or []:
        if location.get("landing_page_url"):
            candidates.append(location["landing_page_url"])

        if location.get("pdf_url"):
            candidates.append(location["pdf_url"])

    for value in candidates:
        if "arxiv" in value.lower():
            return normalize_arxiv_id(value)

    return None


def valid_paper(work):
    return bool(
        work
        and work.get("id")
        and get_arxiv_id(work)
        and work.get("title")
        and reconstruct_abstract(work)
    )


def convert_paper(work):
    authorships = work.get("authorships") or []

    authors = []

    for authorship in authorships:
        author = authorship.get("author") or {}
        name = author.get("display_name")

        if name:
            authors.append(name)

    return {
        "arxiv_id": get_arxiv_id(work),
        "title": work.get("title") or "",
        "abstract": reconstruct_abstract(work) or "",
        "categories": "",
        "authors": ", ".join(authors),
        "update_date": str(work.get("publication_year") or ""),
        "references": [],
    }


def extract_query(prompt):
    try:
        messages = list(prompt)
    except (TypeError, ValueError):
        return None

    for message in messages:
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
        ):
            content = message.get("content")

            if content:
                return str(content).strip()

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(
            Path.home() / "data/pasa/train.parquet"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "recipe/paper_search/local_server/"
            "papers.jsonl"
        ),
    )

    parser.add_argument(
        "--query-output",
        default=(
            "recipe/paper_search/local_server/"
            "queries.json"
        ),
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

    selected_queries = random.sample(
        queries,
        min(args.num_queries, len(queries)),
    )

    print()
    print("Selected queries:")

    for i, query in enumerate(selected_queries):
        print(f"[{i:02d}] {query}")

    query_path = Path(args.query_output)
    query_path.parent.mkdir(parents=True, exist_ok=True)

    with query_path.open("w", encoding="utf-8") as f:
        json.dump(
            selected_queries,
            f,
            ensure_ascii=False,
            indent=2,
        )

    client = OpenAlexClient()

    #
    # Stage 1: Search seed papers.
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
            per_page=args.papers_per_query,
        )

        kept = 0

        for work in results:
            if not valid_paper(work):
                continue

            arxiv_id = get_arxiv_id(work)

            if arxiv_id not in seeds:
                seeds[arxiv_id] = work

            kept += 1

        print(
            f"  returned={len(results)}, "
            f"valid_arxiv={kept}, "
            f"unique_seeds={len(seeds)}"
        )

    if not seeds:
        raise RuntimeError(
            "No usable arXiv seed papers found."
        )

    pool = {
        arxiv_id: convert_paper(work)
        for arxiv_id, work in seeds.items()
    }

    #
    # Stage 2: Get one citation hop.
    #

    print()
    print("=== Stage 2: fetch references ===")

    seed_items = list(seeds.items())

    for i, (seed_arxiv_id, seed_work) in enumerate(seed_items):
        print(
            f"[{i + 1}/{len(seed_items)}] "
            f"{seed_arxiv_id} "
            f"{seed_work.get('title', '')[:70]}"
        )

        reference_ids = (
            seed_work.get("referenced_works") or []
        )

        # Fetch a larger candidate set because many references
        # are not arXiv papers or do not have abstracts.
        candidate_reference_ids = reference_ids[
            :max(args.refs_per_paper * 10, 30)
        ]

        reference_works = client.get_works(
            candidate_reference_ids
        )

        ref_titles = []

        for ref_work in reference_works:

            if len(ref_titles) >= args.refs_per_paper:
                break

            if not valid_paper(ref_work):
                continue

            ref_arxiv_id = get_arxiv_id(ref_work)
            ref_title = ref_work.get("title")

            if not ref_arxiv_id or not ref_title:
                continue

            if ref_arxiv_id not in pool:
                pool[ref_arxiv_id] = convert_paper(ref_work)

            ref_titles.append({
                "title": ref_title,
            })

        pool[seed_arxiv_id]["references"] = ref_titles

        print(
            f"  reference_ids={len(reference_ids)}, "
            f"refs_kept={len(ref_titles)}, "
            f"pool_size={len(pool)}"
        )

    #
    # Stage 3: Sanity-clean citation edges.
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
            if ref["title"].strip().lower()
            in available_titles
        ]

    #
    # Stage 4: Write JSONL.
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

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
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
    print(f"queries:           {len(selected_queries)}")
    print(f"seed papers:       {len(seeds)}")
    print(f"total papers:      {len(sorted_papers)}")
    print(f"citation edges:    {num_edges}")
    print(f"expandable papers: {num_expandable}")
    print(f"paper pool:        {output_path}")
    print(f"queries:           {query_path}")


if __name__ == "__main__":
    main()