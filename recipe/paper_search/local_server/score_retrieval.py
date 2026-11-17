#!/usr/bin/env python3
"""Score ranked PaperScout retrieval outputs against RealScholarQuery gold IDs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean


def normalize(value: object) -> str:
    text = re.sub(r"^arxiv:\s*", "", str(value or "").strip(), flags=re.IGNORECASE)
    return re.sub(r"v\d+$", "", text)


def unique_ranked_ids(items: list[dict]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        arxiv_id = normalize(item.get("arxiv_id"))
        if arxiv_id and arxiv_id not in seen:
            seen.add(arxiv_id)
            result.append(arxiv_id)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold_rows = [
        json.loads(line)
        for line in args.gold.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(predictions) != len(gold_rows):
        raise SystemExit(f"Row mismatch: predictions={len(predictions)}, gold={len(gold_rows)}")
    if not predictions or "retrieved_papers" not in predictions[0]:
        raise SystemExit("Predictions do not contain retrieved_papers; apply the formal-eval fix and rerun evaluation")

    ks: list[int | None] = [25, 50, 100, None]
    per_query = []
    for pred, gold in zip(predictions, gold_rows):
        ranked = unique_ranked_ids(pred.get("retrieved_papers") or [])
        gold_ids = {normalize(item) for item in gold.get("answer_arxiv_id", []) if normalize(item)}
        row = {
            "qid": gold.get("qid", ""),
            "gold": len(gold_ids),
            "retrieved": len(ranked),
            "reward": float(pred.get("score", 0.0)),
        }
        for k in ks:
            label = "all" if k is None else str(k)
            selected = ranked if k is None else ranked[:k]
            hits = len(set(selected) & gold_ids)
            precision = hits / len(selected) if selected else 0.0
            recall = hits / len(gold_ids) if gold_ids else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            row[f"hits@{label}"] = hits
            row[f"precision@{label}"] = precision
            row[f"recall@{label}"] = recall
            row[f"f1@{label}"] = f1
        per_query.append(row)

    total_gold = sum(row["gold"] for row in per_query)
    summary = {
        "queries": len(per_query),
        "mean_reward": mean(row["reward"] for row in per_query),
        "mean_retrieved": mean(row["retrieved"] for row in per_query),
    }
    for k in ks:
        label = "all" if k is None else str(k)
        hits = sum(row[f"hits@{label}"] for row in per_query)
        summary[f"macro_precision@{label}"] = mean(row[f"precision@{label}"] for row in per_query)
        summary[f"macro_recall@{label}"] = mean(row[f"recall@{label}"] for row in per_query)
        summary[f"macro_f1@{label}"] = mean(row[f"f1@{label}"] for row in per_query)
        summary[f"micro_recall@{label}"] = hits / total_gold if total_gold else 0.0

    payload = {"summary": summary, "per_query": per_query}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
