#!/usr/bin/env python3
"""Create or validate the deterministic 200-train/50-validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def normalize(value: object) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_dir.expanduser()
    source_path = root / "train.parquet"
    test_path = root / "test.parquet"
    outputs = [root / "train_200.parquet", root / "val_50.parquet", root / "trainval_250.parquet"]
    if not source_path.is_file() or not test_path.is_file():
        raise SystemExit(f"Missing {source_path} or {test_path}")

    if all(path.is_file() for path in outputs):
        train = pd.read_parquet(outputs[0])
        val = pd.read_parquet(outputs[1])
        if len(train) != 200 or len(val) != 50:
            raise SystemExit("Existing split has wrong row counts; move it aside and rerun")
        print("Keeping existing valid train_200/val_50 split.")
    else:
        full = pd.read_parquet(source_path)
        test = pd.read_parquet(test_path)
        test_prompts = {normalize(item) for item in test["prompt"]}
        eligible = full[[normalize(item) not in test_prompts for item in full["prompt"]]].copy()
        eligible["_prompt_key"] = [normalize(item) for item in eligible["prompt"]]
        eligible = eligible.drop_duplicates("_prompt_key").drop(columns="_prompt_key")
        sampled = eligible.sample(n=250, random_state=42).reset_index(drop=True)
        train = sampled.iloc[:200].reset_index(drop=True)
        val = sampled.iloc[200:].reset_index(drop=True)
        train.to_parquet(outputs[0], index=False)
        val.to_parquet(outputs[1], index=False)
        sampled.to_parquet(outputs[2], index=False)
        print("Created deterministic train_200/val_50 split (seed=42).")

    train_keys = {normalize(item) for item in train["prompt"]}
    val_keys = {normalize(item) for item in val["prompt"]}
    test = pd.read_parquet(test_path)
    test_keys = {normalize(item) for item in test["prompt"]}
    print(f"train rows: {len(train)}")
    print(f"val rows: {len(val)}")
    print(f"train-val overlap: {len(train_keys & val_keys)}")
    print(f"train-test overlap: {len(train_keys & test_keys)}")
    print(f"val-test overlap: {len(val_keys & test_keys)}")
    if train_keys & val_keys or train_keys & test_keys or val_keys & test_keys:
        raise SystemExit("Prompt leakage detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
