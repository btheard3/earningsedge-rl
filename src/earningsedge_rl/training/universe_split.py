from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UniverseSplit:
    train: List[str]
    test: List[str]


def split_universe(
    universe_path: str = "data/processed/universe_top200.csv",
    test_frac: float = 0.2,
    seed: int = 42,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
) -> UniverseSplit:
    """
    Deterministic symbol split for generalization testing.

    - Reads universe CSV with column `symbol`
    - Shuffles with seed
    - Splits into disjoint train/test lists
    - Optional caps max_train/max_test for faster experiments
    """
    uni = pd.read_csv(universe_path)
    symbols = uni["symbol"].astype(str).dropna().unique().tolist()
    symbols = [s.strip() for s in symbols if s.strip()]

    rng = np.random.default_rng(seed)
    rng.shuffle(symbols)

    n = len(symbols)
    n_test = max(1, int(round(n * test_frac)))
    test = symbols[:n_test]
    train = symbols[n_test:]

    if max_train is not None:
        train = train[: max_train]
    if max_test is not None:
        test = test[: max_test]

    # safety checks
    train_set, test_set = set(train), set(test)
    overlap = train_set.intersection(test_set)
    if overlap:
        raise ValueError(f"Train/Test overlap detected: {sorted(list(overlap))[:10]}")

    return UniverseSplit(train=train, test=test)


def save_split(split: UniverseSplit, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"train": split.train, "test": split.test}, f, indent=2)


def load_split(path: str) -> UniverseSplit:
    with open(path, "r") as f:
        obj = json.load(f)
    return UniverseSplit(train=list(obj["train"]), test=list(obj["test"]))
