#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

REQUIRED = ("treatment", "conversion")
OPTIONAL_BIN = ("visit", "exposure")
FEATURES = tuple(f"f{i}" for i in range(12))

def _strong_cast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in REQUIRED + OPTIONAL_BIN:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")
    for c in FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32").fillna(0.0)
    return df

def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    bad = set(np.unique(df["treatment"])) - {0, 1}
    if bad:
        raise ValueError(f"'treatment' must be binary {{0,1}}; got {sorted(bad)}")

def load_criteo(
    path: Optional[Path] = None,
    *,
    columns: Optional[Iterable[str]] = None,
    n_rows: Optional[int] = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Load parquet/csv; optionally downsample rows deterministically."""
    path = Path(path) if path else Path("data/criteo-uplift-v2.1.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")

    usecols = list(columns) if columns else None
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, columns=usecols) if usecols else pd.read_parquet(path)
    else:
        df = pd.read_csv(path, usecols=usecols)

    if n_rows is not None and len(df) > n_rows:
        df = df.sample(n_rows, random_state=seed)

    df = _strong_cast(df)
    _validate_schema(df)
    return df

def load_and_validate(path: Optional[Path] = None, n_rows: Optional[int] = None) -> pd.DataFrame:
    """Convenience wrapper with default schema + optional row cap."""
    cols = list(REQUIRED) + list(OPTIONAL_BIN) + list(FEATURES)
    return load_criteo(path, columns=cols, n_rows=n_rows)

def make_demo_df(n: int = 100000, *, p0: float = 0.02, uplift: float = 0.15, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, size=n, dtype=np.int8)
    logits_base = np.log(p0/(1-p0))
    X = rng.normal(0, 1, size=(n, 12)).astype("float32")
    beta = np.array([0.05, -0.1, 0.08, 0.02, 0.0, 0.0, 0.04, 0.0, 0.0, 0.03, -0.02, 0.0])
    lin = logits_base + (X @ beta) * 0.25 + t * np.log(1 + uplift)
    p = 1/(1+np.exp(-lin))
    y = rng.binomial(1, p).astype("int8")
    df = pd.DataFrame({"treatment": t, "conversion": y})
    for j in range(12): df[f"f{j}"] = X[:, j]
    return _strong_cast(df)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Load/validate Criteo dataset or generate a demo slice.")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--n-rows", type=int, default=None)
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    if args.demo:
        df = make_demo_df(20000)
    else:
        df = load_and_validate(Path(args.data) if args.data else None, n_rows=args.n_rows)
    print(df.head())
    vc = df["treatment"].value_counts()
    print(f"[data_loader] Loaded {len(df):,} rows. Split: 0={int(vc.get(0,0)):,} 1={int(vc.get(1,0)):,}")
